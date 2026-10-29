"""
Privacy-Aware Federated Learning for Collaborative Prediction
============================================================

MVP features:
1. Four simulated clients.
2. Synthetic binary classification dataset.
3. Non-IID data distribution without discarding samples.
4. Local logistic regression training.
5. Federated Averaging (FedAvg).
6. Simulated client dropout.
7. Accuracy, precision, recall and F1-score.
8. Local-only vs centralized vs federated comparison.
9. CSV reports, performance graph and saved models.

Note:
This is a simulation. Differential privacy and secure aggregation
are not implemented.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_CLIENTS = 4
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5

LEARNING_RATE = 0.05
BATCH_SIZE = 32

DROPOUT_ROUND = 4
DROPOUT_CLIENT_ID = 3

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def sigmoid(z):

    z = np.clip(z, -500, 500)

    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, weights, bias):

    return sigmoid(X @ weights + bias)


def predict(X, weights, bias):

    probabilities = predict_proba(
        X,
        weights,
        bias
    )

    return (probabilities >= 0.5).astype(int)


def initialize_model(
    num_features,
    seed=RANDOM_SEED
):

    rng = np.random.default_rng(seed)

    weights = rng.normal(
        0,
        0.01,
        size=num_features
    )

    bias = 0.0

    return weights, bias


# ============================================================
# CLIENT-SIDE TRAINING
# ============================================================

def train_local_model(
    X,
    y,
    weights,
    bias,
    epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    seed=RANDOM_SEED
):

    weights = weights.copy()
    bias = float(bias)

    rng = np.random.default_rng(seed)

    n_samples = len(X)

    if n_samples == 0:

        return weights, bias

    for epoch in range(epochs):

        shuffled_indices = rng.permutation(
            n_samples
        )

        for start in range(
            0,
            n_samples,
            batch_size
        ):

            batch_indices = shuffled_indices[
                start:start + batch_size
            ]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            probabilities = predict_proba(
                X_batch,
                weights,
                bias
            )

            errors = probabilities - y_batch

            grad_weights = (
                X_batch.T @ errors
            ) / len(batch_indices)

            grad_bias = float(
                np.mean(errors)
            )

            weights -= (
                learning_rate * grad_weights
            )

            bias -= (
                learning_rate * grad_bias
            )

    return weights, bias


# ============================================================
# FEDERATED AVERAGING
# ============================================================

def fedavg(client_updates):
    """
    Aggregate client models using sample-weighted FedAvg.

    Each update contains:
        (weights, bias, sample_count)
    """

    if not client_updates:

        raise ValueError(
            "At least one client must participate."
        )

    total_samples = sum(
        sample_count
        for weights, bias, sample_count
        in client_updates
    )

    if total_samples <= 0:

        raise ValueError(
            "Total sample count must be positive."
        )

    global_weights = np.zeros_like(
        client_updates[0][0]
    )

    global_bias = 0.0

    for (
        client_weights,
        client_bias,
        sample_count
    ) in client_updates:

        client_fraction = (
            sample_count / total_samples
        )

        global_weights += (
            client_fraction * client_weights
        )

        global_bias += (
            client_fraction * client_bias
        )

    return global_weights, global_bias


# ============================================================
# DATASET GENERATION
# ============================================================

def create_dataset():

    print("\nGenerating synthetic dataset...")

    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.58, 0.42],
        class_sep=1.0,
        random_state=RANDOM_SEED
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=y
        )
    )

    # Fit the scaler only on training data.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train.astype(float),
        X_test.astype(float),
        y_train.astype(int),
        y_test.astype(int)
    )


# ============================================================
# NON-IID CLIENT DATA DISTRIBUTION
# ============================================================

def split_clients_non_iid(
    X,
    y,
    num_clients=NUM_CLIENTS,
    seed=RANDOM_SEED
):
    """
    Assign every training sample to exactly one client.

    Clients receive different class proportions.
    No samples are intentionally discarded.
    """

    if len(X) != len(y):

        raise ValueError(
            "X and y must contain equal numbers of samples."
        )

    if num_clients < 2:

        raise ValueError(
            "At least two clients are required."
        )

    if len(y) < num_clients:

        raise ValueError(
            "There must be at least one sample per client."
        )

    if not np.array_equal(
        np.unique(y),
        np.array([0, 1])
    ):

        raise ValueError(
            "This function expects binary labels 0 and 1."
        )

    rng = np.random.default_rng(seed)

    # Shuffle indices belonging to each class.
    class_0 = rng.permutation(
        np.where(y == 0)[0]
    )

    class_1 = rng.permutation(
        np.where(y == 1)[0]
    )

    total_samples = len(y)

    # Assign approximately equal total sample counts.
    target_sizes = np.full(
        num_clients,
        total_samples // num_clients,
        dtype=int
    )

    target_sizes[
        :total_samples % num_clients
    ] += 1

    # Desired proportions of class 1.
    # For four clients: 10%, 30%, 70%, and 90%.
    if num_clients == 4:

        ratios = np.array([
            0.10,
            0.30,
            0.70,
            0.90
        ])

    else:

        ratios = np.linspace(
            0.10,
            0.90,
            num_clients
        )

    # Initial allocation of class-1 samples.
    class_1_counts = np.rint(
        target_sizes * ratios
    ).astype(int)

    class_1_counts = np.clip(
        class_1_counts,
        0,
        target_sizes
    )

    # Adjust the allocation to match the actual
    # number of class-1 examples in the dataset.
    difference = (
        len(class_1)
        - int(class_1_counts.sum())
    )

    while difference != 0:

        if difference > 0:

            candidates = np.where(
                class_1_counts < target_sizes
            )[0]

            if len(candidates) == 0:

                raise ValueError(
                    "Cannot allocate all class-1 samples."
                )

            # Add a sample to the client with the
            # lowest current class-1 proportion.
            idx = min(
                candidates,
                key=lambda j: (
                    class_1_counts[j]
                    / target_sizes[j]
                )
            )

            class_1_counts[idx] += 1

            difference -= 1

        else:

            candidates = np.where(
                class_1_counts > 0
            )[0]

            if len(candidates) == 0:

                raise ValueError(
                    "Cannot reduce class-1 allocation."
                )

            # Remove a sample from the client with
            # the highest current class-1 proportion.
            idx = max(
                candidates,
                key=lambda j: (
                    class_1_counts[j]
                    / target_sizes[j]
                )
            )

            class_1_counts[idx] -= 1

            difference += 1

    class_0_counts = (
        target_sizes - class_1_counts
    )

    if class_0_counts.sum() != len(class_0):

        raise ValueError(
            "Client allocation does not match class counts."
        )

    clients = {}

    start_0 = 0
    start_1 = 0

    all_assigned_indices = []

    for i in range(num_clients):

        end_0 = (
            start_0 + class_0_counts[i]
        )

        end_1 = (
            start_1 + class_1_counts[i]
        )

        # Assign different portions of each class.
        client_indices = np.concatenate([
            class_0[start_0:end_0],
            class_1[start_1:end_1]
        ])

        rng.shuffle(client_indices)

        clients[i + 1] = (
            X[client_indices],
            y[client_indices]
        )

        all_assigned_indices.extend(
            client_indices.tolist()
        )

        start_0 = end_0
        start_1 = end_1

    # Verify that every training row is assigned
    # exactly once and no row is duplicated.
    if len(all_assigned_indices) != len(y):

        raise ValueError(
            "Some training samples were not assigned."
        )

    if len(set(all_assigned_indices)) != len(y):

        raise ValueError(
            "Duplicate training samples detected."
        )

    print(
        "\nAll training samples assigned successfully."
    )

    return clients


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_metrics(
    X,
    y,
    weights,
    bias
):

    predictions = predict(
        X,
        weights,
        bias
    )

    return {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0
        )

    }


# ============================================================
# LOCAL-ONLY BASELINE
# ============================================================

def evaluate_local_only(
    clients,
    X_test,
    y_test
):

    client_metrics = []

    for client_id, (
        X_client,
        y_client
    ) in clients.items():

        weights, bias = initialize_model(
            X_client.shape[1],
            seed=RANDOM_SEED + client_id
        )

        weights, bias = train_local_model(
            X_client,
            y_client,
            weights,
            bias,
            seed=RANDOM_SEED + client_id
        )

        metrics = calculate_metrics(
            X_test,
            y_test,
            weights,
            bias
        )

        client_metrics.append(metrics)

    average_metrics = {}

    for metric in client_metrics[0]:

        average_metrics[metric] = float(
            np.mean([
                result[metric]
                for result in client_metrics
            ])
        )

    return average_metrics


# ============================================================
# CENTRALIZED BASELINE
# ============================================================

def evaluate_centralized(
    X_train,
    y_train,
    X_test,
    y_test
):

    weights, bias = initialize_model(
        X_train.shape[1]
    )

    weights, bias = train_local_model(
        X_train,
        y_train,
        weights,
        bias
    )

    metrics = calculate_metrics(
        X_test,
        y_test,
        weights,
        bias
    )

    return (
        metrics,
        weights,
        bias
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create dataset.
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = create_dataset()

    # Divide training data among clients.
    clients = split_clients_non_iid(
        X_train,
        y_train
    )

    print("\n========================================")
    print(" PRIVACY-AWARE FEDERATED LEARNING MVP")
    print("========================================")

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Number of clients: {len(clients)}"
    )

    # Display client data distributions.
    print("\nCLIENT DATA DISTRIBUTION")

    for client_id, (
        X_client,
        y_client
    ) in clients.items():

        count_0 = int(
            np.sum(y_client == 0)
        )

        count_1 = int(
            np.sum(y_client == 1)
        )

        proportion = (
            count_1 / len(y_client)
            if len(y_client) > 0
            else 0
        )

        print(
            f"Client {client_id}: "
            f"{len(y_client)} samples | "
            f"Class 0: {count_0} | "
            f"Class 1: {count_1} | "
            f"Class 1 proportion: {proportion:.1%}"
        )

    # Initialize the global model.
    weights, bias = initialize_model(
        X_train.shape[1]
    )

    history = []

    # --------------------------------------------------------
    # FEDERATED TRAINING
    # --------------------------------------------------------

    for round_num in range(
        1,
        NUM_ROUNDS + 1
    ):

        print(
            f"\n========== ROUND {round_num}/{NUM_ROUNDS} =========="
        )

        client_updates = []

        for client_id, (
            X_client,
            y_client
        ) in clients.items():

            # Simulate Client 3 disconnecting in Round 4.
            if (
                round_num == DROPOUT_ROUND
                and client_id == DROPOUT_CLIENT_ID
            ):

                print(
                    f"Client {client_id} disconnected. "
                    "Skipping this round."
                )

                continue

            # Train locally using the current global model.
            local_weights, local_bias = train_local_model(
                X_client,
                y_client,
                weights,
                bias,
                seed=(
                    RANDOM_SEED
                    + round_num * 100
                    + client_id
                )
            )

            # Return model parameters and sample count.
            # Raw training records are not passed to FedAvg.
            client_updates.append(
                (
                    local_weights,
                    local_bias,
                    len(y_client)
                )
            )

        # Aggregate client models.
        if client_updates:

            weights, bias = fedavg(
                client_updates
            )

        # Evaluate the global model.
        metrics = calculate_metrics(
            X_test,
            y_test,
            weights,
            bias
        )

        history.append({
            "round": round_num,
            "active_clients": len(client_updates),
            **metrics
        })

        print(
            f"Active clients: {len(client_updates)}"
        )

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall: {metrics['recall']:.4f}"
        )

        print(
            f"F1-score: {metrics['f1_score']:.4f}"
        )

    # --------------------------------------------------------
    # COMPARE TRAINING APPROACHES
    # --------------------------------------------------------

    print("\n========================================")
    print(" FINAL MODEL COMPARISON")
    print("========================================")

    # Train independent client models.
    local_metrics = evaluate_local_only(
        clients,
        X_test,
        y_test
    )

    # Train a centralized model.
    (
        centralized_metrics,
        centralized_weights,
        centralized_bias
    ) = evaluate_centralized(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Evaluate the final federated model.
    federated_metrics = calculate_metrics(
        X_test,
        y_test,
        weights,
        bias
    )

    comparison = pd.DataFrame([

        {
            "approach": "Local-only",
            **local_metrics
        },

        {
            "approach": "Centralized",
            **centralized_metrics
        },

        {
            "approach": "Federated FedAvg",
            **federated_metrics
        }

    ])

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        RESULTS_DIR / "federated_metrics.csv",
        index=False
    )

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    # Save the final federated model.
    np.savez(
        RESULTS_DIR / "global_model.npz",
        weights=weights,
        bias=np.array([bias])
    )

    # Save the centralized baseline model.
    np.savez(
        RESULTS_DIR / "centralized_model.npz",
        weights=centralized_weights,
        bias=np.array([centralized_bias])
    )

    # --------------------------------------------------------
    # PLOT PERFORMANCE
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:

        plt.plot(
            history_df["round"],
            history_df[metric],
            marker="o",
            label=metric.replace(
                "_",
                " "
            ).title()
        )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Federated Learning Performance Across Rounds"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xticks(
        range(1, NUM_ROUNDS + 1)
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "performance.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # COMPLETION MESSAGE
    # --------------------------------------------------------

    print("\n========================================")
    print(" EXPERIMENT COMPLETED")
    print("========================================")

    print(
        f"\nResults saved in: {RESULTS_DIR}"
    )

    print(
        "1. federated_metrics.csv"
    )

    print(
        "2. model_comparison.csv"
    )

    print(
        "3. performance.png"
    )

    print(
        "4. global_model.npz"
    )

    print(
        "5. centralized_model.npz"
    )

    print(
        "\nNote: Clients are simulated in one Python process."
    )

    print(
        "Differential privacy and secure aggregation "
        "are not implemented."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()