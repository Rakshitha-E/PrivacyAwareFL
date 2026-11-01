"""
Privacy-Aware Federated Learning for Collaborative Prediction
============================================================

MVP features:
1. Four simulated clients.
2. UCI Cleveland binary classification dataset.
3. Non-IID data distribution without discarding samples.
4. Local logistic regression training.
5. Federated Averaging (FedAvg).
6. Simulated client dropout.
7. A separate client-level DP-FedAvg experiment.
8. Accuracy, precision, recall and F1-score.
9. Local-only, centralized, FedAvg and DP-FedAvg comparison.
10. CSV reports, performance graphs and saved models.

Note:
This is a simulation. Secure aggregation is not implemented. The RDP result
is a conditional theoretical bound for the ideal Gaussian mechanism and
fixed, preprocessed client shards; the NumPy noise source and centralized
preprocessing do not provide a production or end-to-end privacy guarantee.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zipfile
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

# Client-level DP parameters. NUM_ROUNDS above controls the accountant's
# composition horizon as well as both federated experiments.
DP_CLIPPING_NORM = 1.0
DP_NOISE_MULTIPLIER = 2.0
DP_DELTA = 1e-5
DP_NOISE_MULTIPLIERS = (0.5, 1.0, 2.0, 3.0)
DP_EXPERIMENT_REPEATS = 3
SCALABILITY_CLIENT_COUNTS = (2, 4, 6, 8)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
HEART_FEATURE_NAMES = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
)


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


def predict_proba_from_raw(
    X_raw,
    model_path=None
):
    """Return class-1 probabilities for raw features in the model's saved order."""

    if model_path is None:
        run_models = sorted(
            RESULTS_DIR.glob("run_*/global_model.npz"),
            key=lambda path: path.stat().st_mtime
        )
        model_path = (
            run_models[-1]
            if run_models
            else RESULTS_DIR / "global_model.npz"
        )

    X_raw = np.asarray(X_raw, dtype=float)
    is_single_record = X_raw.ndim == 1

    if is_single_record:
        X_raw = X_raw.reshape(1, -1)

    if X_raw.ndim != 2:
        raise ValueError("Raw features must be a one- or two-dimensional array.")

    if np.isinf(X_raw).any():
        raise ValueError("Raw features must not contain infinite values.")

    with np.load(model_path) as model:
        feature_names = tuple(
            model["feature_names"].astype(str)
        ) if "feature_names" in model.files else HEART_FEATURE_NAMES
        if X_raw.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected records with {len(feature_names)} features "
                f"in this order: {', '.join(feature_names)}."
            )

        if not {"weights", "bias", "scaler_mean", "scaler_scale"}.issubset(
            model.files
        ):
            raise ValueError(
                "Model artifact is missing weights or preprocessing data."
            )

        for name in ("weights", "scaler_mean", "scaler_scale"):
            if model[name].shape != (len(feature_names),):
                raise ValueError(
                    f"Model artifact field {name!r} does not match its features."
                )

        if np.isnan(X_raw).any():
            if "imputation_medians" not in model.files:
                raise ValueError(
                    "Model artifact is missing training imputation values. "
                    "Run main.py to regenerate it before scoring missing data."
                )
            if model["imputation_medians"].shape != (len(feature_names),):
                raise ValueError(
                    "Model artifact imputation values do not match its features."
                )
            X_raw = np.where(
                np.isnan(X_raw),
                model["imputation_medians"],
                X_raw
            )

        if np.any(model["scaler_scale"] <= 0):
            raise ValueError("Model artifact contains invalid scaling values.")

        X_scaled = (
            X_raw - model["scaler_mean"]
        ) / model["scaler_scale"]
        probabilities = predict_proba(
            X_scaled,
            model["weights"],
            float(model["bias"][0])
        )

    return float(probabilities[0]) if is_single_record else probabilities


def predict_from_raw(X_raw, model_path=None):
    """Return disease/no-disease labels using the model's 0.5 threshold."""

    probabilities = predict_proba_from_raw(X_raw, model_path=model_path)
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return int(predictions) if predictions.ndim == 0 else predictions


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

    if any(sample_count <= 0 for _, _, sample_count in client_updates):
        raise ValueError("Every participating client must have samples.")

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
        if client_weights.shape != global_weights.shape:
            raise ValueError("All client weight vectors must have equal shape.")

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


def aggregate_dp_client_updates(
    client_updates,
    global_weights,
    global_bias,
    clipping_norm,
    noise_multiplier,
    rng
):
    """
    Aggregate clipped client model deltas and add calibrated Gaussian noise.

    The adjacency is replace-one-client with a fixed participant roster and
    fixed public client sample counts. The sample-weighted mean therefore has
    L2 sensitivity 2 * clipping_norm * the largest active client weight.
    """

    if not client_updates:
        raise ValueError(
            "At least one client must participate in DP-FedAvg."
        )

    if (
        not np.isfinite(clipping_norm)
        or clipping_norm <= 0
        or not np.isfinite(noise_multiplier)
        or noise_multiplier <= 0
    ):
        raise ValueError(
            "clipping_norm and noise_multiplier must be finite and positive."
        )

    total_samples = sum(update[3] for update in client_updates)

    if total_samples <= 0:
        raise ValueError(
            "Total participating sample count must be positive."
        )

    sample_weights = {}
    for client_id, delta_weights, delta_bias, sample_count in client_updates:
        if sample_count <= 0:
            raise ValueError(
                f"Client {client_id} must have a positive sample count."
            )
        if client_id in sample_weights:
            raise ValueError(
                f"Client {client_id} appears more than once in an update round."
            )
        sample_weights[client_id] = sample_count / total_samples

    max_sample_weight = max(sample_weights.values())
    sensitivity = 2.0 * clipping_norm * max_sample_weight
    noise_std = noise_multiplier * sensitivity

    aggregate_delta = np.zeros(len(global_weights) + 1, dtype=float)

    for client_id, delta_weights, delta_bias, sample_count in client_updates:
        if np.asarray(delta_weights).shape != np.asarray(global_weights).shape:
            raise ValueError(
                f"Client {client_id} update shape does not match the model."
            )
        update = np.concatenate((
            np.asarray(delta_weights, dtype=float),
            np.array([delta_bias], dtype=float)
        ))
        update_norm = float(np.linalg.norm(update))
        clipping_factor = min(
            1.0,
            clipping_norm / max(update_norm, np.finfo(float).tiny)
        )
        clipped_update = clipping_factor * update
        aggregate_delta += sample_weights[client_id] * clipped_update

    aggregate_delta += rng.normal(
        loc=0.0,
        scale=noise_std,
        size=aggregate_delta.shape
    )

    updated_parameters = np.concatenate((
        np.asarray(global_weights, dtype=float),
        np.array([global_bias], dtype=float)
    )) + aggregate_delta

    return (
        updated_parameters[:-1],
        float(updated_parameters[-1]),
        sample_weights,
        sensitivity,
        noise_std
    )


def rdp_to_epsilon(rdp, orders, delta):
    """Convert composed Gaussian RDP values to an (epsilon, delta) bound."""

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")
    if rdp.shape != orders.shape or np.any(orders <= 1):
        raise ValueError("RDP values and orders must have matching valid shapes.")

    if np.all(rdp == 0):
        return 0.0

    epsilon_bounds = (
        rdp + np.log(1.0 / delta) / (orders - 1.0)
    )
    return max(0.0, float(np.min(epsilon_bounds)))


# ============================================================
# DATASET GENERATION
# ============================================================

# ============================================================
# REAL DATASET: UCI HEART DISEASE
# ============================================================

def create_dataset():
    """
    Load the UCI Cleveland Heart Disease dataset from a local file or ZIP.

    Put heart+disease.zip in the project root (same folder as main.py),
    OR extract processed.cleveland.data into the project root or data/ folder.
    The target is converted from 0..4 to binary:
      0 = no heart disease
      1 = heart disease present (original values 1, 2, 3, or 4)
    """

    print("\nLoading UCI Cleveland Heart Disease dataset...")

    column_names = [*HEART_FEATURE_NAMES, "target"]

    dataset_path = None
    candidate_paths = [
        BASE_DIR / "processed.cleveland.data",
        BASE_DIR / "data" / "processed.cleveland.data",
        BASE_DIR / "heart-disease" / "processed.cleveland.data",
    ]

    for path in candidate_paths:
        if path.exists():
            dataset_path = path
            break

    if dataset_path is not None:
        df = pd.read_csv(
            dataset_path,
            header=None,
            names=column_names,
            na_values="?"
        )
    else:
        # Also support reading the official dataset directly from its ZIP.
        zip_path = BASE_DIR / "heart+disease.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                "Heart-disease dataset not found. Put heart+disease.zip "
                "beside main.py, or put processed.cleveland.data in the "
                "project root or a data/ folder."
            )

        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open("processed.cleveland.data") as data_file:
                df = pd.read_csv(
                    data_file,
                    header=None,
                    names=column_names,
                    na_values="?"
                )

    # Convert all columns to numeric; '?' values become NaN.
    for column in column_names:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows without a valid target. Impute feature missing values
    # using training-set medians below to avoid test-data leakage.
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)

    X_df = df.drop(columns=["target"])
    y = df["target"].to_numpy(dtype=int)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Fit imputation values on training data only.
    train_medians = X_train_df.median()
    X_train_df = X_train_df.fillna(train_medians)
    X_test_df = X_test_df.fillna(train_medians)

    # Fit scaler on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    print(f"Dataset rows: {len(df)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(
        "Target labels: 0 = no heart disease, "
        "1 = heart disease present"
    )
    print(
        f"Training class counts: "
        f"no disease={int(np.sum(y_train == 0))}, "
        f"disease={int(np.sum(y_train == 1))}"
    )
    print(
        f"Testing class counts: "
        f"no disease={int(np.sum(y_test == 0))}, "
        f"disease={int(np.sum(y_test == 1))}"
    )

    return (
        X_train.astype(float),
        X_test.astype(float),
        y_train.astype(int),
        y_test.astype(int),
        scaler.mean_.astype(float),
        scaler.scale_.astype(float),
        train_medians.to_numpy(dtype=float)
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


def train_federated_model(
    clients,
    X_test,
    y_test,
    num_rounds=NUM_ROUNDS,
    dropout_round=None,
    dropout_client_id=None,
    verbose=True
):
    """Train FedAvg, optionally omitting a client at one fixed round."""

    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    if not clients:
        raise ValueError("At least one client is required.")

    first_client = next(iter(clients.values()))
    weights, bias = initialize_model(first_client[0].shape[1])
    history = []

    for round_num in range(1, num_rounds + 1):
        if verbose:
            print(f"\n========== ROUND {round_num}/{num_rounds} ==========")

        client_updates = []
        for client_id, (X_client, y_client) in clients.items():
            if (
                dropout_round == round_num
                and dropout_client_id == client_id
            ):
                if verbose:
                    print(
                        f"Client {client_id} disconnected. "
                        "Skipping this round."
                    )
                continue

            local_weights, local_bias = train_local_model(
                X_client,
                y_client,
                weights,
                bias,
                seed=RANDOM_SEED + round_num * 100 + client_id
            )
            client_updates.append((
                local_weights,
                local_bias,
                len(y_client)
            ))

        if client_updates:
            weights, bias = fedavg(client_updates)

        metrics = calculate_metrics(X_test, y_test, weights, bias)
        history.append({
            "round": round_num,
            "active_clients": len(client_updates),
            **metrics
        })

        if verbose:
            print(f"Active clients: {len(client_updates)}")
            for name in ("accuracy", "precision", "recall", "f1_score"):
                print(f"{name.replace('_', ' ').title()}: {metrics[name]:.4f}")

    return weights, bias, history


def train_dp_federated_model(
    clients,
    X_test,
    y_test,
    num_rounds=NUM_ROUNDS,
    clipping_norm=DP_CLIPPING_NORM,
    noise_multiplier=DP_NOISE_MULTIPLIER,
    delta=DP_DELTA,
    dropout_round=DROPOUT_ROUND,
    dropout_client_id=DROPOUT_CLIENT_ID,
    verbose=True
):
    """
    Train a separate sample-weighted DP-FedAvg model.

    The accountant uses replace-one-client adjacency, fixed public client
    sample counts, a data-independent participation/dropout schedule, and no
    client subsampling. It composes the Gaussian RDP cost per client using
    that client's actual coefficient in each participating round.
    """

    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive.")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    rdp_orders = np.concatenate((
        np.arange(1.25, 2.0, 0.25),
        np.arange(2.0, 257.0)
    ))
    client_rdp = {
        client_id: np.zeros_like(rdp_orders)
        for client_id in clients
    }
    participation_counts = {
        client_id: 0
        for client_id in clients
    }

    # Do not seed the DP noise: a publicly known fixed seed would make the
    # released noise reproducible rather than private.
    noise_rng = np.random.default_rng()
    if not clients:
        raise ValueError("At least one client is required.")

    weights, bias = initialize_model(next(iter(clients.values()))[0].shape[1])
    history = []

    for round_num in range(1, num_rounds + 1):
        if verbose:
            print(
                f"\n========== DP ROUND {round_num}/{num_rounds} =========="
            )

        client_updates = []

        for client_id, (X_client, y_client) in clients.items():
            if (
                round_num == dropout_round
                and client_id == dropout_client_id
            ):
                if verbose:
                    print(
                        f"Client {client_id} disconnected. "
                        "Skipping this DP round."
                    )
                continue

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
            client_updates.append((
                client_id,
                local_weights - weights,
                local_bias - bias,
                len(y_client)
            ))

        (
            weights,
            bias,
            sample_weights,
            sensitivity,
            noise_std
        ) = aggregate_dp_client_updates(
            client_updates,
            weights,
            bias,
            clipping_norm,
            noise_multiplier,
            noise_rng
        )

        max_sample_weight = max(sample_weights.values())
        for client_id, sample_weight in sample_weights.items():
            relative_sensitivity = sample_weight / max_sample_weight
            client_rdp[client_id] += (
                rdp_orders
                * relative_sensitivity ** 2
                / (2.0 * noise_multiplier ** 2)
            )
            participation_counts[client_id] += 1

        per_client_epsilon = {
            client_id: rdp_to_epsilon(
                rdp_values,
                rdp_orders,
                delta
            )
            for client_id, rdp_values in client_rdp.items()
        }
        cumulative_epsilon = max(per_client_epsilon.values(), default=0.0)

        metrics = calculate_metrics(
            X_test,
            y_test,
            weights,
            bias
        )
        history.append({
            "round": round_num,
            "active_clients": len(client_updates),
            "clipping_norm": clipping_norm,
            "noise_multiplier": noise_multiplier,
            "sensitivity": sensitivity,
            "noise_std": noise_std,
            **metrics,
            "epsilon": cumulative_epsilon,
            "delta": delta
        })

        if verbose:
            print(f"Active clients: {len(client_updates)}")
            print(f"Clipped-update sensitivity: {sensitivity:.6f}")
            print(f"Gaussian noise standard deviation: {noise_std:.6f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1_score']:.4f}")
            print(
                f"Cumulative max-client privacy: "
                f"epsilon={cumulative_epsilon:.4f}, delta={delta:g}"
            )

    per_client_privacy = [
        {
            "client_id": client_id,
            "participating_rounds": participation_counts[client_id],
            "epsilon": rdp_to_epsilon(
                client_rdp[client_id],
                rdp_orders,
                delta
            ),
            "delta": delta
        }
        for client_id in sorted(clients)
    ]
    privacy_summary = {
        "epsilon": max(
            (row["epsilon"] for row in per_client_privacy),
            default=0.0
        ),
        "delta": delta,
        "per_client": per_client_privacy,
        "rdp_orders": rdp_orders,
        "client_rdp": client_rdp
    }

    return weights, bias, history, privacy_summary


def create_run_directory():
    """Create a fresh results directory without replacing previous outputs."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def save_model_artifact(
    path,
    weights,
    bias,
    scaler_mean,
    scaler_scale,
    imputation_medians,
    **metadata
):
    np.savez(
        path,
        weights=weights,
        bias=np.array([bias]),
        feature_names=np.array(HEART_FEATURE_NAMES),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        imputation_medians=imputation_medians,
        **metadata
    )


def plot_metric_grid(data, category, title, output_path):
    metrics = ("accuracy", "precision", "recall", "f1_score")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, metrics):
        axis.bar(data[category].astype(str), data[metric])
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", labelrotation=20)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_client_scalability(data, output_path):
    """Plot metric changes versus the standard four-client FedAvg run."""

    reference = data.loc[data["num_clients"] == NUM_CLIENTS]
    if len(reference) != 1:
        raise ValueError(
            f"Scalability results must contain exactly one {NUM_CLIENTS}-client row."
        )

    client_counts = data["num_clients"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5))
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        changes = (
            data[metric].to_numpy(dtype=float)
            - float(reference.iloc[0][metric])
        ) * 100
        axis.plot(
            client_counts,
            changes,
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title(),
        )

    all_changes = np.concatenate([
        (
            data[metric].to_numpy(dtype=float)
            - float(reference.iloc[0][metric])
        ) * 100
        for metric in ("accuracy", "precision", "recall", "f1_score")
    ])
    extent = max(float(np.max(np.abs(all_changes))), 0.01)
    padding = max(extent * 0.2, 0.01)
    axis.set_ylim(float(np.min(all_changes)) - padding, float(np.max(all_changes)) + padding)
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(client_counts)
    axis.set_xlabel("Number of simulated clients")
    axis.set_ylabel("Change from 4 clients (percentage points)")
    axis.set_title("Client-count metric changes vs. 4-client FedAvg")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_additional_experiments(
    clients,
    X_train,
    y_train,
    X_test,
    y_test,
    fedavg_metrics,
    dp_weights,
    dp_bias,
    privacy_summary,
    scaler_mean,
    scaler_scale,
    imputation_medians,
    output_dir
):
    """Run noise, client-count, and dropout comparisons on the fixed test set."""

    privacy_rows = [{
        "approach": "FedAvg (no DP)",
        "noise_multiplier": 0.0,
        "epsilon": np.nan,
        "delta": np.nan,
        "repetitions": 1,
        **fedavg_metrics
    }]
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        privacy_rows[0][f"{metric}_std"] = 0.0

    privacy_run_rows = []
    accounting_rows = []
    joint_client_rdp = {
        client_id: np.zeros_like(privacy_summary["rdp_orders"])
        for client_id in clients
    }
    joint_participations = {client_id: 0 for client_id in clients}

    for multiplier in DP_NOISE_MULTIPLIERS:
        repeat_metrics = []
        summary = None
        first_model = None
        for repeat in range(1, DP_EXPERIMENT_REPEATS + 1):
            if (
                multiplier == DP_NOISE_MULTIPLIER
                and repeat == 1
            ):
                model_weights, model_bias = dp_weights, dp_bias
                repeat_summary = privacy_summary
            else:
                (
                    model_weights,
                    model_bias,
                    _,
                    repeat_summary
                ) = train_dp_federated_model(
                    clients,
                    X_test,
                    y_test,
                    noise_multiplier=multiplier,
                    verbose=False
                )

            if summary is None:
                summary = repeat_summary
                first_model = (model_weights, model_bias)

            for client_id, rdp_values in repeat_summary["client_rdp"].items():
                joint_client_rdp[client_id] += rdp_values
            for row in repeat_summary["per_client"]:
                joint_participations[row["client_id"]] += (
                    row["participating_rounds"]
                )

            metrics = calculate_metrics(X_test, y_test, model_weights, model_bias)
            repeat_metrics.append(metrics)
            privacy_run_rows.append({
                "noise_multiplier": multiplier,
                "repeat": repeat,
                "epsilon": repeat_summary["epsilon"],
                "delta": repeat_summary["delta"],
                **metrics
            })

        mean_metrics = {
            metric: float(np.mean([row[metric] for row in repeat_metrics]))
            for metric in ("accuracy", "precision", "recall", "f1_score")
        }
        std_metrics = {
            f"{metric}_std": float(
                np.std([row[metric] for row in repeat_metrics], ddof=1)
            )
            for metric in ("accuracy", "precision", "recall", "f1_score")
        }
        privacy_rows.append({
            "approach": f"DP-FedAvg (noise={multiplier:g})",
            "noise_multiplier": multiplier,
            "epsilon": summary["epsilon"],
            "delta": summary["delta"],
            "repetitions": DP_EXPERIMENT_REPEATS,
            **mean_metrics,
            **std_metrics
        })
        accounting_rows.extend(
            {
                "release_scope": "per_model",
                "noise_multiplier": multiplier,
                **row
            }
            for row in summary["per_client"]
        )

        model_weights, model_bias = first_model
        multiplier_label = str(multiplier).replace(".", "_")
        save_model_artifact(
            output_dir / f"dp_global_model_noise_{multiplier_label}.npz",
            model_weights,
            model_bias,
            scaler_mean,
            scaler_scale,
            imputation_medians,
            clipping_norm=np.array([DP_CLIPPING_NORM]),
            noise_multiplier=np.array([multiplier]),
            privacy_epsilon=np.array([summary["epsilon"]]),
            privacy_delta=np.array([summary["delta"]])
        )

    joint_per_client = [
        {
            "release_scope": "joint_sweep",
            "noise_multiplier": np.nan,
            "client_id": client_id,
            "participating_rounds": joint_participations[client_id],
            "epsilon": rdp_to_epsilon(
                joint_client_rdp[client_id],
                privacy_summary["rdp_orders"],
                DP_DELTA
            ),
            "delta": DP_DELTA
        }
        for client_id in sorted(clients)
    ]
    joint_sweep_epsilon = max(
        (row["epsilon"] for row in joint_per_client),
        default=0.0
    )
    accounting_rows.extend(joint_per_client)

    privacy_frame = pd.DataFrame(privacy_rows).sort_values("noise_multiplier")
    dp_privacy_frame = privacy_frame[
        privacy_frame["noise_multiplier"] > 0
    ]
    privacy_frame["joint_sweep_epsilon"] = np.where(
        privacy_frame["noise_multiplier"] > 0,
        joint_sweep_epsilon,
        np.nan
    )
    privacy_frame.to_csv(output_dir / "privacy_utility_metrics.csv", index=False)
    pd.DataFrame(privacy_run_rows).to_csv(
        output_dir / "privacy_utility_runs.csv", index=False
    )
    pd.DataFrame(accounting_rows).to_csv(
        output_dir / "privacy_accounting.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        dp_privacy_frame["noise_multiplier"],
        dp_privacy_frame["epsilon"],
        marker="o"
    )
    axis.set_xlabel("Gaussian noise multiplier")
    axis.set_ylabel("Maximum client epsilon")
    axis.set_title("Privacy Budget vs Noise Multiplier")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "privacy_epsilon_vs_noise.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    dp_accuracy_frame = privacy_frame[
        privacy_frame["noise_multiplier"] > 0
    ]
    axis.plot(
        dp_accuracy_frame["noise_multiplier"],
        dp_accuracy_frame["accuracy"],
        marker="o",
        label="DP-FedAvg mean"
    )
    fedavg_accuracy = float(
        privacy_frame.loc[
            privacy_frame["noise_multiplier"] == 0,
            "accuracy"
        ].iloc[0]
    )
    axis.axhline(
        fedavg_accuracy,
        linestyle="--",
        label="FedAvg (no DP)"
    )
    axis.set_xlabel("Gaussian noise multiplier")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.05)
    axis.set_title("Model Accuracy vs Noise Multiplier")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "privacy_accuracy_vs_noise.png", dpi=150)
    plt.close(figure)

    scalability_rows = []
    four_client_no_dropout = None
    for client_count in SCALABILITY_CLIENT_COUNTS:
        count_clients = (
            clients
            if client_count == NUM_CLIENTS
            else split_clients_non_iid(
                X_train, y_train, client_count
            )
        )
        weights, bias, _ = train_federated_model(
            count_clients,
            X_test,
            y_test,
            dropout_round=None,
            dropout_client_id=None,
            verbose=False
        )
        metrics = calculate_metrics(X_test, y_test, weights, bias)
        scalability_rows.append({
            "num_clients": client_count,
            **metrics
        })
        if client_count == NUM_CLIENTS:
            four_client_no_dropout = metrics

    scalability_frame = pd.DataFrame(scalability_rows)
    scalability_frame.to_csv(
        output_dir / "scalability_metrics.csv", index=False
    )
    plot_client_scalability(
        scalability_frame,
        output_dir / "scalability_comparison.png"
    )

    if four_client_no_dropout is None:
        raise ValueError(
            "SCALABILITY_CLIENT_COUNTS must include NUM_CLIENTS "
            "for the dropout comparison."
        )

    dropout_rows = [
        {
            "scenario": "No dropout",
            "dropped_client": np.nan,
            "dropped_round": np.nan,
            **four_client_no_dropout
        },
        {
            "scenario": f"Client {DROPOUT_CLIENT_ID} dropout",
            "dropped_client": DROPOUT_CLIENT_ID,
            "dropped_round": DROPOUT_ROUND,
            **fedavg_metrics
        }
    ]
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        dropout_rows[0][f"change_{metric}"] = 0.0
        dropout_rows[1][f"change_{metric}"] = (
            dropout_rows[1][metric] - dropout_rows[0][metric]
        )
    dropout_frame = pd.DataFrame(dropout_rows)
    dropout_frame.to_csv(output_dir / "dropout_metrics.csv", index=False)
    plot_metric_grid(
        dropout_frame,
        "scenario",
        "FedAvg Performance With and Without Client Dropout",
        output_dir / "dropout_comparison.png"
    )

    return privacy_frame, scalability_frame, dropout_frame, joint_sweep_epsilon


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    output_dir = create_run_directory()

    # Create dataset.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_mean,
        scaler_scale,
        imputation_medians
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

    # --------------------------------------------------------
    # FEDERATED TRAINING
    # --------------------------------------------------------

    weights, bias, history = train_federated_model(
        clients,
        X_test,
        y_test,
        num_rounds=NUM_ROUNDS,
        dropout_round=DROPOUT_ROUND,
        dropout_client_id=DROPOUT_CLIENT_ID
    )

    # --------------------------------------------------------
    # DIFFERENTIALLY PRIVATE FEDERATED TRAINING
    # --------------------------------------------------------

    (
        dp_weights,
        dp_bias,
        dp_history,
        privacy_summary
    ) = train_dp_federated_model(
        clients,
        X_test,
        y_test,
        num_rounds=NUM_ROUNDS,
        clipping_norm=DP_CLIPPING_NORM,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        delta=DP_DELTA
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
        },

        {
            "approach": "Differentially Private FedAvg",
            **calculate_metrics(
                X_test,
                y_test,
                dp_weights,
                dp_bias
            )
        }

    ])

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    (
        _,
        _,
        _,
        joint_sweep_epsilon
    ) = run_additional_experiments(
        clients,
        X_train,
        y_train,
        X_test,
        y_test,
        federated_metrics,
        dp_weights,
        dp_bias,
        privacy_summary,
        scaler_mean,
        scaler_scale,
        imputation_medians,
        output_dir
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_dir / "federated_metrics.csv",
        index=False
    )

    dp_history_df = pd.DataFrame(dp_history)
    dp_history_df.to_csv(
        output_dir / "dp_federated_metrics.csv",
        index=False
    )

    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # Save the final federated model.
    save_model_artifact(
        output_dir / "global_model.npz",
        weights,
        bias,
        scaler_mean,
        scaler_scale,
        imputation_medians
    )

    # Save the centralized baseline model.
    save_model_artifact(
        output_dir / "centralized_model.npz",
        centralized_weights,
        centralized_bias,
        scaler_mean,
        scaler_scale,
        imputation_medians
    )

    # Save the DP model separately so the original FedAvg artifact remains
    # the standard, non-private baseline.
    save_model_artifact(
        output_dir / "dp_global_model.npz",
        dp_weights,
        dp_bias,
        scaler_mean,
        scaler_scale,
        imputation_medians,
        clipping_norm=np.array([DP_CLIPPING_NORM]),
        noise_multiplier=np.array([DP_NOISE_MULTIPLIER]),
        privacy_epsilon=np.array([privacy_summary["epsilon"]]),
        privacy_delta=np.array([privacy_summary["delta"]])
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
        output_dir / "performance.png",
        dpi=150
    )

    plt.close()

    # Compare all four approaches on the same held-out test set.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(
        axes.flat,
        ["accuracy", "precision", "recall", "f1_score"]
    ):
        axis.bar(
            comparison["approach"],
            comparison[metric],
            color=["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", labelrotation=20)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    figure.suptitle("Model Comparison on the Held-Out Test Set")
    figure.tight_layout()
    figure.savefig(
        output_dir / "model_comparison.png",
        dpi=150
    )
    plt.close(figure)

    raw_probe = X_test[0] * scaler_scale + scaler_mean
    prediction = predict_proba_from_raw(
        raw_probe,
        model_path=output_dir / "global_model.npz"
    )
    if not np.isfinite(prediction) or not 0 <= prediction <= 1:
        raise RuntimeError("Saved-model prediction smoke check failed.")
    predicted_label = predict_from_raw(
        raw_probe,
        model_path=output_dir / "global_model.npz"
    )
    if predicted_label not in (0, 1):
        raise RuntimeError("Saved-model class prediction smoke check failed.")

    # --------------------------------------------------------
    # COMPLETION MESSAGE
    # --------------------------------------------------------

    print("\n========================================")
    print(" EXPERIMENT COMPLETED")
    print("========================================")

    print(
        f"\nResults saved in: {output_dir}"
    )
    print(f"Saved-model prediction smoke check: probability={prediction:.4f}")
    print(f"Saved-model class prediction: {predicted_label}")

    print("\nDIFFERENTIAL PRIVACY ACCOUNTING")
    print(f"Clipping norm: {DP_CLIPPING_NORM}")
    print(f"Noise multiplier: {DP_NOISE_MULTIPLIER}")
    print(f"Federated rounds: {NUM_ROUNDS}")
    print(f"Privacy delta: {privacy_summary['delta']:g}")
    print(f"Calculated max-client epsilon: {privacy_summary['epsilon']:.4f}")
    print(
        "Joint epsilon for all released DP noise-sweep trials: "
        f"{joint_sweep_epsilon:.4f}"
    )
    print(
        "Adjacency: replace-one client, with fixed public client sample "
        "counts and a data-independent participation/dropout schedule."
    )
    print(
        "Accounting: Gaussian RDP composition without subsampling "
        "amplification; epsilon is the maximum across clients."
    )
    print(
        "Scope: conditional on the fixed preprocessed client shards. "
        "Centralized preprocessing statistics are not privatized."
    )

    print("Generated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path.name}")

    print(
        "\nNote: Clients are simulated in one Python process."
    )

    print(
        "Secure aggregation is not implemented."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()