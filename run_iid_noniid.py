"""Run reproducible IID versus Dirichlet Non-IID FedAvg experiments."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import main as project
import train_cardio


DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 10.0)
DEFAULT_SEEDS = (42, 43, 44)
CLIENT_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
)


def create_output_directory(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"iid_noniid_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = results_dir / run_name
    suffix = 1
    while output_dir.exists():
        output_dir = results_dir / f"{run_name}_{suffix}"
        suffix += 1
    output_dir.mkdir()
    return output_dir


def evaluate_with_auc(X, y, weights, bias) -> dict[str, float]:
    metrics = project.calculate_metrics(X, y, weights, bias)
    probabilities = project.predict_proba(X, weights, bias)
    metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    return {name: float(value) for name, value in metrics.items()}


def _client_data(X, y, partitions):
    return {
        client_id: (X[indices], y[indices])
        for client_id, indices in partitions.items()
    }


def _verify_partition(partitions, sample_count):
    assigned = np.concatenate(list(partitions.values()))
    unique_count = int(np.unique(assigned).size)
    if len(assigned) != sample_count or unique_count != sample_count:
        raise RuntimeError("Client partitions lost or duplicated training rows.")
    if not np.array_equal(np.sort(assigned), np.arange(sample_count)):
        raise RuntimeError("Client partitions do not cover each training row once.")
    return {
        "training_rows": sample_count,
        "assigned_rows": len(assigned),
        "unique_assigned_rows": unique_count,
        "duplicate_assignments": len(assigned) - unique_count,
        "missing_training_rows": sample_count - unique_count,
    }


def _client_trial_rows(
    clients,
    X_test,
    y_test,
    method,
    alpha,
    seed,
):
    rows = []
    for client_id, (X_client, y_client) in clients.items():
        weights, bias = project.initialize_model(
            X_client.shape[1],
            seed=seed + client_id,
        )
        weights, bias = project.train_local_model(
            X_client,
            y_client,
            weights,
            bias,
            seed=seed + client_id,
        )
        metrics = evaluate_with_auc(X_test, y_test, weights, bias)
        class_counts = np.bincount(y_client.astype(int), minlength=2)
        rows.append({
            "method": method,
            "alpha": alpha,
            "seed": seed,
            "client_id": client_id,
            "sample_count": len(y_client),
            "class_0_count": int(class_counts[0]),
            "class_1_count": int(class_counts[1]),
            "class_0_fraction": float(class_counts[0] / len(y_client)),
            "class_1_fraction": float(class_counts[1] / len(y_client)),
            **{f"local_{name}": value for name, value in metrics.items()},
        })
    return rows


def _save_performance_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 score",
        "roc_auc": "ROC-AUC",
    }
    treatments = summary.sort_values(["method", "alpha"], na_position="first")
    labels = [
        "IID" if row.method == "IID" else f"alpha={row.alpha:g}"
        for row in treatments.itertuples()
    ]
    figure, axes = project.plt.subplots(2, 3, figsize=(15, 8))
    for axis, (metric, label) in zip(axes.flat, metric_labels.items()):
        means = treatments[f"{metric}_mean"].to_numpy(dtype=float)
        deviations = treatments[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        axis.bar(
            np.arange(len(labels)),
            means,
            yerr=deviations,
            capsize=3,
            color="#1976a5",
        )
        axis.set_title(label)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].set_visible(False)
    figure.suptitle("FedAvg performance: IID versus Dirichlet Non-IID")
    figure.tight_layout()
    figure.savefig(output_dir / "iid_vs_noniid_performance.png", dpi=160)
    project.plt.close(figure)


def _save_distribution_plot(client_metrics: pd.DataFrame, alphas, output_dir):
    smallest_alpha = min(alphas)
    selected = client_metrics[
        (client_metrics["method"] == "IID")
        | (
            (client_metrics["method"] == "Non-IID")
            & np.isclose(client_metrics["alpha"], smallest_alpha)
        )
    ]
    means = selected.groupby(
        ["method", "client_id"], as_index=False
    )[["class_0_count", "class_1_count"]].mean()
    figure, axes = project.plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, method, title in (
        (axes[0], "IID", "IID"),
        (axes[1], "Non-IID", f"Non-IID (alpha={smallest_alpha:g})"),
    ):
        data = means[means["method"] == method].sort_values("client_id")
        clients = data["client_id"].to_numpy()
        class_0 = data["class_0_count"].to_numpy()
        class_1 = data["class_1_count"].to_numpy()
        axis.bar(clients, class_0, label="No cardiovascular disease", color="#6baed6")
        axis.bar(
            clients,
            class_1,
            bottom=class_0,
            label="Cardiovascular disease",
            color="#f28e2b",
        )
        axis.set_title(title)
        axis.set_xlabel("Simulated hospital client")
        axis.set_xticks(clients)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean training records across seeds")
    axes[1].legend(loc="upper right")
    figure.suptitle("Training-label distribution by simulated client")
    figure.tight_layout()
    figure.savefig(output_dir / "client_class_distribution.png", dpi=160)
    project.plt.close(figure)


def _save_round_plot(round_metrics: pd.DataFrame, metric: str, output_dir):
    figure, axis = project.plt.subplots(figsize=(9, 5))
    grouped = round_metrics.groupby(
        ["method", "alpha", "round"], dropna=False
    )[metric].agg(["mean", "std"]).reset_index()
    treatments = grouped[["method", "alpha"]].drop_duplicates()
    for treatment in treatments.itertuples(index=False):
        condition = grouped["method"] == treatment.method
        if pd.isna(treatment.alpha):
            condition &= grouped["alpha"].isna()
            label = "IID"
        else:
            condition &= np.isclose(grouped["alpha"], treatment.alpha)
            label = f"Non-IID alpha={treatment.alpha:g}"
        data = grouped[condition].sort_values("round")
        axis.errorbar(
            data["round"],
            data["mean"],
            yerr=data["std"].fillna(0),
            marker="o",
            capsize=2,
            label=label,
        )
    axis.set(
        xlabel="Communication round",
        ylabel=metric.replace("_", " ").title(),
        ylim=(0, 1.05),
        title=f"{metric.replace('_', ' ').title()} across FedAvg rounds",
    )
    axis.set_xticks(sorted(round_metrics["round"].unique()))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    filename = "accuracy_across_rounds.png" if metric == "accuracy" else "f1_across_rounds.png"
    figure.savefig(output_dir / filename, dpi=160)
    project.plt.close(figure)


def _save_alpha_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    non_iid = summary[summary["method"] == "Non-IID"].sort_values("alpha")
    figure, axes = project.plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, title in (
        (axes[0], "accuracy", "Accuracy by Dirichlet alpha"),
        (axes[1], "f1_score", "F1 score by Dirichlet alpha"),
    ):
        axis.errorbar(
            non_iid["alpha"],
            non_iid[f"{metric}_mean"],
            yerr=non_iid[f"{metric}_std"].fillna(0),
            marker="o",
            capsize=4,
            color="#1976a5",
        )
        axis.set_xscale("log")
        axis.set_xticks(non_iid["alpha"])
        axis.set_xticklabels([f"{alpha:g}" for alpha in non_iid["alpha"]])
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("Dirichlet alpha (log scale)")
        axis.set_ylabel(title.split(" by ")[0])
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle("FedAvg performance variation with client heterogeneity")
    figure.tight_layout()
    figure.savefig(output_dir / "performance_by_alpha.png", dpi=160)
    project.plt.close(figure)


def _summary_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    summary = trials.groupby(
        ["method", "alpha"],
        dropna=False,
        sort=False,
    )[list(CLIENT_METRICS)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def run_experiment(
    archive_path: Path,
    results_dir: Path,
    alphas=DEFAULT_ALPHAS,
    seeds=DEFAULT_SEEDS,
    num_clients=project.NUM_CLIENTS,
    num_rounds=project.NUM_ROUNDS,
) -> Path:
    alphas = tuple(float(alpha) for alpha in alphas)
    seeds = tuple(int(seed) for seed in seeds)
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Provide one or more positive, finite Dirichlet alpha values.")
    if len(seeds) < 2:
        raise ValueError("At least two random seeds are required for repeated trials.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique.")
    if num_clients < 2 or num_rounds < 1:
        raise ValueError("Use at least two clients and one communication round.")

    prepared = train_cardio.prepare_cardio_data(archive_path)
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    test_row_indices = np.asarray(prepared["test_row_indices"], dtype=np.int64)
    test_set_hash = hashlib.sha256(
        np.sort(test_row_indices).tobytes()
    ).hexdigest()

    output_dir = create_output_directory(results_dir)
    trial_rows = []
    round_rows = []
    client_rows = []
    audit_rows = []
    conditions = [("IID", None), *(("Non-IID", alpha) for alpha in alphas)]

    for seed in seeds:
        for method, alpha in conditions:
            if method == "IID":
                partitions = project.partition_indices_iid(
                    y_train,
                    num_clients=num_clients,
                    seed=seed,
                )
            else:
                partitions = project.partition_indices_dirichlet(
                    y_train,
                    num_clients=num_clients,
                    alpha=alpha,
                    seed=seed,
                )
            partition_audit = _verify_partition(partitions, len(y_train))
            audit_rows.append({
                "method": method,
                "alpha": alpha,
                "seed": seed,
                **partition_audit,
                "test_set_sha256": test_set_hash,
            })
            clients = _client_data(X_train, y_train, partitions)
            client_rows.extend(
                _client_trial_rows(
                    clients,
                    X_test,
                    y_test,
                    method,
                    alpha,
                    seed,
                )
            )

            class_fraction_std = float(np.std([
                np.mean(client_labels)
                for _, client_labels in clients.values()
            ]))
            weights, bias, history = project.train_federated_model(
                clients,
                X_test,
                y_test,
                num_rounds=num_rounds,
                dropout_round=None,
                dropout_client_id=None,
                verbose=False,
                seed=seed,
            )
            for round_result in history:
                round_rows.append({
                    "method": method,
                    "alpha": alpha,
                    "seed": seed,
                    **round_result,
                })
            metrics = evaluate_with_auc(X_test, y_test, weights, bias)
            trial_rows.append({
                "method": method,
                "alpha": alpha,
                "seed": seed,
                "test_rows": len(y_test),
                "test_set_sha256": test_set_hash,
                "client_class_fraction_std": class_fraction_std,
                **metrics,
            })
            label = "IID" if method == "IID" else f"Non-IID alpha={alpha:g}"
            print(
                f"seed={seed} {label}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_score']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    trial_frame = pd.DataFrame(trial_rows)
    round_frame = pd.DataFrame(round_rows)
    client_frame = pd.DataFrame(client_rows)
    audit_frame = pd.DataFrame(audit_rows)
    summary_frame = _summary_metrics(trial_frame)
    for frame, filename in (
        (trial_frame, "metrics_by_seed.csv"),
        (summary_frame, "summary_metrics.csv"),
        (round_frame, "round_metrics.csv"),
        (client_frame, "client_metrics_and_distribution.csv"),
        (audit_frame, "partition_audit.csv"),
    ):
        frame.to_csv(output_dir / filename, index=False)

    _save_performance_plot(summary_frame, output_dir)
    _save_distribution_plot(client_frame, alphas, output_dir)
    _save_round_plot(round_frame, "accuracy", output_dir)
    _save_round_plot(round_frame, "f1_score", output_dir)
    _save_alpha_plot(summary_frame, output_dir)

    configuration = {
        "dataset": "Cardiovascular disease (cardio_train.csv)",
        "source_archive_name": archive_path.name,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_set_sha256": test_set_hash,
        "preprocessing": (
            "Fixed stratified split; imputation medians and StandardScaler "
            "fit only on training data."
        ),
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "seeds": list(seeds),
        "alphas": list(alphas),
        "iid_method": "Random shuffle and balanced array split.",
        "non_iid_method": (
            "Per-class Dirichlet client proportions with capacity-aware "
            "redistribution to balanced client sample counts."
        ),
        "model": "Existing custom logistic regression and FedAvg implementation.",
        "dropout": False,
        "differential_privacy": False,
        "meaning": (
            "Smaller alpha generally yields more heterogeneous client label "
            "proportions; larger alpha generally yields more similar proportions."
        ),
        "privacy_note": (
            "This experiment studies data heterogeneity and does not provide "
            "additional privacy protection."
        ),
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    if audit_frame["test_set_sha256"].nunique() != 1:
        raise RuntimeError("A trial did not use the same held-out test set.")

    print(f"Experiment results saved to: {output_dir}")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare IID and Dirichlet Non-IID cardio FedAvg."
    )
    parser.add_argument(
        "--dataset-zip",
        required=True,
        type=Path,
        help="Path to the ZIP archive containing cardio_train.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for a new timestamped experiment bundle.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="One or more positive Dirichlet alpha values.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="At least two unique seeds for repeated comparisons.",
    )
    parser.add_argument("--clients", type=int, default=project.NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=project.NUM_ROUNDS)
    arguments = parser.parse_args()
    run_experiment(
        arguments.dataset_zip,
        arguments.results_dir,
        alphas=arguments.alphas,
        seeds=arguments.seeds,
        num_clients=arguments.clients,
        num_rounds=arguments.rounds,
    )


if __name__ == "__main__":
    main()
