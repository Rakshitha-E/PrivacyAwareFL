"""Train the project models on the 70,000-row cardio_train.csv archive."""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import main as project


EXPECTED_COLUMNS = (
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
)
FEATURE_NAMES = (
    "age_years",
    "gender",
    "height_cm",
    "weight_kg",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "smoke",
    "alcohol",
    "active",
)


def load_cardio_archive(archive_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            entry for entry in archive.namelist()
            if Path(entry).name == "cardio_train.csv"
        ]
        if len(members) != 1:
            raise ValueError(
                "Expected exactly one cardio_train.csv file in the ZIP archive."
            )
        with archive.open(members[0]) as csv_file:
            frame = pd.read_csv(csv_file, sep=";")

    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected cardio_train.csv schema: "
            + ", ".join(str(column) for column in frame.columns)
        )
    if frame.empty or frame.isna().any().any():
        raise ValueError("The cardio dataset is empty or contains missing values.")
    if not frame["cardio"].isin((0, 1)).all():
        raise ValueError("The cardio target must contain only binary 0/1 labels.")

    plausible_measurements = (
        frame["height"].between(120, 220)
        & frame["weight"].between(30, 200)
        & frame["ap_hi"].between(70, 250)
        & frame["ap_lo"].between(40, 200)
        & (frame["ap_hi"] > frame["ap_lo"])
    )
    clean = frame.loc[plausible_measurements].copy()
    if clean.empty or clean["cardio"].nunique() != 2:
        raise ValueError("Measurement filtering left insufficient target classes.")
    clean["age_years"] = clean["age"] / 365.25
    clean = clean.rename(columns={
        "height": "height_cm",
        "weight": "weight_kg",
        "ap_hi": "systolic_bp",
        "ap_lo": "diastolic_bp",
        "gluc": "glucose",
        "alco": "alcohol",
    })
    return clean, plausible_measurements


def prepare_cardio_data(archive_path: Path) -> dict[str, object]:
    """Create the shared, leakage-safe cardio train/test split and features."""

    frame, plausible_measurements = load_cardio_archive(archive_path)
    features = frame.loc[:, list(FEATURE_NAMES)].astype(float)
    labels = frame["cardio"].to_numpy(dtype=int)
    row_indices = np.arange(len(frame))
    (
        X_train_df,
        X_test_df,
        y_train,
        y_test,
        train_row_indices,
        test_row_indices,
    ) = train_test_split(
        features,
        labels,
        row_indices,
        test_size=0.25,
        random_state=project.RANDOM_SEED,
        stratify=labels,
    )

    imputation_medians = X_train_df.median().to_numpy(dtype=float)
    X_train_df = X_train_df.fillna(
        pd.Series(imputation_medians, index=FEATURE_NAMES)
    )
    X_test_df = X_test_df.fillna(
        pd.Series(imputation_medians, index=FEATURE_NAMES)
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    return {
        "frame": frame,
        "plausible_measurements": plausible_measurements,
        "X_train": X_train,
        "X_test": X_test,
        "X_test_raw": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "imputation_medians": imputation_medians,
        "train_row_indices": train_row_indices,
        "test_row_indices": test_row_indices,
    }


def create_output_directory(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cardio_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = results_dir / run_name
    suffix = 1
    while output_dir.exists():
        output_dir = results_dir / f"{run_name}_{suffix}"
        suffix += 1
    output_dir.mkdir()
    return output_dir


def train(archive_path: Path, results_dir: Path) -> Path:
    prepared = prepare_cardio_data(archive_path)
    frame = prepared["frame"]
    plausible_measurements = prepared["plausible_measurements"]
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    X_test_df = prepared["X_test_raw"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    scaler = prepared["scaler"]
    imputation_medians = prepared["imputation_medians"]

    clients = project.split_clients_non_iid(
        X_train,
        y_train,
        num_clients=project.NUM_CLIENTS,
        seed=project.RANDOM_SEED,
    )
    weights, bias, history = project.train_federated_model(
        clients,
        X_test,
        y_test,
        num_rounds=project.NUM_ROUNDS,
        dropout_round=project.DROPOUT_ROUND,
        dropout_client_id=project.DROPOUT_CLIENT_ID,
        verbose=False,
    )
    dp_weights, dp_bias, dp_history, privacy_summary = (
        project.train_dp_federated_model(
            clients,
            X_test,
            y_test,
            num_rounds=project.NUM_ROUNDS,
            clipping_norm=project.DP_CLIPPING_NORM,
            noise_multiplier=project.DP_NOISE_MULTIPLIER,
            delta=project.DP_DELTA,
            dropout_round=project.DROPOUT_ROUND,
            dropout_client_id=project.DROPOUT_CLIENT_ID,
            verbose=False,
        )
    )
    local_metrics = project.evaluate_local_only(clients, X_test, y_test)
    centralized_metrics, centralized_weights, centralized_bias = (
        project.evaluate_centralized(X_train, y_train, X_test, y_test)
    )
    federated_metrics = project.calculate_metrics(X_test, y_test, weights, bias)
    dp_metrics = project.calculate_metrics(X_test, y_test, dp_weights, dp_bias)

    output_dir = create_output_directory(results_dir)
    project.run_additional_experiments(
        clients,
        X_train,
        y_train,
        X_test,
        y_test,
        federated_metrics,
        dp_weights,
        dp_bias,
        privacy_summary,
        scaler.mean_,
        scaler.scale_,
        imputation_medians,
        output_dir,
    )
    model_metadata = {
        "feature_names": np.array(FEATURE_NAMES),
        "target_labels": np.array(
            ["no cardiovascular disease", "cardiovascular disease"]
        ),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "imputation_medians": imputation_medians,
        "training_rows": np.array([len(X_train)]),
        "test_rows": np.array([len(X_test)]),
        "random_seed": np.array([project.RANDOM_SEED]),
    }
    np.savez(
        output_dir / "global_model.npz",
        weights=weights,
        bias=np.array([bias]),
        **model_metadata,
    )
    np.savez(
        output_dir / "centralized_model.npz",
        weights=centralized_weights,
        bias=np.array([centralized_bias]),
        **model_metadata,
    )
    np.savez(
        output_dir / "dp_global_model.npz",
        weights=dp_weights,
        bias=np.array([dp_bias]),
        **model_metadata,
        clipping_norm=np.array([project.DP_CLIPPING_NORM]),
        noise_multiplier=np.array([project.DP_NOISE_MULTIPLIER]),
        privacy_epsilon=np.array([privacy_summary["epsilon"]]),
        privacy_delta=np.array([privacy_summary["delta"]]),
    )
    pd.DataFrame(history).to_csv(
        output_dir / "federated_metrics.csv",
        index=False,
    )
    pd.DataFrame(dp_history).to_csv(
        output_dir / "dp_federated_metrics.csv",
        index=False,
    )
    comparison = pd.DataFrame([
        {"approach": "Local-only", **local_metrics},
        {"approach": "Centralized", **centralized_metrics},
        {"approach": "FedAvg", **federated_metrics},
        {"approach": "Differentially Private FedAvg", **dp_metrics},
    ])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    project.plot_metric_grid(
        comparison,
        "approach",
        "Model Comparison on the Held-Out Cardio Test Set",
        output_dir / "model_comparison.png",
    )
    figure = project.plt.figure(figsize=(9, 5))
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        project.plt.plot(
            pd.DataFrame(history)["round"],
            pd.DataFrame(history)[metric],
            marker="o",
            label=metric.replace("_", " ").title(),
        )
    project.plt.xlabel("Federated Round")
    project.plt.ylabel("Score")
    project.plt.title("Cardio FedAvg Performance Across Rounds")
    project.plt.ylim(0, 1.05)
    project.plt.xticks(range(1, project.NUM_ROUNDS + 1))
    project.plt.grid(True, alpha=0.3)
    project.plt.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "performance.png", dpi=150)
    project.plt.close(figure)
    summary = {
        "dataset": "Cardiovascular disease (cardio_train.csv)",
        "source_archive": archive_path.name,
        "original_rows": int(len(plausible_measurements)),
        "excluded_implausible_rows": int((~plausible_measurements).sum()),
        "rows_used": int(len(frame)),
        "features": list(FEATURE_NAMES),
        "target": "cardio (0 = no cardiovascular disease, 1 = disease)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "client_count": len(clients),
        "federated_rounds": project.NUM_ROUNDS,
        "dropout_round": project.DROPOUT_ROUND,
        "dropout_client_id": project.DROPOUT_CLIENT_ID,
        "differential_privacy": {
            "clipping_norm": project.DP_CLIPPING_NORM,
            "noise_multiplier": project.DP_NOISE_MULTIPLIER,
            "delta": privacy_summary["delta"],
            "epsilon": privacy_summary["epsilon"],
        },
        "metrics": comparison.to_dict(orient="records"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    probe_probability = project.predict_proba_from_raw(
        X_test_df.iloc[0].to_numpy(dtype=float),
        model_path=output_dir / "global_model.npz",
    )
    if not np.isfinite(probe_probability) or not 0 <= probe_probability <= 1:
        raise RuntimeError("Saved cardio model failed its prediction check.")

    print(f"Results saved to: {output_dir}")
    print(f"Training/test rows: {len(X_train)}/{len(X_test)}")
    print(f"Excluded implausible rows: {int((~plausible_measurements).sum())}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"DP epsilon: {privacy_summary['epsilon']:.4f} at delta={privacy_summary['delta']:g}")
    print(f"Saved-model prediction check: {probe_probability:.4f}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train centralized and federated models from cardio_train.csv."
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
        help="Directory for a new timestamped result bundle.",
    )
    arguments = parser.parse_args()
    train(arguments.dataset_zip, arguments.results_dir)


if __name__ == "__main__":
    main()
