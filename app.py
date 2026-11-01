"""Interactive Streamlit dashboard for the existing FL heart-disease project."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import main as project
import train_cardio


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_TYPES = (
    "UCI Cleveland Heart Disease",
    "Cardiovascular Disease (cardio_train.csv)",
)
DATASET_FEATURES = {
    DATASET_TYPES[0]: tuple(project.HEART_FEATURE_NAMES),
    DATASET_TYPES[1]: tuple(train_cardio.FEATURE_NAMES),
}
RUN_DIR = RESULTS_DIR
MODEL_PATH = RUN_DIR / "global_model.npz"
SELECTED_DATASET: str | None = None
METRICS = ("accuracy", "precision", "recall", "f1_score")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
}
PAGES = (
    "Overview",
    "Federated learning",
    "Model comparison",
    "Differential privacy",
    "Additional experiments",
    "IID vs Non-IID Analysis",
    "Heart disease prediction",
    "Project information",
)


class DashboardDataError(Exception):
    """Raised when a required saved artifact is missing or malformed."""


st.set_page_config(
    page_title="Privacy-Aware Federated Learning",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy: #102a43; --blue: #1976a5; --pale: #eaf5fb; }
        .stApp { background: #f6f9fc; color: #172b4d; }
        [data-testid="stSidebar"] { background: #edf5fb; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #102a43; }
        h1, h2, h3 { color: #102a43; letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #dce8f1; padding: 16px;
            border-radius: 12px; box-shadow: 0 2px 10px rgba(16, 42, 67, .04);
        }
        .hero {
            background: linear-gradient(120deg, #102a43, #176b91);
            color: white; border-radius: 18px; padding: 28px 32px;
            margin: 4px 0 22px 0;
        }
        .hero h1 { color: white; margin: 0 0 8px 0; }
        .hero p { color: #e8f4fb; font-size: 1.05rem; margin: 0; }
        .info-card {
            background: white; border: 1px solid #dce8f1; border-radius: 14px;
            padding: 18px; height: 100%;
        }
        .info-card h3 { margin-top: 0; }
        .client-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .client-card, .server-card {
            background: white; color: #102a43; border: 1px solid #cfe0ed;
            border-radius: 12px; padding: 16px; text-align: center;
        }
        .client-card strong, .server-card strong { display: block; font-size: 1rem; }
        .client-card span, .server-card span { color: #52677d; font-size: .88rem; }
        .server-card { background: #e8f4fb; border-color: #9dc9df; }
        .flow-arrow { text-align: center; color: #1976a5; font-weight: 700; padding: 10px; }
        .subtle { color: #52677d; }
        .disclaimer {
            background: #fff8e6; border-left: 4px solid #e6a700;
            border-radius: 8px; padding: 12px 16px; color: #51421b;
        }
        @media (max-width: 700px) {
            .client-grid { grid-template-columns: repeat(2, 1fr); }
            .hero { padding: 22px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(filename: str, run_dir: str) -> pd.DataFrame:
    path = Path(run_dir) / filename
    if not path.is_file():
        raise DashboardDataError(f"Required results file was not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DashboardDataError(f"Could not read {path.name}: {error}") from error
    if frame.empty:
        raise DashboardDataError(f"Results file {path.name} contains no rows.")
    return frame


@st.cache_data(show_spinner=False)
def load_model_metadata(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise DashboardDataError(f"Saved prediction model was not found: {path}")
    try:
        with np.load(path) as model:
            required = {
                "weights",
                "bias",
                "feature_names",
                "scaler_mean",
                "scaler_scale",
                "imputation_medians",
            }
            missing = required.difference(model.files)
            if missing:
                raise DashboardDataError(
                    f"Model artifact is missing fields: {', '.join(sorted(missing))}"
                )
            feature_names = tuple(model["feature_names"].astype(str))
            if not feature_names or len(set(feature_names)) != len(feature_names):
                raise DashboardDataError(
                    "Saved model has an empty or duplicate feature schema."
                )
            feature_count = len(feature_names)
            for name in (
                "weights",
                "scaler_mean",
                "scaler_scale",
                "imputation_medians",
            ):
                if model[name].shape != (feature_count,):
                    raise DashboardDataError(
                        f"Model field {name!r} has an unexpected shape."
                    )
                if not np.isfinite(model[name]).all():
                    raise DashboardDataError(
                        f"Model field {name!r} contains non-finite values."
                    )
            if np.any(model["scaler_scale"] <= 0):
                raise DashboardDataError("Model scaler contains invalid scales.")
            if model["bias"].size != 1 or not np.isfinite(model["bias"]).all():
                raise DashboardDataError("Model bias is invalid.")
            if (
                "target_labels" in model.files
                and model["target_labels"].size != 2
            ):
                raise DashboardDataError("Saved model target labels are invalid.")
            target_labels = (
                tuple(model["target_labels"].astype(str))
                if "target_labels" in model.files
                else ("Class 0", "Class 1")
            )
    except (OSError, ValueError) as error:
        if isinstance(error, DashboardDataError):
            raise
        raise DashboardDataError(f"Could not load saved model: {error}") from error
    return {
        "feature_names": feature_names,
        "feature_count": feature_count,
        "target_labels": target_labels,
    }


@st.cache_data(show_spinner=False)
def load_run_summary(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir) / "training_summary.json"
    if path.is_file():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DashboardDataError(
                f"Could not read training summary {path.name}: {error}"
            ) from error
        if not isinstance(summary, dict):
            raise DashboardDataError("Training summary must be a JSON object.")
        return summary
    return {
        "dataset": "UCI Cleveland Heart Disease",
        "original_rows": 303,
        "rows_used": 303,
        "train_rows": 227,
        "test_rows": 76,
        "client_count": 4,
        "target": "heart disease",
    }


@st.cache_data(show_spinner=False)
def load_iid_configuration(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir) / "configuration.json"
    if not path.is_file():
        raise DashboardDataError(f"Experiment configuration was not found: {path}")
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(
            f"Could not read experiment configuration: {error}"
        ) from error
    if not isinstance(configuration, dict):
        raise DashboardDataError("Experiment configuration must be a JSON object.")
    return configuration


def available_iid_experiment_dirs() -> list[Path]:
    return sorted(
        (
            path
            for path in RESULTS_DIR.glob("iid_noniid_run_*")
            if path.is_dir()
            and (path / "summary_metrics.csv").is_file()
            and (path / "client_metrics_and_distribution.csv").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def available_run_dirs() -> list[Path]:
    return sorted(
        (
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir()
            and (path / "global_model.npz").is_file()
            and (path / "model_comparison.csv").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def dataset_type_for_features(feature_names: tuple[str, ...]) -> str | None:
    normalized_names = tuple(str(name) for name in feature_names)
    return next(
        (
            dataset_name
            for dataset_name, expected_names in DATASET_FEATURES.items()
            if normalized_names == expected_names
        ),
        None,
    )


def compatible_run_dirs(dataset_name: str) -> list[Path]:
    if dataset_name not in DATASET_FEATURES:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

    compatible = []
    expected_features = DATASET_FEATURES[dataset_name]
    for path in available_run_dirs():
        try:
            metadata = load_model_metadata(str(path / "global_model.npz"))
            summary = load_run_summary(str(path))
        except DashboardDataError:
            continue
        summary_features = tuple(summary.get("features", ()))
        if (
            metadata["feature_names"] == expected_features
            and (not summary_features or summary_features == expected_features)
        ):
            compatible.append(path)
    return compatible


def run_label(path: Path) -> str:
    return path.name


def load_active_run_context() -> dict[str, Any]:
    if SELECTED_DATASET is None:
        raise DashboardDataError("Select a dataset before loading its training run.")
    metadata = load_model_metadata(str(MODEL_PATH))
    model_dataset = dataset_type_for_features(metadata["feature_names"])
    if model_dataset != SELECTED_DATASET:
        raise DashboardDataError(
            f"The selected run's model is not compatible with {SELECTED_DATASET}."
        )
    summary = load_run_summary(str(RUN_DIR))
    summary_features = tuple(summary.get("features", ()))
    if summary_features and summary_features != metadata["feature_names"]:
        raise DashboardDataError(
            "The selected run summary and saved model use different feature schemas."
        )
    return {
        **summary,
        **metadata,
        "dataset": SELECTED_DATASET,
        "selected_run": RUN_DIR.name,
    }


def show_data_error(error: Exception) -> None:
    st.error(f"Unable to display this saved experiment: {error}")


def section_heading(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def show_graph(filename: str, caption: str) -> None:
    path = RUN_DIR / filename
    if not path.is_file():
        st.error(f"Saved graph not found: {path}")
        return
    st.image(str(path), caption=caption, width="stretch")


def format_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    return result.rename(columns=METRIC_LABELS)


def metric_comparison_chart(frame: pd.DataFrame, category: str, title: str):
    long_frame = frame.melt(
        id_vars=[category],
        value_vars=[metric for metric in METRICS if metric in frame.columns],
        var_name="Metric",
        value_name="Score",
    )
    long_frame["Metric"] = long_frame["Metric"].map(METRIC_LABELS)
    chart = px.bar(
        long_frame,
        x=category,
        y="Score",
        color="Metric",
        barmode="group",
        title=title,
        color_discrete_sequence=["#1976a5", "#45a4c8", "#65b891", "#e5a84b"],
        hover_data={"Score": ":.3f"},
    )
    chart.update_layout(
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        xaxis_title=None,
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"t": 55, "b": 30},
    )
    return chart


def render_architecture(client_count: int) -> None:
    clients = "".join(
        f'<div class="client-card"><strong>Client {client_id}</strong>'
        "<span>Local data · local training</span></div>"
        for client_id in range(1, client_count + 1)
    )
    st.markdown(
        f"""
        <div class="client-grid">{clients}</div>
        <div class="flow-arrow">Local model updates → central server · sample-weighted FedAvg</div>
        <div class="server-card"><strong>Federated coordination server</strong>
          <span>Aggregates updates and broadcasts the global model</span></div>
        <div class="flow-arrow">← Updated global model returned to the simulated clients</div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    context = load_active_run_context()
    dataset_name = str(context.get("dataset", "Selected dataset"))
    row_count = int(context.get("rows_used", context.get("original_rows", 0)))
    client_count = int(context.get("client_count", 4))
    train_count = int(context.get("train_rows", 0))
    test_count = int(context.get("test_rows", 0))
    feature_count = int(context["feature_count"])
    dp_available = (RUN_DIR / "privacy_utility_metrics.csv").is_file()
    dp_text = (
        "A differentially private variant is available for this run."
        if dp_available
        else "Differential privacy results are not available for this run."
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>Privacy-Preserving Federated Learning</h1>
          <p>Heart disease prediction with simulated clients and FedAvg. {dp_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "### A collaborative learning demonstration"
        "\nHospitals can train a shared model by exchanging model updates instead of pooling "
        f"their records in one central dataset. The selected run uses **{dataset_name}**; "
        "it is an educational simulation, not a clinical deployment."
    )
    m1, m2, m3, m4 = st.columns(4)
    for column, value, label in (
        (m1, f"{row_count:,}", "Records used"),
        (m2, str(feature_count), "Input features"),
        (m3, str(client_count), "Simulated clients"),
        (m4, f"{test_count:,}", "Held-out test records"),
    ):
        column.metric(label, value)
    if train_count and test_count:
        st.caption(
            f"Training split: {train_count:,} records · test split: {test_count:,} records."
        )
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="info-card">
              <h3>Problem & motivation</h3>
              <p>Clinical data is sensitive and can be siloed across institutions. Central pooling
              can be operationally and privacy-wise challenging, while isolated models may not
              benefit from broader collaboration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
              <h3>Proposed solution</h3>
              <p>Each simulated hospital trains locally. A coordinating server combines updates
              with sample-size-weighted FedAvg. A separate differentially private run is shown
              only when privacy artifacts exist for the selected experiment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.subheader("System architecture")
    render_architecture(client_count)
    st.caption(
        "These clients are simulated partitions running in one Python process—not real hospitals "
        "or a deployed network. Raw records are not sent between simulated clients."
    )
    try:
        comparison = load_csv("model_comparison.csv", str(RUN_DIR))
        st.subheader("At a glance: held-out accuracy")
        st.bar_chart(
            comparison.set_index("approach")[["accuracy"]],
            color="#1976a5",
        )
        if (RUN_DIR / "model_comparison.png").is_file():
            show_graph("model_comparison.png", "Saved model comparison from the selected run.")
    except DashboardDataError as error:
        show_data_error(error)


def render_federated_learning() -> None:
    context = load_active_run_context()
    client_count = int(context.get("client_count", 4))
    dropout_round = context.get("dropout_round")
    dropout_client_id = context.get("dropout_client_id")
    section_heading(
        "Federated learning demonstration",
        f"Recorded results for {context.get('dataset', 'the selected dataset')}; this page does not retrain models.",
    )
    render_architecture(client_count)
    st.markdown(
        f"""
        **Round workflow:** each participating client trains a local Logistic Regression model;
        the server aggregates the client models using their training sample counts; the updated
        global model is then available for the next round.
        """
    )
    if dropout_round is not None and dropout_client_id is not None:
        st.info(
            f"Client {dropout_client_id} was scheduled to drop out in round "
            f"{dropout_round}."
        )
    st.markdown(
        """
        - **Local-only:** a model is trained using one client's shard, without federation.
        - **Centralized:** training uses the combined training split at one location.
        - **Federated:** clients retain their data locally and share model updates for aggregation.
        """
    )
    try:
        history = load_csv("federated_metrics.csv", str(RUN_DIR))
        required = {"round", "active_clients", *METRICS}
        if not required.issubset(history.columns):
            raise DashboardDataError("federated_metrics.csv is missing expected round metrics.")
        st.subheader("Recorded FedAvg rounds")
        st.plotly_chart(
            metric_comparison_chart(history, "round", "FedAvg test metrics by round"),
            width="stretch",
        )
        st.dataframe(format_metrics(history), width="stretch", hide_index=True)
        st.caption("Round metrics use the saved held-out test set; active client counts are recorded per round.")
        if (RUN_DIR / "performance.png").is_file():
            show_graph("performance.png", "Saved performance overview.")
    except DashboardDataError as error:
        show_data_error(error)


def comparison_with_privacy() -> pd.DataFrame:
    comparison = load_csv("model_comparison.csv", str(RUN_DIR))
    required = {"approach", *METRICS}
    if not required.issubset(comparison.columns):
        raise DashboardDataError("model_comparison.csv is missing expected metrics.")
    comparison["epsilon"] = np.nan
    comparison["delta"] = np.nan
    privacy_path = RUN_DIR / "privacy_utility_metrics.csv"
    if not privacy_path.is_file():
        return comparison
    privacy = load_csv("privacy_utility_metrics.csv", str(RUN_DIR))
    dp_rows = privacy.loc[
        privacy["approach"].astype(str).str.startswith("DP-FedAvg")
        & np.isclose(pd.to_numeric(privacy["noise_multiplier"], errors="coerce"), 2.0)
    ]
    dp_rows = dp_rows.head(1)
    if not dp_rows.empty:
        mask = comparison["approach"].astype(str).str.contains(
            "Differentially Private", case=False, regex=False
        )
        comparison.loc[mask, "epsilon"] = float(dp_rows.iloc[0]["epsilon"])
        comparison.loc[mask, "delta"] = float(dp_rows.iloc[0]["delta"])
    return comparison


def render_model_comparison() -> None:
    context = load_active_run_context()
    section_heading(
        "Model comparison",
        f"Metrics on the {int(context.get('test_rows', 0)):,}-record held-out "
        f"test set for {context.get('dataset', 'the selected dataset')}.",
    )
    try:
        comparison = comparison_with_privacy()
        st.plotly_chart(
            metric_comparison_chart(comparison, "approach", "Held-out model comparison"),
            width="stretch",
        )
        st.dataframe(
            format_metrics(comparison),
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
                "F1-score": st.column_config.NumberColumn(format="%.4f"),
                "epsilon": st.column_config.NumberColumn("Epsilon", format="%.4f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.1e"),
            },
        )
        for filename, caption in (
            ("model_comparison.png", "Saved comparison graph from this run."),
            ("performance.png", "Saved metric overview from this run."),
        ):
            if (RUN_DIR / filename).is_file():
                show_graph(filename, caption)
        if "epsilon" in comparison.columns and comparison["epsilon"].notna().any():
            st.caption("Epsilon and delta are populated only for the saved DP model.")
    except DashboardDataError as error:
        show_data_error(error)


def render_differential_privacy() -> None:
    section_heading(
        "Differential privacy",
        "Privacy experiments are shown only when they were produced for the selected run.",
    )
    if not (RUN_DIR / "privacy_utility_metrics.csv").is_file():
        st.info(
            "This run contains no differentially private model or privacy–utility "
            "sweep. The displayed accuracy results are not differentially private."
        )
        return
    st.markdown(
        """
        **Why add differential privacy?** Updates can carry information about a client's data.
        The experiment limits the influence of a client update before aggregation and adds
        calibrated Gaussian noise. This reduces update-level information under the accountant's
        assumptions, at a potential cost to model utility.

        **Mechanism in this project:** each complete client update (weights and bias) is L2-clipped.
        Sample-weighted clipped updates are aggregated, then Gaussian noise is added. The RDP
        accountant composes the per-round Gaussian bounds and reports epsilon for the selected delta.
        """
    )
    st.markdown(
        '<div class="disclaimer"><strong>Privacy caveat.</strong> Reported ε/δ is a conditional '
        'theoretical bound for the documented fixed, preprocessed client-shard assumptions and an '
        'ideal Gaussian mechanism. It is not a certified end-to-end privacy guarantee. Preprocessing '
        'statistics are centralized and not privatized; NumPy pseudorandom noise is not a production '
        'cryptographic source. Secure aggregation is not implemented.</div>',
        unsafe_allow_html=True,
    )
    try:
        results = load_csv("privacy_utility_metrics.csv", str(RUN_DIR))
        required = {
            "approach", "noise_multiplier", "epsilon", "delta",
            "repetitions", *METRICS,
        }
        if not required.issubset(results.columns):
            raise DashboardDataError("privacy_utility_metrics.csv is missing expected fields.")
        dp_results = results.loc[
            results["approach"].astype(str).str.startswith("DP-FedAvg")
        ].copy()
        dp_results = dp_results.sort_values("noise_multiplier")
        st.subheader("Actual privacy–utility experiment")
        display = dp_results[
            ["noise_multiplier", "epsilon", "delta", "repetitions", *METRICS]
        ].rename(
            columns={
                "noise_multiplier": "Noise multiplier",
                "epsilon": "Epsilon",
                "delta": "Delta",
                "repetitions": "Repeats",
                **METRIC_LABELS,
            }
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Epsilon": st.column_config.NumberColumn(format="%.4f"),
                "Delta": st.column_config.NumberColumn(format="%.1e"),
                **{
                    label: st.column_config.NumberColumn(format="%.4f")
                    for label in METRIC_LABELS.values()
                },
            },
        )
        st.plotly_chart(
            px.line(
                dp_results,
                x="noise_multiplier",
                y="epsilon",
                markers=True,
                title="Per-model epsilon by noise multiplier",
                labels={"noise_multiplier": "Noise multiplier", "epsilon": "Epsilon"},
                color_discrete_sequence=["#1976a5"],
            ).update_layout(plot_bgcolor="white", paper_bgcolor="white"),
            width="stretch",
        )
        st.plotly_chart(
            px.line(
                dp_results,
                x="noise_multiplier",
                y="accuracy",
                markers=True,
                title="Mean test accuracy by noise multiplier",
                labels={"noise_multiplier": "Noise multiplier", "accuracy": "Mean accuracy"},
                color_discrete_sequence=["#45a4c8"],
            ).update_layout(yaxis={"range": [0, 1]}, plot_bgcolor="white", paper_bgcolor="white"),
            width="stretch",
        )
        fedavg = results.loc[results["approach"].eq("FedAvg (no DP)")]
        if not fedavg.empty:
            st.metric("FedAvg accuracy (no DP)", f"{fedavg.iloc[0]['accuracy']:.3f}")
        joint = pd.to_numeric(dp_results.get("joint_sweep_epsilon"), errors="coerce").dropna()
        if not joint.empty:
            st.warning(
                f"The recorded joint composition for all 12 DP sweep releases is "
                f"ε = {joint.iloc[0]:.4f} at δ = {dp_results.iloc[0]['delta']:.1e}."
            )
        st.caption(
            "Higher noise reduced the reported epsilon in this run. Utility is based on three "
            "independent noise repetitions and is nonmonotonic; do not infer a stable trend from this small experiment."
        )
        for filename, caption in (
            ("privacy_epsilon_vs_noise.png", "Saved epsilon-versus-noise graph."),
            ("privacy_accuracy_vs_noise.png", "Saved accuracy-versus-noise graph."),
        ):
            if (RUN_DIR / filename).is_file():
                show_graph(filename, caption)
    except DashboardDataError as error:
        show_data_error(error)


def render_additional_experiments() -> None:
    section_heading(
        "Additional experiments",
        "Client-count and dropout analyses are displayed only when included in this run.",
    )
    scalability_tab, dropout_tab, evaluation_tab = st.tabs(
        ["Client scalability", "Client dropout", "Performance evaluation"]
    )
    with scalability_tab:
        try:
            data = load_csv("scalability_metrics.csv", str(RUN_DIR))
            if not {"num_clients", *METRICS}.issubset(data.columns):
                raise DashboardDataError("scalability_metrics.csv is missing expected metrics.")
            data = data.sort_values("num_clients")
            st.markdown(
                "The experiment compares 2, 4, 6, and 8 simulated clients while keeping the "
                "held-out test set fixed. The plot shows changes relative to the 4-client "
                "baseline in percentage points, so small differences are visible around zero; "
                "the table shows the absolute test scores."
            )
            if (RUN_DIR / "scalability_comparison.png").is_file():
                show_graph(
                    "scalability_comparison.png",
                    "Metric changes versus the 4-client baseline; labels are percentage points.",
                )
            else:
                st.plotly_chart(
                    metric_comparison_chart(
                        data,
                        "num_clients",
                        "Performance by client count (full 0–1 scale)",
                    ),
                    width="stretch",
                )
            display = format_metrics(data)
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    label: st.column_config.NumberColumn(format="%.6f")
                    for label in METRIC_LABELS.values()
                },
            )
        except DashboardDataError as error:
            st.info(f"Client-count experiment not available: {error}")
    with dropout_tab:
        try:
            data = load_csv("dropout_metrics.csv", str(RUN_DIR))
            required = {"scenario", *METRICS}
            if not required.issubset(data.columns):
                raise DashboardDataError("dropout_metrics.csv is missing expected metrics.")
            st.markdown(
                "When a client is unavailable, the server aggregates only the received client "
                "updates and renormalizes their sample-count weights."
            )
            st.plotly_chart(
                metric_comparison_chart(data, "scenario", "With and without client dropout"),
                width="stretch",
            )
            st.dataframe(format_metrics(data), width="stretch", hide_index=True)
            if (RUN_DIR / "dropout_comparison.png").is_file():
                show_graph("dropout_comparison.png", "Saved dropout comparison graph.")
        except DashboardDataError as error:
            st.info(f"Client-dropout experiment not available: {error}")
    with evaluation_tab:
        try:
            comparison = load_csv("model_comparison.csv", str(RUN_DIR))
            if not {"approach", *METRICS}.issubset(comparison.columns):
                raise DashboardDataError("model_comparison.csv is missing expected metrics.")
            st.plotly_chart(
                metric_comparison_chart(comparison, "approach", "Accuracy, precision, recall and F1"),
                width="stretch",
            )
            st.dataframe(format_metrics(comparison), width="stretch", hide_index=True)
            for filename, caption in (
                ("model_comparison.png", "Saved model comparison."),
                ("performance.png", "Saved performance overview."),
            ):
                if (RUN_DIR / filename).is_file():
                    show_graph(filename, caption)
        except DashboardDataError as error:
            show_data_error(error)


def render_iid_noniid_analysis() -> None:
    section_heading(
        "IID vs Non-IID Analysis",
        "FedAvg comparison using repeated, reproducible partitions of the cardio training data.",
    )
    st.markdown(
        """
        **IID** means each simulated hospital receives a random, balanced share of
        training records, so its disease mix is close to the overall mix.
        **Non-IID** means each hospital can have a different disease/no-disease
        mix. The Dirichlet alpha parameter controls the sampled heterogeneity:
        smaller values generally produce more uneven class proportions, while
        larger values generally make them more alike.

        These are simulated hospitals, not real institutions. The same fixed
        held-out test set and training-only preprocessing are used in each trial.
        This experiment studies data heterogeneity; it does **not** add privacy
        protection.
        """
    )
    experiment_dirs = available_iid_experiment_dirs()
    if not experiment_dirs:
        st.info(
            "No IID versus Non-IID results are saved yet. Run "
            "`python run_iid_noniid.py --dataset-zip \"C:\\path\\to\\archive.zip\"` "
            "from the project directory to create them."
        )
        return

    selected_dir = st.selectbox(
        "Saved IID/Non-IID experiment",
        experiment_dirs,
        format_func=lambda path: path.name,
        key="iid_noniid_experiment",
    )
    try:
        configuration = load_iid_configuration(str(selected_dir))
        summary = load_csv("summary_metrics.csv", str(selected_dir))
        client_data = load_csv(
            "client_metrics_and_distribution.csv",
            str(selected_dir),
        )
        required_summary = {
            "method",
            "alpha",
            "accuracy_mean",
            "accuracy_std",
            "f1_score_mean",
            "f1_score_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "roc_auc_mean",
            "roc_auc_std",
        }
        required_clients = {
            "method",
            "alpha",
            "client_id",
            "sample_count",
            "class_0_count",
            "class_1_count",
            "local_accuracy",
            "local_f1_score",
        }
        if not required_summary.issubset(summary.columns):
            raise DashboardDataError("summary_metrics.csv is missing expected metrics.")
        if not required_clients.issubset(client_data.columns):
            raise DashboardDataError(
                "Client distribution results are missing expected columns."
            )
    except DashboardDataError as error:
        show_data_error(error)
        return

    alpha_values = sorted(
        float(alpha)
        for alpha in summary.loc[
            summary["method"] == "Non-IID", "alpha"
        ].dropna().unique()
    )
    if not alpha_values:
        st.error("This experiment has no Non-IID alpha settings.")
        return
    details = st.columns(4)
    for column, value, label in (
        (details[0], configuration.get("num_clients", "—"), "Simulated clients"),
        (details[1], configuration.get("num_rounds", "—"), "FedAvg rounds"),
        (details[2], len(configuration.get("seeds", [])), "Random seeds"),
        (details[3], ", ".join(f"{alpha:g}" for alpha in alpha_values), "Dirichlet alpha"),
    ):
        column.metric(label, value)
    st.caption(
        f"Training/test rows: {configuration.get('train_rows', '—')}/"
        f"{configuration.get('test_rows', '—')} · "
        f"shared test-set SHA-256: {configuration.get('test_set_sha256', 'not recorded')}"
    )
    st.markdown(
        "**Partitioning:** "
        + str(configuration.get("iid_method", "IID details unavailable"))
        + " **Non-IID:** "
        + str(configuration.get("non_iid_method", "Dirichlet details unavailable"))
    )
    st.markdown(
        "**Alpha interpretation:** "
        + str(configuration.get("meaning", "Not recorded."))
    )

    st.subheader("Held-out FedAvg performance")
    metric_names = (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 score"),
        ("roc_auc", "ROC-AUC"),
    )
    comparison_rows = []
    for row in summary.to_dict(orient="records"):
        label = "IID" if row["method"] == "IID" else f"Non-IID (alpha={row['alpha']:g})"
        for metric, display_name in metric_names:
            comparison_rows.append({
                "Condition": label,
                "Metric": display_name,
                "Mean": row[f"{metric}_mean"],
                "Standard deviation": row[f"{metric}_std"],
            })
    comparison = pd.DataFrame(comparison_rows)
    chart = px.bar(
        comparison,
        x="Condition",
        y="Mean",
        color="Metric",
        barmode="group",
        error_y="Standard deviation",
        title="Mean held-out scores across independent seeds",
        color_discrete_sequence=[
            "#1976a5", "#45a4c8", "#65b891", "#e5a84b", "#8764a8"
        ],
    )
    chart.update_layout(
        yaxis={"range": [0, 1.05]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"t": 55, "b": 30},
    )
    st.plotly_chart(chart, width="stretch")
    display_summary = summary.copy()
    display_summary["condition"] = [
        "IID" if row.method == "IID" else f"Non-IID, alpha={row.alpha:g}"
        for row in summary.itertuples()
    ]
    display_columns = ["condition"]
    for metric, label in metric_names:
        display_summary[label] = display_summary.apply(
            lambda row: (
                f"{row[f'{metric}_mean']:.4f} ± "
                f"{row[f'{metric}_std']:.4f}"
            ),
            axis=1,
        )
        display_columns.append(label)
    st.dataframe(
        display_summary[display_columns],
        width="stretch",
        hide_index=True,
    )
    performance_plot = selected_dir / "iid_vs_noniid_performance.png"
    if performance_plot.is_file():
        st.image(
            str(performance_plot),
            caption="Saved mean performance with standard deviations.",
        )

    st.subheader("Client-wise training distribution and local performance")
    chosen_alpha = st.selectbox(
        "Non-IID distribution alpha",
        alpha_values,
        format_func=lambda alpha: f"{alpha:g}",
        key="iid_noniid_distribution_alpha",
    )
    chosen = client_data[
        (client_data["method"] == "IID")
        | (
            (client_data["method"] == "Non-IID")
            & np.isclose(client_data["alpha"], chosen_alpha)
        )
    ]
    client_means = chosen.groupby(
        ["method", "client_id"],
        as_index=False,
    )[[
        "sample_count",
        "class_0_count",
        "class_1_count",
        "local_accuracy",
        "local_f1_score",
    ]].mean()
    client_means["condition"] = np.where(
        client_means["method"] == "IID",
        "IID",
        f"Non-IID (alpha={chosen_alpha:g})",
    )
    distribution_long = client_means.melt(
        id_vars=["client_id", "condition"],
        value_vars=["class_0_count", "class_1_count"],
        var_name="Class",
        value_name="Mean training records",
    )
    distribution_long["Class"] = distribution_long["Class"].map({
        "class_0_count": "No cardiovascular disease",
        "class_1_count": "Cardiovascular disease",
    })
    distribution_chart = px.bar(
        distribution_long,
        x="client_id",
        y="Mean training records",
        color="Class",
        facet_col="condition",
        barmode="stack",
        title="Mean training-label distribution across seeds",
        color_discrete_map={
            "No cardiovascular disease": "#6baed6",
            "Cardiovascular disease": "#f28e2b",
        },
    )
    distribution_chart.update_layout(
        xaxis_title="Simulated hospital client",
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(distribution_chart, width="stretch")
    st.caption(
        "Client-wise local-model scores below are each client’s independently trained "
        "model evaluated on the same global held-out test set; they are not scores on "
        "the client’s training records."
    )
    display_clients = client_means.rename(columns={
        "client_id": "Client",
        "sample_count": "Mean training samples",
        "class_0_count": "Mean no-disease samples",
        "class_1_count": "Mean disease samples",
        "local_accuracy": "Local model accuracy",
        "local_f1_score": "Local model F1",
        "condition": "Condition",
    })
    st.dataframe(
        display_clients[[
            "Condition",
            "Client",
            "Mean training samples",
            "Mean no-disease samples",
            "Mean disease samples",
            "Local model accuracy",
            "Local model F1",
        ]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Performance across communication rounds")
    round_columns = st.columns(2)
    for column, filename, caption in (
        (round_columns[0], "accuracy_across_rounds.png", "Accuracy"),
        (round_columns[1], "f1_across_rounds.png", "F1 score"),
    ):
        image_path = selected_dir / filename
        if image_path.is_file():
            column.image(
                str(image_path),
                caption=f"{caption} by round; error bars show seed SD.",
            )
        else:
            column.error(f"Saved graph not found: {image_path}")
    alpha_plot = selected_dir / "performance_by_alpha.png"
    if alpha_plot.is_file():
        st.subheader("Performance variation across alpha values")
        st.image(
            str(alpha_plot),
            caption="Measured mean accuracy and F1 score across seeds; error bars show seed SD.",
        )
    st.caption(
        "Metrics are empirical results from the selected saved CSV bundle. "
        "Non-IID is not assumed to be worse; compare the measured values."
    )


CARDIO_INPUT_SPECS: dict[str, tuple[str, str, Any]] = {
    "age_years": ("Age", "years", (18, 100, 55)),
    "gender": ("Gender", "1 = female, 2 = male", (1, 2)),
    "height_cm": ("Height", "cm", (120, 220, 165)),
    "weight_kg": ("Weight", "kg", (30, 200, 72)),
    "systolic_bp": ("Systolic blood pressure", "mmHg", (70, 250, 120)),
    "diastolic_bp": ("Diastolic blood pressure", "mmHg", (40, 200, 80)),
    "cholesterol": ("Cholesterol", "1 = normal, 2 = above normal, 3 = well above", (1, 2, 3)),
    "glucose": ("Glucose", "1 = normal, 2 = above normal, 3 = well above", (1, 2, 3)),
    "smoke": ("Smokes", "0 = no, 1 = yes", (0, 1)),
    "alcohol": ("Alcohol intake", "0 = no, 1 = yes", (0, 1)),
    "active": ("Physically active", "0 = no, 1 = yes", (0, 1)),
}
SELECT_FEATURES = {
    "gender", "cholesterol", "glucose", "smoke", "alcohol", "active",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
}

CLEVELAND_INPUT_SPECS: dict[str, tuple[str, str, Any]] = {
    "age": ("Age", "years", (1, 120, 54)),
    "sex": ("Sex", "0 = female, 1 = male", (0, 1)),
    "cp": ("Chest-pain type", "Encoded category", (1, 2, 3, 4)),
    "trestbps": ("Resting blood pressure", "mmHg", (50, 300, 130)),
    "chol": ("Serum cholesterol", "mg/dl", (50, 700, 250)),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "0 = no, 1 = yes", (0, 1)),
    "restecg": ("Resting ECG", "Encoded category", (0, 1, 2)),
    "thalach": ("Maximum heart rate achieved", "beats per minute", (40, 250, 150)),
    "exang": ("Exercise-induced angina", "0 = no, 1 = yes", (0, 1)),
    "oldpeak": ("ST depression", "relative to rest", (0.0, 10.0, 1.0)),
    "slope": ("Peak exercise ST slope", "Encoded category", (1, 2, 3)),
    "ca": ("Major vessels colored by fluoroscopy", "Encoded category", (0, 1, 2, 3)),
    "thal": ("Thalassemia", "Encoded category", (3, 6, 7)),
}


def feature_spec(feature: str) -> tuple[str, str, Any]:
    if feature in CARDIO_INPUT_SPECS:
        return CARDIO_INPUT_SPECS[feature]
    return CLEVELAND_INPUT_SPECS.get(
        feature,
        (feature.replace("_", " ").title(), "model input", None),
    )


def patient_form(features: tuple[str, ...]) -> dict[str, float] | None:
    with st.form(f"patient_prediction_form_{RUN_DIR.name}"):
        st.markdown(
            f"Enter one record. Required model feature order: {', '.join(features)}."
        )
        columns = st.columns(3)
        values: dict[str, float] = {}
        for index, feature in enumerate(features):
            label, unit, choices = feature_spec(feature)
            with columns[index % len(columns)]:
                widget_label = f"{label} ({feature})"
                if isinstance(choices, tuple) and len(choices) >= 2:
                    if feature in SELECT_FEATURES:
                        selected = st.selectbox(
                            widget_label,
                            list(choices),
                            index=0,
                            help=unit,
                            key=f"{RUN_DIR.name}_{feature}",
                        )
                        values[feature] = float(selected)
                    else:
                        minimum, maximum, default = choices
                        values[feature] = float(st.number_input(
                            widget_label,
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(default),
                            help=unit,
                            key=f"{RUN_DIR.name}_{feature}",
                        ))
                else:
                    values[feature] = float(st.number_input(
                        widget_label,
                        value=0.0,
                        help=unit,
                        key=f"{RUN_DIR.name}_{feature}",
                    ))
        submitted = st.form_submit_button("Run saved global model", type="primary")
    return values if submitted else None


def validate_patient(
    values: dict[str, float],
    features: tuple[str, ...],
) -> list[str]:
    errors = []
    if set(values) != set(features):
        return ["Prediction inputs do not match the model's required feature schema."]
    if not all(np.isfinite(value) for value in values.values()):
        errors.append("Every field must contain a finite numeric value.")
    bounds = {
        "age_years": (18, 100),
        "gender": (1, 2),
        "height_cm": (120, 220),
        "weight_kg": (30, 200),
        "systolic_bp": (70, 250),
        "diastolic_bp": (40, 200),
        "cholesterol": (1, 3),
        "glucose": (1, 3),
        "smoke": (0, 1),
        "alcohol": (0, 1),
        "active": (0, 1),
        "age": (1, 120),
        "trestbps": (50, 300),
        "chol": (50, 700),
        "thalach": (40, 250),
        "oldpeak": (0, 10),
    }
    for name, (minimum, maximum) in bounds.items():
        if name in values and not minimum <= values[name] <= maximum:
            label = feature_spec(name)[0]
            errors.append(f"{label} must be between {minimum} and {maximum}.")
    return errors


def render_prediction() -> None:
    context = load_active_run_context()
    feature_names = context["feature_names"]
    target_labels = context["target_labels"]
    section_heading(
        "Interactive heart disease prediction",
        f"Predictions use this run's saved FedAvg model for {context.get('dataset', 'the selected dataset')}.",
    )
    st.markdown(
        '<div class="disclaimer"><strong>Educational demonstration only.</strong> This model '
        'is not a medical diagnosis, screening tool, or substitute for professional medical advice.</div>',
        unsafe_allow_html=True,
    )
    try:
        metadata = load_model_metadata(str(MODEL_PATH))
        st.success(
            f"Loaded saved model: {MODEL_PATH.relative_to(BASE_DIR)} · "
            f"{metadata['feature_count']} ordered features · training imputation and scaler parameters verified."
        )
    except DashboardDataError as error:
        show_data_error(error)
        st.stop()
    submitted_values = patient_form(feature_names)
    if submitted_values is None:
        st.caption(
            "Inputs follow the model's exact feature order: "
            + ", ".join(feature_names)
            + ". These predictions are for educational demonstration only."
        )
        return
    errors = validate_patient(submitted_values, feature_names)
    if errors:
        for error in errors:
            st.error(error)
        return
    patient = np.asarray(
        [submitted_values[name] for name in feature_names],
        dtype=float,
    )
    try:
        probability = project.predict_proba_from_raw(patient, model_path=MODEL_PATH)
        prediction = project.predict_from_raw(patient, model_path=MODEL_PATH)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Prediction failed: {error}")
        return
    prediction_label = target_labels[int(prediction)]
    if prediction == 1:
        st.warning(f"### Predicted class: {prediction_label}")
    else:
        st.success(f"### Predicted class: {prediction_label}")
    c1, c2 = st.columns(2)
    c1.metric(f"Probability: {target_labels[1]}", f"{probability:.1%}")
    c2.metric("Predicted class", prediction_label)
    st.progress(float(probability))
    st.caption(
        f"Class 1 threshold: probability ≥ 0.50. "
        "This output is for academic demonstration only."
    )


def render_project_information() -> None:
    context = load_active_run_context()
    section_heading("Project information", "Methodology, contribution, and limitations.")
    info_left, info_right = st.columns(2)
    with info_left:
        st.subheader("Dataset & technology")
        st.markdown(
            f"""
            - **Dataset:** {context.get('dataset', 'selected run')};
              {int(context.get('rows_used', context.get('original_rows', 0))):,} records used
              and {context['feature_count']} model features.
            - **Target:** {context.get('target', 'binary target')}; class labels are
              {', '.join(context['target_labels'])}.
            - **Preprocessing:** stratified 75/25 split; training-only median imputation and
              standardization, with saved parameters reused for predictions.
            - **Stack:** Python, NumPy, pandas, scikit-learn, Streamlit, and Plotly.
            """
        )
        st.subheader("Methodology")
        st.markdown(
            """
            Local Logistic Regression is compared with centralized learning and sample-weighted
            FedAvg. Differential privacy experiments are available only in runs that include them.
            Evaluation reports accuracy, precision, recall, and F1 on the shared held-out test set.
            """
        )
    with info_right:
        st.subheader("Research motivation & contributions")
        st.markdown(
            """
            The project demonstrates how collaborative model training can be studied when data
            is distributed across organizations. Contributions include non-IID client partitions,
            client dropout, scalability comparisons, privacy–utility experiments, RDP reports,
            reusable model artifacts, and an interactive prediction demonstration.
            """
        )
        st.subheader("Limitations & future work")
        st.markdown(
            """
            - The clients are simulated in one process on a small, historical dataset.
            - **Secure aggregation is not implemented.**
            - Where used, centralized imputation and scaling statistics are **not privatized**.
            - DP accounting is conditional on its documented assumptions and is not certified.
            - Results are not clinical evidence; external validation is required.

            Future work could use larger multi-site data, a formally audited DP framework,
            protected preprocessing, secure aggregation, stronger baselines, and external validation.
            """
        )
    st.subheader("Saved experiment bundle")
    st.code(str(RUN_DIR), language="text")
    st.caption(
        "This dashboard reads the existing result bundle and does not launch training or overwrite experiment outputs."
    )


def main() -> None:
    global RUN_DIR, MODEL_PATH, SELECTED_DATASET

    apply_styles()
    with st.sidebar:
        st.markdown("## 🫀 FL Heart Lab")
        st.caption("Academic project demonstration")
        selected_dataset = st.selectbox(
            "Dataset",
            DATASET_TYPES,
            index=1,
            key="active_dataset_type",
        )
        compatible_runs = compatible_run_dirs(selected_dataset)
        if not compatible_runs:
            expected_features = DATASET_FEATURES[selected_dataset]
            st.error(
                f"No compatible trained model was found for {selected_dataset}. "
                f"Expected a saved model with {len(expected_features)} features "
                f"in the required schema. Check the result bundles under {RESULTS_DIR}."
            )
            st.stop()

        if len(compatible_runs) > 1:
            selected_run = st.selectbox(
                "Training run",
                compatible_runs,
                index=0,
                format_func=run_label,
                key=(
                    "training_run_cleveland"
                    if selected_dataset == DATASET_TYPES[0]
                    else "training_run_cardio"
                ),
            )
        else:
            selected_run = compatible_runs[0]
            st.caption(f"Training run: {selected_run.name}")

        SELECTED_DATASET = selected_dataset
        RUN_DIR = selected_run
        MODEL_PATH = RUN_DIR / "global_model.npz"
        try:
            context = load_active_run_context()
        except DashboardDataError as error:
            st.error(f"Cannot load the selected dataset's training run: {error}")
            st.stop()

        original_rows = int(
            context.get("original_rows", context.get("rows_used", 0))
        )
        rows_used = int(context.get("rows_used", original_rows))
        st.markdown("**Selected dataset**")
        st.write(selected_dataset)
        if rows_used != original_rows:
            st.caption(
                f"Records: {original_rows:,} original · "
                f"{rows_used:,} used after filtering"
            )
        else:
            st.caption(f"Records: {rows_used:,}")
        st.caption(f"Features: {context['feature_count']}")
        st.caption(f"Selected run: {RUN_DIR.name}")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Educational simulation · not a clinical tool")

    renderers = {
        "Overview": render_overview,
        "Federated learning": render_federated_learning,
        "Model comparison": render_model_comparison,
        "Differential privacy": render_differential_privacy,
        "Additional experiments": render_additional_experiments,
        "IID vs Non-IID Analysis": render_iid_noniid_analysis,
        "Heart disease prediction": render_prediction,
        "Project information": render_project_information,
    }
    renderers[page]()


if __name__ == "__main__":
    main()
