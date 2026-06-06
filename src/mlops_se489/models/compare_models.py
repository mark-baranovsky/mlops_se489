"""Model comparison visualization for MLflow experiment tracking.

This module generates a bar chart comparing all model runs by validation RMSE
and logs it as an artifact to the active MLflow experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.patches import Patch

from mlops_se489.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = BASE_DIR / "reports" / "figures"
EXPERIMENT_NAME = "demand_forecast_v1"


def generate_comparison_chart(output_path: Path = REPORTS_DIR) -> str:
    """Generate a bar chart comparing all model runs by validation RMSE.

    Args:
        output_path: Directory where the chart image will be saved.

    Returns:
        String path to the saved chart image.
    """
    logger.info("Fetching runs from MLflow experiment: %s", EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error("Experiment '%s' not found. Run training first.", EXPERIMENT_NAME)
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"],
    )

    if not runs:
        logger.error("No runs found in experiment '%s'.", EXPERIMENT_NAME)
        raise ValueError("No runs found in experiment. Run training first.")

    data = []
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        val_rmse = run.data.metrics.get("val_rmse")
        is_champion = run.data.tags.get("is_champion", "false")

        if val_rmse is not None:
            data.append(
                {
                    "model": name,
                    "val_rmse": val_rmse,
                    "is_champion": is_champion == "true",
                }
            )

    if not data:
        logger.error("No val_rmse metrics found in MLflow runs.")
        raise ValueError("No val_rmse metrics found in runs.")

    df = pd.DataFrame(data).sort_values("val_rmse")
    logger.info("Found %d runs with val_rmse metrics", len(df))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71" if champion else "#3498db" for champion in df["is_champion"]]
    bars = ax.barh(df["model"], df["val_rmse"], color=colors)

    for bar, val in zip(bars, df["val_rmse"], strict=False):
        ax.text(
            bar.get_width() + 100,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Validation RMSE (lower is better)", fontsize=12)
    ax.set_title("Model Comparison — Validation RMSE", fontsize=14, fontweight="bold")
    ax.axvline(
        x=df["val_rmse"].min(),
        color="red",
        linestyle="--",
        alpha=0.3,
        label="Best RMSE",
    )

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Champion"),
        Patch(facecolor="#3498db", label="Other models"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "model_comparison_val_rmse.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Chart saved to %s", chart_path)

    with mlflow.start_run(run_name="model_comparison_chart"):
        mlflow.log_artifact(str(chart_path), artifact_path="charts")
        mlflow.log_table(
            data=df.to_dict(orient="records"),
            artifact_file="model_comparison.json",
        )
        logger.info("Chart and comparison table logged to MLflow")

    return str(chart_path)


if __name__ == "__main__":
    path = generate_comparison_chart()
    print(f"Chart saved to: {path}")
