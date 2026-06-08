import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import mlflow
except ImportError:
    mlflow = None


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "gold_weekly_product_demand_features.parquet",
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHAMPION_PATH = os.path.join(MODELS_DIR, "champion_model.pkl")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
CML_REPORT_PATH = os.path.join(REPORTS_DIR, "cml_report.md")
MLFLOW_METRICS_PATH = os.path.join(REPORTS_DIR, "mlflow_metrics.csv")

SPLIT_DATE = "2016-10-01"
MLFLOW_EXPERIMENT_NAME = "demand_forecast_v1"

FEATURE_COLS = [
    "lag_1_week_demand",
    "lag_2_week_demand",
    "lag_3_week_demand",
    "lag_4_week_demand",
    "lag_52_week_demand",
    "rolling_4wk_avg_demand",
    "rolling_4wk_std_demand",
    "rolling_8wk_avg_demand",
    "demand_momentum",
    "yoy_demand_change",
    "yoy_demand_pct_change",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "is_quarter_end_week",
    "is_peak_month",
    "order_count",
]

TARGET_COL = "weekly_order_demand"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_validation_data():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    df_train = df[df["week_start_date"] < SPLIT_DATE].copy()
    df_val = df[df["week_start_date"] >= SPLIT_DATE].copy()

    print(f"Training rows: {len(df_train):,}")
    print(f"Validation rows: {len(df_val):,}")

    assert len(df_train) > 0, "Training set is empty — move SPLIT_DATE later"
    assert len(df_val) > 0, "Validation set is empty — move SPLIT_DATE earlier"

    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]

    return X_val, y_val


def make_actual_vs_predicted_plot(y_val, preds):
    plot_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_val, preds, alpha=0.6)
    plt.xlabel("Actual Weekly Order Demand")
    plt.ylabel("Predicted Weekly Order Demand")
    plt.title("Actual vs Predicted - Champion Model")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def make_feature_importance_plot(model):
    if not hasattr(model, "named_steps"):
        print("Champion model is not a sklearn Pipeline. Skipping feature importance plot.")
        return None

    model_step = model.named_steps.get("model")

    importances = pd.Series(
        model_step.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    top_importances = importances.head(10)
    plot_path = os.path.join(FIGURES_DIR, "feature_importance.png")

    plt.figure(figsize=(8, 5))
    top_importances.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances - Champion Model")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def make_residual_plot(y_val, preds):
    residuals = y_val - preds
    plot_path = os.path.join(FIGURES_DIR, "residual_plot.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Weekly Order Demand")
    plt.ylabel("Residuals")
    plt.title("Residual Plot - Champion Model")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def read_peak_memory_usage():
    memory_profile_path = os.path.join(
        REPORTS_DIR,
        "profiling",
        "training_memory_usage.txt",
    )

    if not os.path.exists(memory_profile_path):
        print(f"Memory profiling file not found: {memory_profile_path}")
        return None

    try:
        with open(memory_profile_path, encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"Peak memory MiB:\s*([\d.]+)", content)

        if match:
            return float(match.group(1))

    except Exception as e:
        print(f"Error reading memory profiling file: {e}")

    return None


def fetch_mlflow_model_metrics():
    if mlflow is None:
        print("MLflow not available.")
        return pd.DataFrame()

    try:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            order_by=["start_time DESC"],
        )

        records = []

        for _, run in runs.iterrows():
            run_name = run.get("tags.mlflow.runName", "")
            model_type = run.get("params.model_type", "")
            model_name = model_type if model_type else run_name

            train_rmse = run.get("metrics.train_rmse", np.nan)
            val_rmse = run.get("metrics.val_rmse", np.nan)
            val_mae = run.get("metrics.val_mae", np.nan)
            is_champion = run.get("tags.is_champion", "false")
            run_id = run.get("run_id", "")

            if pd.isna(val_rmse):
                continue

            records.append(
                {
                    "run_name": run_name,
                    "model": model_name,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "is_champion": is_champion,
                    "run_id": run_id,
                }
            )

        if not records:
            print("No MLflow runs with validation RMSE were found.")
            return pd.DataFrame()

        metrics_df = pd.DataFrame(records)

        # Keep latest run for each model type.
        metrics_df = metrics_df.drop_duplicates(subset=["model"], keep="first")
        metrics_df = metrics_df.sort_values("val_rmse").reset_index(drop=True)

        os.makedirs(os.path.dirname(MLFLOW_METRICS_PATH), exist_ok=True)
        metrics_df.to_csv(MLFLOW_METRICS_PATH, index=False)

        print(f"MLflow metrics saved to: {MLFLOW_METRICS_PATH}")
        print(metrics_df)

        return metrics_df

    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")
        return pd.DataFrame()


def find_champion_from_mlflow(mlflow_metrics_df, fallback_metrics):
    if mlflow_metrics_df.empty:
        return {
            "model": "champion_model.pkl",
            "val_rmse": fallback_metrics["val_rmse"],
            "val_mae": fallback_metrics["val_mae"],
        }

    champion_rows = mlflow_metrics_df[mlflow_metrics_df["is_champion"].astype(str).str.lower() == "true"]

    if not champion_rows.empty:
        champion_row = champion_rows.iloc[0]
    else:
        champion_row = mlflow_metrics_df.sort_values("val_rmse").iloc[0]

    return {
        "model": champion_row["model"],
        "val_rmse": champion_row["val_rmse"],
        "val_mae": champion_row["val_mae"],
    }


def write_cml_report(champion_eval_metrics, plot_paths, peak_memory, mlflow_metrics_df):
    champion = find_champion_from_mlflow(mlflow_metrics_df, champion_eval_metrics)

    report = "# CML Model Training Report\n\n"

    report += "## Pipeline Status\n\n"
    report += "- Data pipeline: completed\n"
    report += "- Feature pipeline: completed\n"
    report += "- Model training: completed\n"
    report += "- CML plots: completed\n"
    report += "- Profiling: checked\n\n"

    report += "## Model Metrics from MLflow\n\n"

    if not mlflow_metrics_df.empty:
        report += "| Model | Train RMSE | Validation RMSE | Validation MAE |\n"

        for _, row in mlflow_metrics_df.iterrows():
            train_rmse = row["train_rmse"]
            val_rmse = row["val_rmse"]
            val_mae = row["val_mae"]

            train_rmse_text = "N/A" if pd.isna(train_rmse) else f"{train_rmse:,.1f}"
            val_rmse_text = f"{val_rmse:,.1f}"
            val_mae_text = f"{val_mae:,.1f}"

            report += f"| {row['model']} | {train_rmse_text} | {val_rmse_text} | {val_mae_text} |\n "

        report += "\n"
    else:
        report += "No MLflow metrics were found.\n\n"

    report += "## Champion Model\n\n"
    report += f"- Champion model: **{champion['model']}**\n"
    report += f"- Champion validation RMSE: **{champion['val_rmse']:,.1f}**\n"
    report += f"- Champion validation MAE: **{champion['val_mae']:,.1f}**\n\n"

    report += "## Champion Model Re-Evaluation\n\n"
    report += (
        "The saved champion model was loaded from `models/champion_model.pkl` "
        "and re-evaluated on the validation set.\n\n"
    )
    report += "| Metric | Value |\n"
    report += f"|Validation RMSE | {champion_eval_metrics['val_rmse']:,.1f} |\n"
    report += f"| Validation MAE | {champion_eval_metrics['val_mae']:,.1f} |\n\n"

    report += "## Profiling Summary\n\n"
    report += "- CPU profiling completed successfully if profiling records are present.\n"
    report += "- Memory profiling checked.\n"

    report += f"-Peak memory usage: **{peak_memory:.2f} MiB**.\n\n"
    report += "Profiling outputs are saved in `reports/profiling/`.\n\n"

    report += "## Plots\n\n"
    report += "All plots are saved in `reports/figures/`.\n\n"

    for title, path in plot_paths.items():
        if path is not None:
            relative_path = os.path.relpath(path, BASE_DIR)
            report += f"### {title}\n\n"
            report += f"![{title}]({relative_path})\n\n"

    report += "This report is posted automatically by the CML workflow on pull requests.\n"

    with open(CML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    champion_model = joblib.load(CHAMPION_PATH)

    X_val, y_val = load_validation_data()
    preds = champion_model.predict(X_val)

    champion_eval_metrics = {
        "val_rmse": rmse(y_val, preds),
        "val_mae": mean_absolute_error(y_val, preds),
    }

    actual_vs_predicted_path = make_actual_vs_predicted_plot(y_val, preds)
    feature_importance_path = make_feature_importance_plot(champion_model)
    residual_plot_path = make_residual_plot(y_val, preds)

    plot_paths = {
        "Actual vs Predicted": actual_vs_predicted_path,
        "Feature Importance": feature_importance_path,
        "Residual Plot": residual_plot_path,
    }

    peak_memory = read_peak_memory_usage()
    mlflow_metrics_df = fetch_mlflow_model_metrics()

    write_cml_report(
        champion_eval_metrics,
        plot_paths,
        peak_memory,
        mlflow_metrics_df,
    )


if __name__ == "__main__":
    main()
