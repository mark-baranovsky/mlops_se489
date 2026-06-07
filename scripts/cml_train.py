from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = BASE_DIR / "data" / "processed" / "gold_weekly_product_demand_features.parquet"
CHAMPION_PATH = BASE_DIR / "models" / "champion_model.pkl"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CML_REPORT_PATH = REPORTS_DIR / "cml_report.md"

SPLIT_DATE = "2016-10-01"

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
    if not FEATURES_PATH.exists():
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
    plt.figure(figsize=(7, 5))
    plt.scatter(y_val, preds, alpha=0.6)
    plt.xlabel("Actual Weekly Order Demand")
    plt.ylabel("Predicted Weekly Order Demand")
    plt.title("Actual vs Predicted - Champion Model")
    plt.tight_layout()

    plot_path = FIGURES_DIR / "actual_vs_predicted.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def make_feature_importance_plot(model):
    if not hasattr(model, "named_steps"):
        print("Champion model is not a sklearn Pipeline. Skipping feature importance plot.")
        return None

    model_step = model.named_steps.get("model")

    if model_step is None:
        print("No model step found in champion Pipeline. Skipping feature importance plot.")
        return None

    if not hasattr(model_step, "feature_importances_"):
        print("Champion model does not support feature_importances_. Skipping feature importance plot.")
        return None

    importances = pd.Series(
        model_step.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    top_importances = importances.head(10)

    plt.figure(figsize=(8, 5))
    top_importances.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances - Champion Model")
    plt.tight_layout()

    plot_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def make_residual_plot(y_val, preds):
    residuals = y_val - preds

    plt.figure(figsize=(7, 5))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Weekly Order Demand")
    plt.ylabel("Residuals")
    plt.title("Residual Plot - Champion Model")
    plt.tight_layout()

    plot_path = FIGURES_DIR / "residual_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def write_cml_report(metrics, plot_paths):
    report = "# CML Model Report\n\n"

    report += "## What ran\n\n"
    report += "The GitHub Actions workflow trained the model using:\n\n"
    report += "`notebooks/05_model_training_product_demand.py`\n\n"

    report += "The champion model was loaded from:\n\n"
    report += "`models/champion_model.pkl`\n\n"

    report += "## Champion Model Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "|---|---:|\n"
    report += f"| Validation RMSE | {metrics['val_rmse']:.4f} |\n"
    report += f"| Validation MAE | {metrics['val_mae']:.4f} |\n\n"

    report += "## CML Plots\n\n"
    report += "All plots are saved in `reports/figures/`.\n\n"

    for title, path in plot_paths.items():
        if path is not None:
            relative_path = path.relative_to(BASE_DIR)
            report += f"### {title}\n\n"
            report += f"![{title}]({relative_path})\n\n"

    CML_REPORT_PATH.write_text(report)

    print("===== CML REPORT CREATED =====")
    print(report)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found: {CHAMPION_PATH}. Run notebooks/05_model_training_product_demand.py first."
        )

    champion_model = joblib.load(CHAMPION_PATH)

    X_val, y_val = load_validation_data()
    preds = champion_model.predict(X_val)

    metrics = {
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

    write_cml_report(metrics, plot_paths)


if __name__ == "__main__":
    main()
