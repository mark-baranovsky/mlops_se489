import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "gold_weekly_product_demand_features.parquet")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
SPLIT_DATE    = "2016-10-01"

os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Features file : {FEATURES_PATH}")
print(f"Split date    : {SPLIT_DATE}")

df = pd.read_parquet(FEATURES_PATH)
df["week_start_date"] = pd.to_datetime(df["week_start_date"])

print(f"\nTotal rows : {len(df):,}")
print(f"Date range : {df['week_start_date'].min()} to {df['week_start_date'].max()}")
print(f"Warehouses : {df['warehouse'].nunique()}")
print(f"Products   : {df['product_code'].nunique()}")
print(df.head(5).to_string())

FEATURE_COLS = [
    "lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand",
    "lag_4_week_demand", "lag_52_week_demand",
    "rolling_4wk_avg_demand", "rolling_4wk_std_demand",
    "rolling_8wk_avg_demand", "demand_momentum",
    "yoy_demand_change", "yoy_demand_pct_change",
    "year", "month", "quarter", "week_of_year",
    "is_quarter_end_week", "is_peak_month", "order_count",
]
TARGET_COL = "weekly_order_demand"

df_train = df[df["week_start_date"] <  SPLIT_DATE].copy()
df_val   = df[df["week_start_date"] >= SPLIT_DATE].copy()

print(f"\nTrain rows : {len(df_train):,}")
print(f"Val rows   : {len(df_val):,}")

assert len(df_val)   > 0, "Validation set is empty — move SPLIT_DATE earlier"
assert len(df_train) > 0, "Training set is empty — move SPLIT_DATE later"

X_train = df_train[FEATURE_COLS]
y_train = df_train[TARGET_COL]
X_val   = df_val[FEATURE_COLS]
y_val   = df_val[TARGET_COL]

mlflow.set_experiment("demand_forecast_v1")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

with mlflow.start_run(run_name="baseline_mean_lag"):
    lag_cols    = ["lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand", "lag_4_week_demand"]
    preds_train = df_train[lag_cols].fillna(0).mean(axis=1)
    preds_val   = df_val[lag_cols].fillna(0).mean(axis=1)

    train_rmse = rmse(y_train, preds_train)
    val_rmse   = rmse(y_val,   preds_val)
    val_mae    = mean_absolute_error(y_val, preds_val)

    mlflow.log_param("model_type", "baseline_mean_lag")
    mlflow.log_param("split_date", SPLIT_DATE)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("val_rmse",   val_rmse)
    mlflow.log_metric("val_mae",    val_mae)
    mlflow.set_tag("is_champion", "false")

    baseline_val_rmse = val_rmse
    baseline_run_id   = mlflow.active_run().info.run_id
    print(f"Baseline      ->  train RMSE: {train_rmse:,.1f}  |  val RMSE: {val_rmse:,.1f}  |  val MAE: {val_mae:,.1f}")

rf_params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}

with mlflow.start_run(run_name="random_forest"):
    pipeline_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(**rf_params))
    ])
    pipeline_rf.fit(X_train, y_train)

    train_rmse_rf = rmse(y_train, pipeline_rf.predict(X_train))
    val_rmse_rf   = rmse(y_val,   pipeline_rf.predict(X_val))
    val_mae_rf    = mean_absolute_error(y_val, pipeline_rf.predict(X_val))

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("split_date", SPLIT_DATE)
    for k, v in rf_params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("train_rmse", train_rmse_rf)
    mlflow.log_metric("val_rmse",   val_rmse_rf)
    mlflow.log_metric("val_mae",    val_mae_rf)
    mlflow.sklearn.log_model(pipeline_rf, "random_forest_model")
    mlflow.set_tag("is_champion", "false")

    rf_val_rmse = val_rmse_rf
    rf_run_id   = mlflow.active_run().info.run_id
    print(f"Random Forest ->  train RMSE: {train_rmse_rf:,.1f}  |  val RMSE: {val_rmse_rf:,.1f}  |  val MAE: {val_mae_rf:,.1f}")

gbt_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}

with mlflow.start_run(run_name="gradient_boosting"):
    pipeline_gbt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   GradientBoostingRegressor(**gbt_params))
    ])
    pipeline_gbt.fit(X_train, y_train)

    train_rmse_gbt = rmse(y_train, pipeline_gbt.predict(X_train))
    val_rmse_gbt   = rmse(y_val,   pipeline_gbt.predict(X_val))
    val_mae_gbt    = mean_absolute_error(y_val, pipeline_gbt.predict(X_val))

    mlflow.log_param("model_type", "gradient_boosting")
    mlflow.log_param("split_date", SPLIT_DATE)
    for k, v in gbt_params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("train_rmse", train_rmse_gbt)
    mlflow.log_metric("val_rmse",   val_rmse_gbt)
    mlflow.log_metric("val_mae",    val_mae_gbt)
    mlflow.sklearn.log_model(pipeline_gbt, "gradient_boosting_model")
    mlflow.set_tag("is_champion", "false")

    gbt_val_rmse = val_rmse_gbt
    gbt_run_id   = mlflow.active_run().info.run_id
    print(f"GBT           ->  train RMSE: {train_rmse_gbt:,.1f}  |  val RMSE: {val_rmse_gbt:,.1f}  |  val MAE: {val_mae_gbt:,.1f}")

top_product = df_train.groupby("product_code")["weekly_order_demand"].count().idxmax()

df_prophet_train = (
    df_train[df_train["product_code"] == top_product][["week_start_date", "weekly_order_demand"]]
    .rename(columns={"week_start_date": "ds", "weekly_order_demand": "y"})
    .dropna()
)
df_prophet_val = (
    df_val[df_val["product_code"] == top_product][["week_start_date", "weekly_order_demand"]]
    .rename(columns={"week_start_date": "ds", "weekly_order_demand": "y"})
    .dropna()
)

with mlflow.start_run(run_name="prophet_top_product"):
    prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(df_prophet_train)

    future    = prophet_model.make_future_dataframe(periods=len(df_prophet_val), freq="W")
    forecast  = prophet_model.predict(future)
    val_preds = forecast.tail(len(df_prophet_val))["yhat"].values

    if len(val_preds) == len(df_prophet_val):
        prophet_rmse = rmse(df_prophet_val["y"].values, val_preds)
        prophet_mae  = mean_absolute_error(df_prophet_val["y"].values, val_preds)
    else:
        prophet_rmse = float("nan")
        prophet_mae  = float("nan")

    mlflow.log_param("model_type", "prophet")
    mlflow.log_param("product",    top_product)
    mlflow.log_param("split_date", SPLIT_DATE)
    mlflow.log_metric("val_rmse",  prophet_rmse)
    mlflow.log_metric("val_mae",   prophet_mae)
    mlflow.set_tag("is_champion",  "false")
    mlflow.set_tag("scope",        "single_product_demo")

    print(f"Prophet       ->  product={top_product}  |  val RMSE: {prophet_rmse:,.1f}  |  val MAE: {prophet_mae:,.1f}")

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Model':<30} {'val RMSE':>10}")
print("  " + "-" * 42)
print(f"  {'baseline_mean_lag':<30} {baseline_val_rmse:>10,.1f}")
print(f"  {'random_forest':<30} {rf_val_rmse:>10,.1f}")
print(f"  {'gradient_boosting':<30} {gbt_val_rmse:>10,.1f}")

results = {
    "random_forest"     : (rf_val_rmse,  rf_run_id,  pipeline_rf),
    "gradient_boosting" : (gbt_val_rmse, gbt_run_id, pipeline_gbt),
}

champion_name   = min(results, key=lambda k: results[k][0])
champion_rmse   = results[champion_name][0]
champion_run_id = results[champion_name][1]
champion_model  = results[champion_name][2]

print(f"\n  Champion : {champion_name}")
print(f"  Val RMSE : {champion_rmse:,.1f}")

CHAMPION_PATH = os.path.join(MODELS_DIR, "champion_model.pkl")
joblib.dump(champion_model, CHAMPION_PATH)
print(f"\nChampion model saved to : {CHAMPION_PATH}")

client = mlflow.tracking.MlflowClient()
client.set_tag(champion_run_id, "is_champion", "true")
