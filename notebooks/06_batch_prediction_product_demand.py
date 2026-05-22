import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cProfile
import pstats

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "gold_weekly_product_demand_features.parquet")
PRED_PATH     = os.path.join(BASE_DIR, "data", "processed", "demand_predictions.parquet")
CHAMPION_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")
CHAMPION_NAME = "champion"
CHAMPION_RMSE = 0.0


# Start profiling
profiler = cProfile.Profile()
profiler.enable()


print(f"Features file       : {FEATURES_PATH}")
print(f"Predictions file    : {PRED_PATH}")
print(f"Champion model path : {CHAMPION_PATH}")

champion_model = joblib.load(CHAMPION_PATH)
print(f"\nChampion model loaded : {type(champion_model.named_steps['model']).__name__}")

df = pd.read_parquet(FEATURES_PATH)
df["week_start_date"] = pd.to_datetime(df["week_start_date"])

print(f"\nFeatures rows : {len(df):,}")
print(f"Date range    : {df['week_start_date'].min()} to {df['week_start_date'].max()}")

latest_week = df["week_start_date"].max()
next_week   = latest_week + timedelta(days=7)

print(f"\nLatest week in features : {latest_week.date()}")
print(f"Predicting for week     : {next_week.date()}")

df_latest = df[df["week_start_date"] == latest_week].copy()
print(f"Rows in latest week     : {len(df_latest):,}")

FEATURE_COLS = [
    "lag_1_week_demand", "lag_2_week_demand", "lag_3_week_demand",
    "lag_4_week_demand", "lag_52_week_demand",
    "rolling_4wk_avg_demand", "rolling_4wk_std_demand",
    "rolling_8wk_avg_demand", "demand_momentum",
    "yoy_demand_change", "yoy_demand_pct_change",
    "year", "month", "quarter", "week_of_year",
    "is_quarter_end_week", "is_peak_month", "order_count",
]

X_latest  = df_latest[FEATURE_COLS]
raw_preds = champion_model.predict(X_latest)

df_latest["predicted_demand"]           = np.maximum(np.round(raw_preds).astype(int), 0)
df_latest["prediction_week_start_date"] = next_week
df_latest["features_as_of_week"]        = latest_week
df_latest["model_name"]                 = CHAMPION_NAME
df_latest["model_val_rmse"]             = CHAMPION_RMSE
df_latest["predicted_at"]               = datetime.utcnow().isoformat()

df_preds = df_latest[[
    "warehouse", "product_code", "product_category",
    "prediction_week_start_date", "predicted_demand",
    "features_as_of_week", "weekly_order_demand",
    "model_name", "model_val_rmse", "predicted_at",
]].copy()

df_preds = df_preds.sort_values(["warehouse", "product_code"]).reset_index(drop=True)

print(f"\nFinal prediction rows : {len(df_preds):,}")

print("\nSanity checks:")
print(f"  Null predicted_demand : {df_preds['predicted_demand'].isna().sum()}")
print(f"  Negative predictions  : {(df_preds['predicted_demand'] < 0).sum()}")
print(f"  Prediction week       : {df_preds['prediction_week_start_date'].unique()}")
print(f"  Min predicted_demand  : {df_preds['predicted_demand'].min():,}")
print(f"  Max predicted_demand  : {df_preds['predicted_demand'].max():,}")
print(f"  Avg predicted_demand  : {df_preds['predicted_demand'].mean():,.1f}")

print("\nTop 10 highest predicted demand:")
print(df_preds.nlargest(10, "predicted_demand")[[
    "warehouse", "product_code", "prediction_week_start_date", "predicted_demand", "weekly_order_demand"
]].to_string())

os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
df_preds.to_parquet(PRED_PATH, index=False)

# Stop profiling
profiler.disable()

stats = pstats.Stats(profiler)
print(f"Total Time: {stats.total_tt} seconds")

profiler.dump_stats("data\processed\profiler_batch_prediction.prof")

