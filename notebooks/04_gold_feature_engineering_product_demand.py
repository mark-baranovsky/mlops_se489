import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_PATH = os.path.join(
    BASE_DIR, "data", "processed", "gold_weekly_product_demand.parquet"
)
FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "gold_weekly_product_demand_features.parquet"
)

print(f"Input  : {GOLD_PATH}")
print(f"Output : {FEATURES_PATH}")

df = pd.read_parquet(GOLD_PATH)
df = df.sort_values(["warehouse", "product_code", "week_start_date"]).reset_index(
    drop=True
)

print(f"\nRows loaded from Gold : {len(df):,}")
print(df.head(5).to_string())

entity_cols = ["warehouse", "product_code"]

df["lag_1_week_demand"] = df.groupby(entity_cols)["weekly_order_demand"].shift(1)
df["lag_2_week_demand"] = df.groupby(entity_cols)["weekly_order_demand"].shift(2)
df["lag_3_week_demand"] = df.groupby(entity_cols)["weekly_order_demand"].shift(3)
df["lag_4_week_demand"] = df.groupby(entity_cols)["weekly_order_demand"].shift(4)
df["lag_52_week_demand"] = df.groupby(entity_cols)["weekly_order_demand"].shift(52)

df["rolling_4wk_avg_demand"] = df.groupby(entity_cols)["weekly_order_demand"].transform(
    lambda x: x.shift(1).rolling(4).mean()
)
df["rolling_4wk_std_demand"] = df.groupby(entity_cols)["weekly_order_demand"].transform(
    lambda x: x.shift(1).rolling(4).std()
)
df["rolling_8wk_avg_demand"] = df.groupby(entity_cols)["weekly_order_demand"].transform(
    lambda x: x.shift(1).rolling(8).mean()
)

df["demand_momentum"] = df["lag_1_week_demand"] - df["rolling_4wk_avg_demand"]

df["yoy_demand_change"] = df["lag_1_week_demand"] - df["lag_52_week_demand"]

df["yoy_demand_pct_change"] = df.apply(
    lambda row: (
        (row["lag_1_week_demand"] - row["lag_52_week_demand"])
        / row["lag_52_week_demand"]
        * 100
        if pd.notna(row["lag_52_week_demand"]) and row["lag_52_week_demand"] != 0
        else None
    ),
    axis=1,
)

df["quarter"] = pd.to_datetime(df["week_start_date"]).dt.quarter
df["is_quarter_end_week"] = df["week_of_year"].isin([13, 26, 39, 52]).astype(int)
df["is_peak_month"] = df["month"].isin([11, 12, 1]).astype(int)

feature_cols = [
    "warehouse",
    "product_code",
    "product_category",
    "week_start_date",
    "weekly_order_demand",
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
    "has_returns",
]

df_features = df[feature_cols].copy()
df_features["features_processed_at"] = datetime.utcnow().isoformat()

print("\nFeature null report:")
print(f"{'Feature':<35} {'Nulls':>8}  {'% null':>8}")
print("-" * 57)
total = len(df_features)
for c in feature_cols:
    n = df_features[c].isna().sum()
    pct = n / total * 100
    print(f"  {c:<35} {n:>8,}  {pct:>7.1f}%")

os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
df_features.to_parquet(FEATURES_PATH, index=False)

print(f"\nFeatures file written : {FEATURES_PATH}")
print(f"Rows                  : {len(df_features):,}")
print(f"Columns               : {len(df_features.columns)}")
