import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SILVER_PATH = os.path.join(BASE_DIR, "data", "interim", "silver_product_demand_clean.parquet")
GOLD_PATH = os.path.join(BASE_DIR, "data", "processed", "gold_weekly_product_demand.parquet")

print(f"Input  : {SILVER_PATH}")
print(f"Output : {GOLD_PATH}")

df = pd.read_parquet(SILVER_PATH)

print(f"\nRows loaded from Silver : {len(df):,}")
print(f"Date range              : {df['order_date'].min()} to {df['order_date'].max()}")
print(f"Distinct warehouses     : {df['warehouse'].nunique()}")
print(f"Distinct products       : {df['product_code'].nunique()}")

df["demand_type"] = df["order_demand"].apply(lambda x: "return" if x < 0 else "order")

print("\nDemand type split:")
print(df.groupby("demand_type")["order_demand"].agg(["count", "sum"]))

group_cols = ["warehouse", "product_code", "product_category", "week_start_date", "year", "month", "week_of_year"]

orders = df[df["demand_type"] == "order"]
returns = df[df["demand_type"] == "return"]

weekly_orders = (
    orders.groupby(group_cols)["order_demand"]
    .agg(
        weekly_order_demand="sum",
        avg_daily_order_demand="mean",
        max_daily_order_demand="max",
        min_daily_order_demand="min",
        positive_order_count="count",
    )
    .reset_index()
)

weekly_returns = returns.groupby(group_cols)["order_demand"].agg(weekly_return_demand="sum").reset_index()

weekly_all = df.groupby(group_cols)["order_demand"].agg(weekly_net_demand="sum", order_count="count").reset_index()

has_returns = returns.groupby(group_cols).size().reset_index(name="return_count")
has_returns["has_returns"] = True

df_gold = weekly_all.merge(weekly_orders, on=group_cols, how="left")
df_gold = df_gold.merge(weekly_returns, on=group_cols, how="left")
df_gold = df_gold.merge(has_returns[group_cols + ["has_returns"]], on=group_cols, how="left")

df_gold["weekly_order_demand"] = df_gold["weekly_order_demand"].fillna(0)
df_gold["weekly_return_demand"] = df_gold["weekly_return_demand"].fillna(0)
df_gold["has_returns"] = df_gold["has_returns"].fillna(False)

df_gold = df_gold.sort_values(["warehouse", "product_code", "week_start_date"]).reset_index(drop=True)
df_gold["gold_processed_at"] = datetime.utcnow().isoformat()

print(f"\nGold rows : {len(df_gold):,}")

print("\nSanity checks:")
print(f"  Negative weekly_order_demand : {(df_gold['weekly_order_demand'] < 0).sum()}")
for col in ["warehouse", "product_code", "week_start_date", "weekly_order_demand"]:
    print(f"  {col:35s} -> {df_gold[col].isna().sum():,} nulls")

total = len(df_gold)
distinct_keys = df_gold[["warehouse", "product_code", "week_start_date"]].drop_duplicates().shape[0]
print(f"\n  Total rows    : {total:,}")
print(f"  Distinct keys : {distinct_keys:,}")
print(f"  Duplicates    : {total - distinct_keys:,}")

print(
    f"\n  weekly_order_demand  min={df_gold['weekly_order_demand'].min():,.0f}  "
    f"max={df_gold['weekly_order_demand'].max():,.0f}  "
    f"avg={df_gold['weekly_order_demand'].mean():,.1f}"
)

os.makedirs(os.path.dirname(GOLD_PATH), exist_ok=True)
df_gold.to_parquet(GOLD_PATH, index=False)

print(f"\nGold file written : {GOLD_PATH}")
print("\nWeekly demand summary by warehouse:")
print(df_gold.groupby("warehouse")["weekly_order_demand"].agg(["count", "sum", "mean"]).to_string())
