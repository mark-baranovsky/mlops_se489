import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRONZE_PATH = os.path.join(
    BASE_DIR, "data", "interim", "bronze_product_demand_raw.parquet"
)
SILVER_PATH = os.path.join(
    BASE_DIR, "data", "interim", "silver_product_demand_clean.parquet"
)

print(f"Input  : {BRONZE_PATH}")
print(f"Output : {SILVER_PATH}")

df = pd.read_parquet(BRONZE_PATH)
print(f"\nRows loaded from Bronze : {len(df):,}")
print(df.head(10).to_string())

df = df.rename(
    columns={
        "Product_Code": "product_code",
        "Warehouse": "warehouse",
        "Product_Category": "product_category",
        "Date": "order_date_raw",
        "Order_Demand": "order_demand_raw",
    }
)

df = df.drop(
    columns=["source_file", "ingestion_date", "ingestion_timestamp"], errors="ignore"
)

print("\nSample raw order_demand values:")
print(df["order_demand_raw"].dropna().unique()[:30])

df["order_demand_str"] = df["order_demand_raw"].str.strip()

mask = df["order_demand_str"].str.startswith("(", na=False)
df.loc[mask, "order_demand_str"] = "-" + df.loc[mask, "order_demand_str"].str.replace(
    r"[()]", "", regex=True
)

df["order_demand"] = pd.to_numeric(df["order_demand_str"], errors="coerce")
df["order_date"] = pd.to_datetime(df["order_date_raw"], errors="coerce")

total = len(df)
failed_dates = df["order_date"].isna().sum()
print(f"\nTotal rows          : {total:,}")
print(f"Failed date parses  : {failed_dates:,}")
print(f"Parse success rate  : {((total - failed_dates) / total * 100):.2f}%")

df = df.dropna(subset=["order_date", "order_demand", "product_code", "warehouse"])
df = df[df["order_demand"] != 0]

print(f"\nRows after cleaning : {len(df):,}")

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
df["week_start_date"] = df["order_date"] - pd.to_timedelta(
    df["order_date"].dt.weekday, unit="d"
)
df["week_start_date"] = df["week_start_date"].dt.normalize()

df_silver = df[
    [
        "product_code",
        "warehouse",
        "product_category",
        "order_date",
        "order_demand",
        "year",
        "month",
        "week_of_year",
        "week_start_date",
    ]
].copy()

df_silver["silver_processed_at"] = datetime.utcnow().isoformat()

print("\nFinal Silver schema:")
print(df_silver.dtypes)
print(f"\nFinal Silver row count : {len(df_silver):,}")

print("\nData quality summary:")
for col in [
    "product_code",
    "warehouse",
    "product_category",
    "order_date",
    "order_demand",
    "week_start_date",
]:
    print(f"  {col:30s} -> {df_silver[col].isna().sum():,} nulls")

print(f"\n  order_demand min : {df_silver['order_demand'].min():,}")
print(f"  order_demand max : {df_silver['order_demand'].max():,}")
print(
    f"  date range : {df_silver['order_date'].min()} "  # noqa: E501
)

os.makedirs(os.path.dirname(SILVER_PATH), exist_ok=True)
df_silver.to_parquet(SILVER_PATH, index=False)
