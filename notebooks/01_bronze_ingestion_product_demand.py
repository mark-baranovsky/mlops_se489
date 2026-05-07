import os
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "Historical Product Demand.csv")
BRONZE_PATH = os.path.join(
    BASE_DIR, "data", "interim", "bronze_product_demand_raw.parquet"
)

print(f"Source file  : {RAW_FILE_PATH}")
print(f"Target file  : {BRONZE_PATH}")

df_raw = pd.read_csv(RAW_FILE_PATH, dtype=str)

print(f"\nRows loaded  : {len(df_raw):,}")
print(f"Columns      : {list(df_raw.columns)}")
print(f"\nDtypes:\n{df_raw.dtypes}")
print(f"\nNull counts:\n{df_raw.isnull().sum()}")
print(f"\nSample rows:\n{df_raw.head(10).to_string()}")

df_bronze = df_raw.copy()
df_bronze["ingestion_timestamp"] = datetime.utcnow().isoformat()
df_bronze["source_file"] = RAW_FILE_PATH
df_bronze["ingestion_date"] = datetime.utcnow().date().isoformat()

os.makedirs(os.path.dirname(BRONZE_PATH), exist_ok=True)
df_bronze.to_parquet(BRONZE_PATH, index=False)

print(f"\nBronze file written : {BRONZE_PATH}")
print(f"Rows                : {len(df_bronze):,}")
print(f"Columns             : {list(df_bronze.columns)}")
