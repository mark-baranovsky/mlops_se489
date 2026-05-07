import os
import sys

import mlflow
import pandas as pd
import sklearn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_INTERIM = os.path.join(BASE_DIR, "data", "interim")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

BRONZE_PATH = os.path.join(DATA_INTERIM, "bronze_product_demand_raw.parquet")
SILVER_PATH = os.path.join(DATA_INTERIM, "silver_product_demand_clean.parquet")
GOLD_PATH = os.path.join(DATA_PROCESSED, "gold_weekly_product_demand.parquet")
FEATURES_PATH = os.path.join(
    DATA_PROCESSED, "gold_weekly_product_demand_features.parquet"
)
PRED_PATH = os.path.join(DATA_PROCESSED, "demand_predictions.parquet")

print(f"Python     : {sys.version}")
print(f"Pandas     : {pd.__version__}")
print(f"Sklearn    : {sklearn.__version__}")
print(f"MLflow     : {mlflow.__version__}")

print("\nDirectories created:")
print(f"  Raw       -> {DATA_RAW}")
print(f"  Interim   -> {DATA_INTERIM}")
print(f"  Processed -> {DATA_PROCESSED}")
print(f"  Models    -> {MODELS_DIR}")

print("\nData paths:")
print(f"  Bronze   -> {BRONZE_PATH}")
print(f"  Silver   -> {SILVER_PATH}")
print(f"  Gold     -> {GOLD_PATH}")
print(f"  Features -> {FEATURES_PATH}")
print(f"  Preds    -> {PRED_PATH}")
