"""Data ingestion and cleaning pipeline for retail demand forecasting.

This module handles the Bronze and Silver layers of the medallion architecture.
Bronze ingests raw CSV data as-is. Silver cleans and standardizes it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_FILE = BASE_DIR / "data" / "raw" / "Historical Product Demand.csv"
BRONZE_PATH = BASE_DIR / "data" / "interim" / "bronze_product_demand_raw.parquet"
SILVER_PATH = BASE_DIR / "data" / "interim" / "silver_product_demand_clean.parquet"


def ingest_bronze(raw_file: Path = RAW_FILE, output_path: Path = BRONZE_PATH) -> pd.DataFrame:
    """Read the raw CSV and save it as a Bronze parquet file.

    Bronze rule: land data exactly as-is with no transformation.
    Only audit metadata columns are added.

    Args:
        raw_file: Path to the raw CSV file.
        output_path: Destination path for the Bronze parquet file.

    Returns:
        DataFrame containing the raw Bronze data.
    """
    logger.info("Starting Bronze ingestion from %s", raw_file)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    df = pd.read_csv(raw_file, dtype=str)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    null_counts = df.isnull().sum()
    logger.info("Null counts per column:\n%s", null_counts.to_string())

    df["ingestion_timestamp"] = datetime.utcnow().isoformat()
    df["source_file"] = str(raw_file)
    df["ingestion_date"] = datetime.utcnow().date().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("Bronze file written to %s with %d rows", output_path, len(df))
    return df


def clean_silver(bronze_path: Path = BRONZE_PATH, output_path: Path = SILVER_PATH) -> pd.DataFrame:
    """Clean and standardize Bronze data into a reliable Silver dataset.

    Performs the following transformations:
    - Renames columns to snake_case
    - Parses parenthesized demand values like (500) into -500
    - Parses date strings into proper datetime objects
    - Drops rows with null dates, null demand, or zero demand
    - Adds calendar columns: year, month, week_of_year, week_start_date

    Args:
        bronze_path: Path to the Bronze parquet file.
        output_path: Destination path for the Silver parquet file.

    Returns:
        DataFrame containing the cleaned Silver data.

    Raises:
        FileNotFoundError: If the Bronze parquet file does not exist.
    """
    logger.info("Starting Silver cleaning from %s", bronze_path)

    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze file not found: {bronze_path}")

    df = pd.read_parquet(bronze_path)
    logger.info("Loaded %d rows from Bronze", len(df))

    df = df.rename(
        columns={
            "Product_Code": "product_code",
            "Warehouse": "warehouse",
            "Product_Category": "product_category",
            "Date": "order_date_raw",
            "Order_Demand": "order_demand_raw",
        }
    )
    df = df.drop(columns=["source_file", "ingestion_date", "ingestion_timestamp"], errors="ignore")

    df["order_demand_str"] = df["order_demand_raw"].str.strip()
    mask = df["order_demand_str"].str.startswith("(", na=False)
    df.loc[mask, "order_demand_str"] = "-" + df.loc[mask, "order_demand_str"].str.replace(r"[()]", "", regex=True)
    df["order_demand"] = pd.to_numeric(df["order_demand_str"], errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date_raw"], errors="coerce")

    total = len(df)
    failed_dates = df["order_date"].isna().sum()
    logger.info("Date parse success rate: %.2f%%", (total - failed_dates) / total * 100)

    df = df.dropna(subset=["order_date", "order_demand", "product_code", "warehouse"])
    df = df[df["order_demand"] != 0]
    logger.info("Rows after cleaning: %d (dropped %d)", len(df), total - len(df))

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["week_start_date"] = df["order_date"] - pd.to_timedelta(df["order_date"].dt.weekday, unit="d")
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

    for col in ["product_code", "warehouse", "order_date", "order_demand"]:
        null_count = df_silver[col].isna().sum()
        if null_count > 0:
            logger.warning("Null values found in %s: %d", col, null_count)
        else:
            logger.info("Column %s: no nulls", col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_silver.to_parquet(output_path, index=False)
    logger.info("Silver file written to %s with %d rows", output_path, len(df_silver))
    return df_silver


def process_data(
    raw_file: Path = RAW_FILE,
    bronze_path: Path = BRONZE_PATH,
    silver_path: Path = SILVER_PATH,
) -> pd.DataFrame:
    """Run the full Bronze and Silver processing pipeline."""
    ingest_bronze(raw_file=raw_file, output_path=bronze_path)
    return clean_silver(bronze_path=bronze_path, output_path=silver_path)


if __name__ == "__main__":
    process_data()
