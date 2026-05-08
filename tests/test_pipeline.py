import numpy as np
import pandas as pd


def test_bronze_columns_preserved() -> None:
    raw = pd.DataFrame(
        {
            "Product_Code": ["Product_001"],
            "Warehouse": ["Whse_A"],
            "Product_Category": ["Category_001"],
            "Date": ["2012/1/1"],
            "Order_Demand": ["100"],
        }
    )

    bronze = raw.copy()
    bronze["ingestion_timestamp"] = "2026-01-01T00:00:00"
    bronze["source_file"] = "Historical Product Demand.csv"
    bronze["ingestion_date"] = "2026-01-01"

    for col in raw.columns:
        assert col in bronze.columns


def test_bronze_all_strings() -> None:
    raw = pd.DataFrame(
        {
            "Product_Code": ["Product_001"],
            "Warehouse": ["Whse_A"],
            "Product_Category": ["Category_001"],
            "Date": ["2012/1/1"],
            "Order_Demand": ["100"],
        },
        dtype=str,
    )

    assert all(dtype == "object" for dtype in raw.dtypes)


def test_silver_demand_parsing() -> None:
    df = pd.DataFrame({"order_demand_raw": ["100", "(25)", "0"]})

    df["order_demand_str"] = df["order_demand_raw"].str.strip()
    mask = df["order_demand_str"].str.startswith("(", na=False)
    df.loc[mask, "order_demand_str"] = "-" + df.loc[mask, "order_demand_str"].str.replace(r"[()]", "", regex=True)
    df["order_demand"] = pd.to_numeric(df["order_demand_str"], errors="coerce")

    assert df.loc[0, "order_demand"] == 100
    assert df.loc[1, "order_demand"] == -25
    assert df.loc[2, "order_demand"] == 0


def test_silver_no_nulls_after_cleaning() -> None:
    df = pd.DataFrame(
        {
            "product_code": ["Product_001", None],
            "warehouse": ["Whse_A", "Whse_B"],
            "product_category": ["Category_001", "Category_002"],
            "order_date": pd.to_datetime(["2012-01-01", None]),
            "order_demand": [100, 50],
        }
    )

    clean = df.dropna(subset=["order_date", "order_demand", "product_code", "warehouse"])

    assert clean["product_code"].isna().sum() == 0
    assert clean["warehouse"].isna().sum() == 0
    assert clean["order_date"].isna().sum() == 0
    assert clean["order_demand"].isna().sum() == 0


def test_silver_no_zero_demand() -> None:
    df = pd.DataFrame({"order_demand": [100, 0, -20, 50]})

    clean = df[df["order_demand"] != 0]

    assert (clean["order_demand"] == 0).sum() == 0


def test_silver_date_parsing() -> None:
    df = pd.DataFrame({"order_date_raw": ["2012/1/1", "2013/05/10"]})

    df["order_date"] = pd.to_datetime(df["order_date_raw"], errors="coerce")

    assert df["order_date"].isna().sum() == 0
    assert str(df.loc[0, "order_date"].date()) == "2012-01-01"


def test_gold_weekly_aggregation() -> None:
    df = pd.DataFrame(
        {
            "warehouse": ["Whse_A", "Whse_A"],
            "product_code": ["Product_001", "Product_001"],
            "product_category": ["Category_001", "Category_001"],
            "week_start_date": pd.to_datetime(["2012-01-02", "2012-01-02"]),
            "year": [2012, 2012],
            "month": [1, 1],
            "week_of_year": [1, 1],
            "order_demand": [100, 50],
        }
    )

    group_cols = [
        "warehouse",
        "product_code",
        "product_category",
        "week_start_date",
        "year",
        "month",
        "week_of_year",
    ]

    gold = df.groupby(group_cols)["order_demand"].agg(weekly_order_demand="sum", order_count="count").reset_index()

    assert len(gold) == 1
    assert gold.loc[0, "weekly_order_demand"] == 150
    assert gold.loc[0, "order_count"] == 2


def test_gold_no_duplicate_keys() -> None:
    df_gold = pd.DataFrame(
        {
            "warehouse": ["Whse_A", "Whse_A"],
            "product_code": ["Product_001", "Product_002"],
            "week_start_date": pd.to_datetime(["2012-01-02", "2012-01-02"]),
        }
    )

    total = len(df_gold)
    distinct_keys = df_gold[["warehouse", "product_code", "week_start_date"]].drop_duplicates().shape[0]

    assert total == distinct_keys


def test_lag_features_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "warehouse": ["Whse_A"] * 5,
            "product_code": ["Product_001"] * 5,
            "week_start_date": pd.date_range("2012-01-02", periods=5, freq="W-MON"),
            "weekly_order_demand": [10, 20, 30, 40, 50],
        }
    )

    df["lag_1_week_demand"] = df.groupby(["warehouse", "product_code"])["weekly_order_demand"].shift(1)

    assert pd.isna(df.loc[0, "lag_1_week_demand"])
    assert df.loc[1, "lag_1_week_demand"] == 10
    assert df.loc[2, "lag_1_week_demand"] == 20
    assert df.loc[2, "lag_1_week_demand"] != df.loc[2, "weekly_order_demand"]


def test_predictions_non_negative() -> None:
    raw_preds = np.array([100.4, -20.7, 0.0, 55.8])

    final_preds = np.maximum(np.round(raw_preds).astype(int), 0)

    assert (final_preds >= 0).all()


def test_time_based_split() -> None:
    split_date = "2016-10-01"

    df = pd.DataFrame(
        {
            "week_start_date": pd.to_datetime(["2016-09-19", "2016-09-26", "2016-10-03", "2016-10-10"]),
            "weekly_order_demand": [100, 120, 130, 140],
        }
    )

    train = df[df["week_start_date"] < split_date]
    val = df[df["week_start_date"] >= split_date]

    assert len(train) == 2
    assert len(val) == 2
    assert train["week_start_date"].max() < pd.to_datetime(split_date)
    assert val["week_start_date"].min() >= pd.to_datetime(split_date)
