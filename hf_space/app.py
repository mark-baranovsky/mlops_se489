from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.pkl"

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


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Champion model file was not found. "
            f"Expected model path: {MODEL_PATH}"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


def build_input_dataframe():
    """Build one prediction row from Streamlit sidebar inputs."""

    st.sidebar.header("Demand lag features")

    lag_1_week_demand = st.sidebar.number_input(
        "Lag 1 week demand",
        value=1000.0,
    )
    lag_2_week_demand = st.sidebar.number_input(
        "Lag 2 week demand",
        value=950.0,
    )
    lag_3_week_demand = st.sidebar.number_input(
        "Lag 3 week demand",
        value=900.0,
    )
    lag_4_week_demand = st.sidebar.number_input(
        "Lag 4 week demand",
        value=875.0,
    )
    lag_52_week_demand = st.sidebar.number_input(
        "Lag 52 week demand",
        value=1000.0,
    )

    st.sidebar.header("Rolling demand features")

    rolling_4wk_avg_demand = st.sidebar.number_input(
        "Rolling 4 week average demand",
        value=930.0,
    )
    rolling_4wk_std_demand = st.sidebar.number_input(
        "Rolling 4 week standard deviation",
        value=50.0,
    )
    rolling_8wk_avg_demand = st.sidebar.number_input(
        "Rolling 8 week average demand",
        value=920.0,
    )
    demand_momentum = st.sidebar.number_input(
        "Demand momentum",
        value=25.0,
    )

    st.sidebar.header("Year-over-year features")

    yoy_demand_change = st.sidebar.number_input(
        "YoY demand change",
        value=100.0,
    )
    yoy_demand_pct_change = st.sidebar.number_input(
        "YoY demand percent change",
        value=0.10,
    )

    st.sidebar.header("Calendar and order features")

    year = st.sidebar.number_input(
        "Year",
        value=2017,
        step=1,
    )
    month = st.sidebar.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )
    quarter = st.sidebar.number_input(
        "Quarter",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
    )
    week_of_year = st.sidebar.number_input(
        "Week of year",
        min_value=1,
        max_value=53,
        value=25,
        step=1,
    )

    is_quarter_end_week = int(st.sidebar.checkbox("Is quarter end week?"))
    is_peak_month = int(st.sidebar.checkbox("Is peak month?"))
    order_count = st.sidebar.number_input(
        "Order count",
        value=50.0,
    )

    row = {
        "lag_1_week_demand": lag_1_week_demand,
        "lag_2_week_demand": lag_2_week_demand,
        "lag_3_week_demand": lag_3_week_demand,
        "lag_4_week_demand": lag_4_week_demand,
        "lag_52_week_demand": lag_52_week_demand,
        "rolling_4wk_avg_demand": rolling_4wk_avg_demand,
        "rolling_4wk_std_demand": rolling_4wk_std_demand,
        "rolling_8wk_avg_demand": rolling_8wk_avg_demand,
        "demand_momentum": demand_momentum,
        "yoy_demand_change": yoy_demand_change,
        "yoy_demand_pct_change": yoy_demand_pct_change,
        "year": year,
        "month": month,
        "quarter": quarter,
        "week_of_year": week_of_year,
        "is_quarter_end_week": is_quarter_end_week,
        "is_peak_month": is_peak_month,
        "order_count": order_count,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def main():
    st.set_page_config(
        page_title="Retail Demand Forecasting",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Retail Demand Forecasting")
    st.write(
        "This Streamlit app predicts weekly order demand using the trained "
        "champion model from the retail demand forecasting pipeline."
    )

    model = load_model()
    input_df = build_input_dataframe()

    st.subheader("Input Features")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Weekly Demand"):
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted weekly order demand: {prediction:,.2f}")

    st.caption(
        "Model expected at `hf_space/models/champion_model.pkl`. "
        "The GitHub Actions deployment workflow copies the trained model there."
    )


if __name__ == "__main__":
    main()
