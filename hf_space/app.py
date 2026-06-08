import os

import pandas as pd
import requests
import streamlit as st


API_URL = "https://retail-demand-api-699949078927.us-central1.run.app"

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


st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Retail Demand Forecasting")

st.write("This app sends feature values to the deployed FastAPI model endpoint.")

st.sidebar.header("Input features")

inputs = {
    "lag_1_week_demand": st.sidebar.number_input("Lag 1 week demand", value=1000.0),
    "lag_2_week_demand": st.sidebar.number_input("Lag 2 week demand", value=950.0),
    "lag_3_week_demand": st.sidebar.number_input("Lag 3 week demand", value=900.0),
    "lag_4_week_demand": st.sidebar.number_input("Lag 4 week demand", value=875.0),
    "lag_52_week_demand": st.sidebar.number_input("Lag 52 week demand", value=1000.0),
    "rolling_4wk_avg_demand": st.sidebar.number_input(
        "Rolling 4 week average demand",
        value=930.0,
    ),
    "rolling_4wk_std_demand": st.sidebar.number_input(
        "Rolling 4 week standard deviation",
        value=50.0,
    ),
    "rolling_8wk_avg_demand": st.sidebar.number_input(
        "Rolling 8 week average demand",
        value=920.0,
    ),
    "demand_momentum": st.sidebar.number_input("Demand momentum", value=25.0),
    "yoy_demand_change": st.sidebar.number_input("YoY demand change", value=100.0),
    "yoy_demand_pct_change": st.sidebar.number_input(
        "YoY demand percent change",
        value=0.10,
    ),
    "year": st.sidebar.number_input("Year", value=2017, step=1),
    "month": st.sidebar.number_input("Month", min_value=1, max_value=12, value=6),
    "quarter": st.sidebar.number_input("Quarter", min_value=1, max_value=4, value=2),
    "week_of_year": st.sidebar.number_input(
        "Week of year",
        min_value=1,
        max_value=53,
        value=25,
    ),
    "is_quarter_end_week": int(st.sidebar.checkbox("Is quarter end week?")),
    "is_peak_month": int(st.sidebar.checkbox("Is peak month?")),
    "order_count": st.sidebar.number_input("Order count", value=50.0),
}

input_df = pd.DataFrame([inputs], columns=FEATURE_COLS)

st.subheader("Input Preview")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Weekly Demand"):
    if not API_URL:
        st.error("Missing API_URL. Add it in Hugging Face Space settings.")
        st.stop()

    try:
        response = requests.post(API_URL, json=inputs, timeout=30)
        response.raise_for_status()

        result = response.json()
        prediction = result["prediction"]

        st.success(f"Predicted weekly order demand: {prediction:,.2f}")

    except requests.exceptions.RequestException as error:
        st.error("Could not connect to the FastAPI prediction service.")
        st.code(str(error))

    except KeyError:
        st.error("The API response did not include a `prediction` field.")
        st.write("API response:")
        st.json(response.json())