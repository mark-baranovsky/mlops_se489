import os

import requests
import streamlit as st

BASE_API_URL = os.getenv(
    "API_URL",
    "https://retail-demand-api-699949078927.us-central1.run.app",
)

st.set_page_config(
    page_title="Product Demand Predictor",
    layout="centered",
)

st.title("📦 Product Demand Predictor")
st.write(
    "This Streamlit app sends product demand features to our deployed "
    "FastAPI backend on Google Cloud Run."
)

with st.sidebar:
    st.header("API Settings")
    base_api_url = st.text_input("Cloud Run API Base URL", value=BASE_API_URL)
    predict_url = f"{base_api_url.rstrip('/')}/predict"
    health_url = f"{base_api_url.rstrip('/')}/health"

    st.caption("This app calls the deployed FastAPI service on GCP Cloud Run.")

    if st.button("Check API Health"):
        try:
            health_response = requests.get(health_url, timeout=20)
            if health_response.ok:
                st.success("API is healthy.")
                st.json(health_response.json())
            else:
                st.error(f"Health check failed: {health_response.status_code}")
                st.text(health_response.text)
        except requests.exceptions.RequestException as exc:
            st.error("Could not connect to the API health endpoint.")
            st.exception(exc)

st.subheader("Input Features")

lag_1_week_demand = st.number_input("Lag 1 Week Demand", value=1000.0)
lag_2_week_demand = st.number_input("Lag 2 Week Demand", value=950.0)
lag_3_week_demand = st.number_input("Lag 3 Week Demand", value=900.0)
lag_4_week_demand = st.number_input("Lag 4 Week Demand", value=875.0)
lag_52_week_demand = st.number_input("Lag 52 Week Demand", value=1000.0)

rolling_4wk_avg_demand = st.number_input("Rolling 4 Week Avg Demand", value=856.0)
rolling_4wk_std_demand = st.number_input("Rolling 4 Week Std Demand", value=50.0)
rolling_8wk_avg_demand = st.number_input("Rolling 8 Week Avg Demand", value=920.0)

demand_momentum = st.number_input("Demand Momentum", value=25.0)
yoy_demand_change = st.number_input("Year-over-Year Demand Change", value=10.0)
yoy_demand_pct_change = st.number_input("Year-over-Year Demand Percent Change", value=0.05)

year = st.number_input("Year", value=2017, step=1)
month = st.number_input("Month", value=1, min_value=1, max_value=12, step=1)
quarter = st.number_input("Quarter", value=1, min_value=1, max_value=4, step=1)
week_of_year = st.number_input("Week of Year", value=2, min_value=1, max_value=53, step=1)

is_quarter_end_week = st.selectbox("Is Quarter End Week?", [0, 1])
is_peak_month = st.selectbox("Is Peak Month?", [0, 1])
order_count = st.number_input("Order Count", value=10, min_value=0, step=1)

payload = {
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
    "year": int(year),
    "month": int(month),
    "quarter": int(quarter),
    "week_of_year": int(week_of_year),
    "is_quarter_end_week": int(is_quarter_end_week),
    "is_peak_month": int(is_peak_month),
    "order_count": int(order_count),
}

st.subheader("Request Payload")
with st.expander("View JSON sent to API"):
    st.json(payload)

if st.button("Predict Weekly Demand"):
    try:
        response = requests.post(predict_url, json=payload, timeout=30)

        if response.ok:
            result = response.json()
            st.success("Prediction successful!")

            if "predicted_weekly_demand" in result:
                st.metric(
                    "Predicted Weekly Demand",
                    f"{result['predicted_weekly_demand']:,.2f}",
                )
            else:
                st.warning("Prediction key not found. Showing raw API response.")
                st.json(result)

            with st.expander("View Raw API Response"):
                st.json(result)
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.text(response.text)

    except requests.exceptions.RequestException as exc:
        st.error("Could not connect to the prediction API.")
        st.write("Make sure the Cloud Run FastAPI service is deployed and public.")
        st.exception(exc)