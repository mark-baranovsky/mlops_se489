import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8888/predict")

st.set_page_config(
    page_title="Product Demand Predictor",
    layout="centered",
)

st.title("📦 Product Demand Predictor")
st.write(
    "This Streamlit app sends product demand features to your deployed "
    "FastAPI or Cloud Function prediction endpoint."
)

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("Prediction API URL", value=API_URL)
    st.caption("Example: http://localhost:8888/predict or your Cloud Function URL")

st.subheader("Input Features")

lag_1_week_demand = st.number_input("Lag 1 Week Demand", value=100.0)
lag_2_week_demand = st.number_input("Lag 2 Week Demand", value=95.0)
lag_3_week_demand = st.number_input("Lag 3 Week Demand", value=90.0)
lag_4_week_demand = st.number_input("Lag 4 Week Demand", value=85.0)
lag_52_week_demand = st.number_input("Lag 52 Week Demand", value=100.0)

rolling_4wk_avg_demand = st.number_input("Rolling 4 Week Avg Demand", value=92.5)
rolling_4wk_std_demand = st.number_input("Rolling 4 Week Std Demand", value=10.0)
rolling_8wk_avg_demand = st.number_input("Rolling 8 Week Avg Demand", value=90.0)

demand_momentum = st.number_input("Demand Momentum", value=5.0)
yoy_demand_change = st.number_input("Year-over-Year Demand Change", value=0.0)
yoy_demand_pct_change = st.number_input("Year-over-Year Demand Percent Change", value=0.0)

year = st.number_input("Year", value=2016, step=1)
month = st.number_input("Month", value=10, min_value=1, max_value=12, step=1)
quarter = st.number_input("Quarter", value=4, min_value=1, max_value=4, step=1)
week_of_year = st.number_input("Week of Year", value=40, min_value=1, max_value=53, step=1)

is_quarter_end_week = st.selectbox("Is Quarter End Week?", [0, 1])
is_peak_month = st.selectbox("Is Peak Month?", [0, 1])
order_count = st.number_input("Order Count", value=10.0)

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
    "order_count": order_count,
}

st.subheader("Request Payload")
with st.expander("View JSON sent to API"):
    st.json(payload)

if st.button("Predict Demand"):
    try:
        response = requests.post(api_url, json=payload, timeout=30)

        if response.ok:
            result = response.json()
            st.success("Prediction successful!")

            if "predicted_demand" in result:
                st.metric("Predicted Demand", result["predicted_demand"])
            else:
                st.json(result)
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.text(response.text)

    except requests.exceptions.RequestException as e:
        st.error("Could not connect to the prediction API.")
        st.exception(e)
