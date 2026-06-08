"""FastAPI service for retail demand forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/champion_model.pkl")

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


class DemandRequest(BaseModel):
    """Input features for one retail demand prediction."""

    lag_1_week_demand: float = Field(1000.0)
    lag_2_week_demand: float = Field(950.0)
    lag_3_week_demand: float = Field(900.0)
    lag_4_week_demand: float = Field(875.0)
    lag_52_week_demand: float = Field(1000.0)
    rolling_4wk_avg_demand: float = Field(856.0)
    rolling_4wk_std_demand: float = Field(50.0)
    rolling_8wk_avg_demand: float = Field(920.0)
    demand_momentum: float = Field(25.0)
    yoy_demand_change: float = Field(10.0)
    yoy_demand_pct_change: float = Field(0.05)
    year: int = Field(2017)
    month: int = Field(1)
    quarter: int = Field(1)
    week_of_year: int = Field(2)
    is_quarter_end_week: int = Field(0)
    is_peak_month: int = Field(0)
    order_count: int = Field(10)


app = FastAPI(title="Retail Demand Forecasting API", version="1.0.0")


def load_model() -> Any:
    """Load the trained champion model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "retail-demand-forecasting-api",
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(request: DemandRequest) -> dict[str, Any]:
    """Predict weekly retail demand."""
    model = load_model()

    row = {col: getattr(request, col) for col in FEATURE_COLS}
    input_df = pd.DataFrame([row], columns=FEATURE_COLS)

    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_weekly_demand": round(prediction, 2),
        "model_path": str(MODEL_PATH),
        "input_features": row,
    }
