from typing import Any
import sys
from flask import Request
import functions_framework
import gcsfs
import joblib
import pandas as pd
from mlops_se489.api.main import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Production inference endpoint for predicting weekly product demand.",
    version="1.0.0"
)

# Enable CORS for frontend UI interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define Model Placeholder with Explicit Typing to satisfy Mypy
model: Any = None
BUCKET_NAME = "mlops489-retail-bucket"
MODEL_PATH = f"gs://{BUCKET_NAME}/models/champion_model.pkl"


def load_model_from_gcs() -> None:
    """Lazily loads the model from GCS on the first request."""
    global model
    if model is None:
        try:
            print(f"Connecting to GCS and downloading model from: {MODEL_PATH}...")
            fs = gcsfs.GCSFileSystem()
            with fs.open(MODEL_PATH, "rb") as f:
                model = joblib.load(f)
            print("Model loaded into memory successfully.")
        except Exception as err:
            print(f"CRITICAL: Failed to load model from GCS: {err}", file=sys.stderr)
            # Fixed Ruff B904 error by chaining exception with 'from err'
            raise RuntimeError(f"Model initialization failed: {str(err)}") from err


# 3. API Routes with Explicit PEP-585 and Return Types
@app.get("/health")
def health_check() -> dict[str, Any]:
    """Verifies API visibility."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: dict[str, list[Any]]) -> dict[str, Any]:
    """Accepts an object of column parallel feature lists.

    Example: {"lag_1_week_demand": [150], "lag_2_week_demand": [120]}
    """
    global model
    # Ensure model is available
    if model is None:
        try:
            load_model_from_gcs()
        except Exception as err:
            raise HTTPException(status_code=500, detail=str(err)) from err

    try:
        # Convert incoming JSON payload straight into a Pandas DataFrame
        df_features = pd.DataFrame(payload)
        
        # Generate predictions
        predictions = model.predict(df_features)
        
        # Return structured response
        return {
            "status": "success",
            "predictions": predictions.tolist()
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(err)}") from err


# 4. Google Cloud Functions Framework Wrapper
@functions_framework.http
def fastapi_server(request: Request) -> Any:
    """GCP HTTP Function handler wrapping the FastAPI app instance."""
    # Removed unused inner imports causing Ruff F401/Pylance missing import loops
    return functions_framework.create_app(app)(request)