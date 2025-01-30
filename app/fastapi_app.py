from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
import os

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Define input data structure
class PredictionInput(BaseModel):
    data: list

# 🔥 Define model paths
PROJECT_DIR = "F:/Portfolio Projects/fault_prediction_project"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_rf_model.pkl")  # If RF is best
BEST_MODEL_NAME = "best_rf_model"  # MLflow model name

# ✅ Load the best model
print("🔄 Loading best model for deployment...")

try:
    if os.path.exists(BEST_MODEL_PATH):
        print("🌲 Loading Random Forest model from local file...")
        model = joblib.load(BEST_MODEL_PATH)
        model_type = "random_forest"
    else:
        print(f"🔄 Loading best model from MLflow: {BEST_MODEL_NAME}")
        model = mlflow.sklearn.load_model(f"models:/{BEST_MODEL_NAME}/latest")
        model_type = "mlflow"
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_type = None


@app.post("/predict")
def predict(input_data: PredictionInput):
    """Handle prediction requests."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        # ✅ Ensure correct input format
        data = np.array(input_data.data, dtype=np.float32)

        # ✅ Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # ✅ Make predictions
        predictions = model.predict(data)

        # ✅ If Random Forest, return class labels
        if model_type == "random_forest":
            predictions = predictions.tolist()
        
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
