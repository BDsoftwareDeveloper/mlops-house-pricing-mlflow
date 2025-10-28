import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI(title="House Price Predictor (demo)")

# Only feature columns (exclude target)
class PredictRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

model_artifact = None
feature_columns = []

def load_model(path=MODEL_PATH):
    global model_artifact, feature_columns
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Has the trainer run?")
    model_artifact = joblib.load(path)
    # Exclude target column automatically
    feature_columns = [c for c in model_artifact.get("columns") if c != "MedHouseVal"]
    print("Model loaded. Feature columns:", feature_columns)

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print("Warning on startup:", e)

@app.post("/predict")
async def predict(req: PredictRequest):
    if model_artifact is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

    model = model_artifact.get("model")
    # Convert request to ordered feature array
    features = [getattr(req, col) for col in feature_columns]
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": float(pred)}
