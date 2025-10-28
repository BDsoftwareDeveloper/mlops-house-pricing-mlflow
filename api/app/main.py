import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model.joblib")

app = FastAPI(title="House Price Predictor (demo)")

class PredictRequest(BaseModel):
    features: list

model_artifact = None

def load_model(path=MODEL_PATH):
    global model_artifact
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Has the trainer run?")
    model_artifact = joblib.load(path)
    print("Model loaded. Columns:", model_artifact.get("columns"))

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
    cols = model_artifact.get("columns")
    model = model_artifact.get("model")
    features = req.features
    if len(features) != len(cols):
        raise HTTPException(status_code=400, detail=f"Expected {len(cols)} features in order {cols}")
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": float(pred)}
