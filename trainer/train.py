# Trainer logs experiment and model to MLflow tracking server, registers model, and saves local artifact.
import os
import time
import mlflow
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

MODEL_DIR = "/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "house_price_model"

os.makedirs(MODEL_DIR, exist_ok=True)

def wait_for_mlflow(uri, timeout=30):
    import requests
    start = time.time()
    while True:
        try:
            resp = requests.get(uri)
            # if reachable, break
            print("MLflow reachable:", resp.status_code)
            break
        except Exception as e:
            if time.time() - start > timeout:
                print("Timeout waiting for MLflow, continuing anyway:", e)
                break
            print("Waiting for MLflow to be ready...")
            time.sleep(2)

print("Setting MLflow tracking URI ->", MLFLOW_URI)
mlflow.set_tracking_uri(MLFLOW_URI)

# wait for mlflow server to be ready (best-effort)
try:
    wait_for_mlflow(MLFLOW_URI)
except Exception:
    pass

print("Loading California housing dataset...")
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 50
print("Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = mean_squared_error(y_val, preds) ** 0.5
print(f"Validation RMSE: {rmse:.4f}")

# Start MLflow run and log params/metrics/model
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print("MLflow run id:", run_id)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", float(rmse))

    # log sklearn model artifact (this also saves artifacts to mlflow server)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Register the model in the MLflow Model Registry (best-effort; may fail if server not configured)
    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        print("Registered model:", mv.name, mv.version)
    except Exception as e:
        print("Model registration failed (continuing):", e)

# Also save a local joblib to shared volume so API can load it
print(f"Saving model locally to {MODEL_PATH}")
joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)

print("Trainer finished.")
