#!/bin/sh
# Start mlflow server with sqlite backend and local artifact store
mkdir -p /mlflow/artifacts
echo "Starting MLflow server..."
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --default-artifact-root file:///mlflow/artifacts
