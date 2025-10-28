# mlops-house-pricing (MLflow-enabled demo)

This is a minimal, runnable MLOps-style demo you can run locally with Docker Compose.
It trains a simple scikit-learn model (California housing) in a `trainer` container, logs
the experiment & model to a local MLflow Tracking Server, saves a local model artifact
into a shared Docker volume, and serves predictions from a FastAPI `api` service.

## Services
- `mlflow` : MLflow tracking server (sqlite backend, file artifact store) — UI at http://localhost:5000
- `trainer` : trains model, logs to MLflow, registers model, writes model.joblib to shared volume
- `api` : FastAPI server that loads model from shared volume and serves /predict

## Quickstart (from project root)
1. Build and run:
   ```bash
   docker compose up --build
   ```
   The mlflow server will start, then the trainer will run (logs experiment to MLflow and saves artifact),
   then the API will start and be available at port 8000.

2. Open MLflow UI:
   - http://localhost:5000

3. Test prediction (example using curl):
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [8, 41, 6, 2, 5, 1, 3000, 2]}'
   ```

4. Add GitHub secrets for CI/CD
In your GitHub repo → Settings → Secrets → Actions:

| Secret               | Value                                 |
| -------------------- | ------------------------------------- |
| `DOCKERHUB_USERNAME` | Your Docker Hub username              |
| `DOCKERHUB_TOKEN`    | Your Docker Hub personal access token |



5. Test API:
```bash

   curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [0.038, 0.02, 5.0, 1.0, 0.02, 3.0, 37.0, -122.0]}'
```

Expected output:

```json
{"prediction": 2.5}
```



📊 MLflow Tracking

- View experiment runs in MLflow UI

- Track parameters, metrics, and artifacts

- Model registration available via UI

✅ CI/CD

- GitHub Actions pipeline automatically:

- Builds Docker containers (trainer + api)

- Runs training + logs metrics to MLflow

- Runs basic tests (tests/test_api.py)

- Pushes Docker images to Docker Hub





📦 Project Structure

├── .github/workflows/mlops-ci.yml
├── trainer/
│   ├── Dockerfile
│   └── train.py
├── api/
│   ├── Dockerfile
│   └── app/main.py
├── mlflow/
│   ├── Dockerfile
│   └── start-mlflow.sh
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── README.md
└── .gitignore




Notes:
- This demo uses an sqlite backend for MLflow (file mlflow/mlflow.db inside the mlflow_store volume).
- The trainer waits briefly for mlflow service to be available (simple retry).
- You can extend by adding authentication, remote artifact store, or a real database backend for MLflow.