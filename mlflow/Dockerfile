FROM python:3.10-slim
WORKDIR /app

# Install curl for healthcheck + mlflow dependencies
RUN apt-get update && apt-get install -y curl && apt-get clean

# Install MLflow and DB dependencies
RUN pip install --no-cache-dir \
    mlflow==2.4.0 \
    sqlalchemy \
    alembic \
    psycopg2-binary

# Create MLflow artifact and backend store directory
RUN mkdir -p /mlflow/artifacts /mlflow/db
ENV BACKEND_STORE_URI=sqlite:////mlflow/db/mlflow.db
ENV ARTIFACT_ROOT=/mlflow/artifacts

COPY start-mlflow.sh /app/start-mlflow.sh
RUN chmod +x /app/start-mlflow.sh

EXPOSE 5000

CMD ["/app/start-mlflow.sh"]
