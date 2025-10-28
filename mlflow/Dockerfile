FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.4.0 sqlalchemy alembic
# create artifact dir
RUN mkdir -p /mlflow/artifacts
COPY start-mlflow.sh /app/start-mlflow.sh
RUN chmod +x /app/start-mlflow.sh
CMD ["/app/start-mlflow.sh"]
