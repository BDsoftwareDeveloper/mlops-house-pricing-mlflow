FROM python:3.10-slim
WORKDIR /app
COPY train.py /app/train.py
RUN pip install --no-cache-dir scikit-learn pandas joblib mlflow==2.4.0
RUN mkdir -p /app/models
CMD ["python", "train.py"]
