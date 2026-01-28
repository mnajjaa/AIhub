FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --upgrade pip && pip install mlflow==2.11.0

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
