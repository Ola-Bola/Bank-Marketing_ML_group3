# MLflow tracking server image
# Pre-installing dependencies at build time means container startup is instant —
# no pip download on every `docker compose up`.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.10.0 psycopg2-binary==2.9.10

EXPOSE 5000
