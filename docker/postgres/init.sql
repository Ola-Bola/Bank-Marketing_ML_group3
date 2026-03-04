-- Initialise databases for the bank-marketing ML project.
-- This script runs automatically on first container startup.

-- Create MLflow backend database and user
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- The main app database (bank_marketing) is created by the POSTGRES_DB env var.
-- Grant the postgres superuser full access (already default).
