# ─────────────────────────────────────────────────────────────────────────────
# Bank-Marketing ML — Makefile
# Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev lint test \
        infra-up infra-down \
        ingest preprocess train evaluate eda \
        pipeline clean

PYTHON := uv run python
UV     := uv

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Bank-Marketing ML — available commands"
	@echo "──────────────────────────────────────"
	@echo "  make install       Install project dependencies with uv"
	@echo "  make install-dev   Install dev dependencies (lint, test)"
	@echo ""
	@echo "  make infra-up      Start PostgreSQL + MLflow via Docker Compose"
	@echo "  make infra-down    Stop and remove containers"
	@echo ""
	@echo "  make ingest        Download dataset from UCI and load into DB"
	@echo "  make preprocess    Clean and engineer features; save to DB"
	@echo "  make eda           Run exploratory data analysis; save plots"
	@echo "  make train         Train all models; log runs to MLflow"
	@echo "  make evaluate      Generate evaluation report"
	@echo "  make pipeline      Run full pipeline (ingest → evaluate)"
	@echo ""
	@echo "  make lint          Run ruff linter"
	@echo "  make test          Run pytest with coverage"
	@echo "  make clean         Remove generated artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(UV) sync --frozen

install-dev:
	$(UV) sync --frozen --extra dev

# ── Infrastructure ────────────────────────────────────────────────────────────
infra-up:
	@echo "Starting PostgreSQL and MLflow..."
	docker compose up -d postgres mlflow
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "PostgreSQL: localhost:5432"
	@echo "MLflow UI:  http://localhost:5000"

infra-down:
	docker compose down

# ── Pipeline steps ────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) -m scripts.run_pipeline --ingest

preprocess:
	$(PYTHON) -m scripts.run_pipeline --preprocess

eda:
	$(PYTHON) -m scripts.run_eda

train:
	$(PYTHON) -m scripts.run_pipeline --train

evaluate:
	$(PYTHON) -m scripts.run_pipeline --evaluate

pipeline:
	$(PYTHON) -m scripts.run_pipeline --all

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	$(UV) run ruff check src/ scripts/ --fix

test:
	$(UV) run pytest

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "Cleaned build artefacts."
