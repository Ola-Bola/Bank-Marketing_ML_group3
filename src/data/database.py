"""
PostgreSQL database client.

Wraps SQLAlchemy to provide a simple interface for:
  - Schema initialisation
  - Bulk insert of DataFrames
  - Query helpers
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "data" / "sql" / "schema.sql"


class DatabaseClient:
    """Thin wrapper around a SQLAlchemy engine for this project's DB."""

    def __init__(self, db_url: str | None = None) -> None:
        url = db_url or settings.db_url
        self._engine: Engine = create_engine(url, pool_pre_ping=True)
        logger.debug("DatabaseClient connected to %s", url.split("@")[-1])

    # ── Schema ────────────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Create tables from schema.sql if they do not already exist."""
        if not SCHEMA_FILE.exists():
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")
        sql = SCHEMA_FILE.read_text()
        with self._engine.begin() as conn:
            conn.execute(text(sql))
        logger.info("Database schema ensured.")

    # ── Write ─────────────────────────────────────────────────────────────────

    def bulk_insert(self, table: str, df: pd.DataFrame, if_exists: str = "append") -> int:
        """
        Insert a DataFrame into *table*.

        Parameters
        ----------
        table     : target table name
        df        : DataFrame whose columns match the table columns
        if_exists : passed to pandas; default 'append'

        Returns the number of rows inserted.
        """
        df.to_sql(
            table,
            con=self._engine,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=1000,
        )
        return len(df)

    def execute(self, sql: str, params: dict | None = None) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(sql), params or {})

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def table_to_df(self, table: str) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {table}")  # noqa: S608

    def save_experiment_result(self, result: dict) -> None:
        """Upsert an experiment result row (keyed by mlflow_run_id)."""
        sql = """
            INSERT INTO experiment_results
                (mlflow_run_id, model_name, roc_auc, f1_score,
                 precision_score, recall_score, accuracy, params_json)
            VALUES
                (:mlflow_run_id, :model_name, :roc_auc, :f1_score,
                 :precision_score, :recall_score, :accuracy,
                 CAST(:params_json AS jsonb))
            ON CONFLICT (mlflow_run_id)
            DO UPDATE SET
                roc_auc         = EXCLUDED.roc_auc,
                f1_score        = EXCLUDED.f1_score,
                precision_score = EXCLUDED.precision_score,
                recall_score    = EXCLUDED.recall_score,
                accuracy        = EXCLUDED.accuracy,
                params_json     = EXCLUDED.params_json
        """
        self.execute(sql, result)
        logger.info("Saved experiment result for run %s", result.get("mlflow_run_id"))
