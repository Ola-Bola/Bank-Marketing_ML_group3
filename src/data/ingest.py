"""
Data ingestion module.

Downloads the Bank Marketing dataset from the UCI ML Repository, saves it to
disk (data/raw/), and optionally loads it into PostgreSQL (raw_features table).

Dataset (ucimlrepo id=222) — bank-full version:
  45 211 rows × 16 features + 1 target column ('y' → renamed 'subscribed').

Actual feature columns returned by the API
------------------------------------------
  age, job, marital, education, default, balance, housing, loan,
  contact, day_of_week (int, 1-31), month, duration,
  campaign, pdays, previous, poutcome
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import settings
from src.data.database import DatabaseClient

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

UCI_DATASET_ID = 222  # Bank Marketing dataset

# The ucimlrepo library returns clean snake_case column names for this dataset
# (no dot-notation). We only rename the target column from 'y' → 'subscribed'.
COLUMN_RENAME: dict[str, str] = {}   # no renaming needed for features


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_raw_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Download the Bank Marketing dataset from the UCI ML Repository.

    Returns
    -------
    X : pd.DataFrame — feature matrix (45 211 rows × 16 columns)
    y : pd.Series   — target series ('y' column: 'yes' / 'no')
    """
    logger.info("Fetching Bank Marketing dataset (id=%d) from UCI...", UCI_DATASET_ID)
    bank_marketing = fetch_ucirepo(id=UCI_DATASET_ID)

    X: pd.DataFrame = bank_marketing.data.features.copy()
    # targets is a single-column DataFrame; extract as a Series to avoid 2D issues
    y: pd.Series = bank_marketing.data.targets.iloc[:, 0].copy()

    logger.info(
        "Dataset loaded — features: %s, target rows: %d",
        X.shape,
        len(y),
    )
    return X, y


def save_raw_to_disk(X: pd.DataFrame, y: pd.Series) -> Path:
    """
    Persist raw features + target as a single CSV in data/raw/.

    Returns the path of the saved file.
    """
    settings.ensure_paths()
    out_path = settings.data_raw_path / "bank_marketing_raw.csv"

    df = X.copy()
    df["subscribed"] = y.values   # y is a 1-D Series here

    df.to_csv(out_path, index=False)
    logger.info("Raw data saved to %s (%d rows)", out_path, len(df))
    return out_path


def load_raw_from_disk() -> pd.DataFrame:
    """Load previously saved raw CSV from disk."""
    path = settings.data_raw_path / "bank_marketing_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run the ingest step first."
        )
    df = pd.read_csv(path)
    logger.info("Raw data loaded from disk — %d rows, %d columns", *df.shape)
    return df


def load_raw_to_db(db: DatabaseClient, df: pd.DataFrame) -> int:
    """
    Insert the raw DataFrame into the raw_features table.

    Returns the number of rows inserted.
    """
    # Map DataFrame → DB column names
    db_df = df.rename(
        columns={
            "default": "default",   # reserved word; quoting handled by SQLAlchemy
        }
    )

    rows = db.bulk_insert("raw_features", db_df)
    logger.info("Inserted %d rows into raw_features", rows)
    return rows


def run_ingest(load_to_db: bool = True) -> pd.DataFrame:
    """
    Full ingestion step:
      1. Download from UCI
      2. Save to disk
      3. (optionally) load into PostgreSQL

    Returns the combined raw DataFrame.
    """
    X, y = fetch_raw_data()
    save_raw_to_disk(X, y)

    df = X.copy()
    df["subscribed"] = y.values   # y is a 1-D Series

    if load_to_db:
        try:
            db = DatabaseClient()
            db.ensure_schema()
            load_raw_to_db(db, df)
        except Exception as exc:
            logger.warning(
                "Could not load data into PostgreSQL (%s). "
                "Continuing without DB — CSV is available on disk.",
                exc,
            )

    return df
