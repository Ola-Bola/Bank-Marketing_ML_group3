"""
Data cleaning and train/val/test splitting.

This module is responsible for:
  - Handling 'unknown' values in categorical columns
  - Encoding the binary target (yes → 1, no → 0)
  - Dropping the 'duration' column when include_duration=False
    (duration is known only after the call; using it causes data leakage
    in realistic deployment scenarios)
  - Splitting data into train / val / test subsets
  - Saving the cleaned DataFrame to data/processed/

The actual feature-engineering pipeline (sklearn transformers) lives in
src/features/build_features.py.

Column reference (ucimlrepo id=222, bank-full version)
-------------------------------------------------------
Categorical : job, marital, education, default, housing, loan,
              contact, month, poutcome
Numerical   : age, balance, day_of_week (day-of-month int), campaign,
              pdays, previous
Excluded    : duration (data leakage)
Engineered  : previously_contacted  (pdays != 999)
Target      : subscribed (yes→1, no→0)
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings

logger = logging.getLogger(__name__)

# ── Column categories ─────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

NUMERICAL_COLS = [
    # day_of_week here is actually the day-of-month (1-31) as an integer
    "age", "balance", "day_of_week", "campaign", "pdays", "previous",
]

TARGET_COL = "subscribed"


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dataset-specific cleaning rules.

    Steps
    -----
    1. Rename dot-columns that slipped through (e.g. from CSV reload).
    2. Optionally drop 'duration' (data leakage for deployment).
    3. Replace 'unknown' strings with NaN so imputation can handle them later.
    4. Encode target: 'yes' → 1, 'no' → 0.
    5. Fix pdays: 999 means "not contacted" — keep as numeric sentinel; models
       will learn this. Also create a helper boolean flag.
    """
    df = df.copy()

    # 1. Normalise column names
    df.columns = [c.replace(".", "_") for c in df.columns]

    # 2. Drop duration if not including it
    if not settings.include_duration and "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.info("Dropped 'duration' column (data leakage prevention).")

    # 3. Replace 'unknown' with NaN in categoricals present in df
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    df[cat_cols_present] = df[cat_cols_present].replace("unknown", pd.NA)

    # 4. Encode target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})
        if df[TARGET_COL].isna().any():
            raise ValueError("Unexpected values in target column after mapping.")

    # 5. Flag previously-not-contacted clients
    if "pdays" in df.columns:
        df["previously_contacted"] = (df["pdays"] != 999).astype(int)

    logger.info(
        "Cleaned DataFrame: %d rows, %d columns, %.1f%% positive class",
        len(df),
        df.shape[1],
        100 * df[TARGET_COL].mean() if TARGET_COL in df.columns else float("nan"),
    )
    return df


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    Returns (train_df, val_df, test_df).
    """
    test_size = test_size or settings.test_size
    val_size = val_size or settings.val_size
    rs = random_state if random_state is not None else settings.random_state

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COL],
        random_state=rs,
    )

    # val_size is expressed as a fraction of the *original* dataset
    val_fraction_of_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        stratify=train_val[TARGET_COL],
        random_state=rs,
    )

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train), 100 * len(train) / len(df),
        len(val),   100 * len(val)   / len(df),
        len(test),  100 * len(test)  / len(df),
    )
    return train, val, test


# ── Save processed data ───────────────────────────────────────────────────────

def save_processed(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Persist train/val/test splits as Parquet files.

    index=True preserves the original raw-data row index so that
    raw_id = index + 1 maps back to raw_features.id in PostgreSQL.
    """
    settings.ensure_paths()
    base = settings.data_processed_path
    train.to_parquet(base / "train.parquet", index=True)
    val.to_parquet(base / "val.parquet", index=True)
    test.to_parquet(base / "test.parquet", index=True)
    logger.info("Processed splits saved to %s", base)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test Parquet files from disk."""
    base = settings.data_processed_path
    splits = {}
    for name in ("train", "val", "test"):
        path = base / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed split '{name}' not found. Run the preprocess step first."
            )
        splits[name] = pd.read_parquet(path)
    return splits["train"], splits["val"], splits["test"]


# ── DB write helpers ──────────────────────────────────────────────────────────

def _save_splits_to_db(splits: dict[str, pd.DataFrame]) -> None:
    """Write cleaned splits to the processed_features table in PostgreSQL.

    raw_id = pandas index + 1 because raw_features uses SERIAL (starts at 1)
    and rows were inserted in original CSV order.
    features_json stores every feature column (excl. target) as JSONB.
    """
    from src.data.database import DatabaseClient

    frames = []
    for split_name, df in splits.items():
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        # Use pandas' JSON encoder (handles numpy types and NaN → null)
        features_list = json.loads(df[feature_cols].to_json(orient="records"))
        frames.append(pd.DataFrame({
            "raw_id":        df.index + 1,
            "features_json": [json.dumps(r) for r in features_list],
            "subscribed":    df[TARGET_COL].astype(int).values,
            "split":         split_name,
        }))

    combined = pd.concat(frames, ignore_index=True)
    db = DatabaseClient()
    db.ensure_schema()
    n = db.bulk_insert("processed_features", combined)
    logger.info("Saved %d rows to processed_features (train/val/test)", n)


# ── Main entry ────────────────────────────────────────────────────────────────

def run_preprocess(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing step: clean → split → save.

    Returns (train_df, val_df, test_df).
    """
    cleaned = clean_raw(raw_df)
    train, val, test = split_data(cleaned)
    save_processed(train, val, test)

    try:
        _save_splits_to_db({"train": train, "val": val, "test": test})
    except Exception as exc:
        logger.warning("Could not save processed features to DB: %s", exc)

    return train, val, test
