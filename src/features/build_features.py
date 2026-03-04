"""
scikit-learn feature-engineering pipeline.

Produces a single ColumnTransformer that:
  - Imputes and one-hot-encodes categorical features
  - Imputes and standard-scales numerical features

The pipeline is intentionally kept as a sklearn Pipeline object so that it can
be serialised together with the model (no separate preprocessing step at
inference time).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocess import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Feature sets ──────────────────────────────────────────────────────────────

# Additional engineered features added by clean_raw()
ENGINEERED_NUMERICAL = ["previously_contacted"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Determine which numerical and categorical columns are present in *df*
    (accounts for optional 'duration' column and engineered features).
    """
    exclude = {TARGET_COL}
    all_cols = [c for c in df.columns if c not in exclude]

    numerical = [
        c for c in NUMERICAL_COLS + ENGINEERED_NUMERICAL
        if c in all_cols
    ]
    categorical = [c for c in CATEGORICAL_COLS if c in all_cols]

    # Catch any remaining columns not yet categorised
    known = set(numerical) | set(categorical)
    extra_num = [c for c in all_cols if c not in known and df[c].dtype != object]
    numerical.extend(extra_num)

    return numerical, categorical


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer fitted to the column names present in *df*.

    Does NOT fit — call `.fit(X_train)` on the returned object.
    """
    numerical_cols, categorical_cols = get_feature_columns(df)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",          # avoid multicollinearity
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Preprocessor built — %d numerical, %d categorical features",
        len(numerical_cols),
        len(categorical_cols),
    )
    return preprocessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after the preprocessor has been fitted."""
    return list(preprocessor.get_feature_names_out())


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a DataFrame into features X and target y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy(dtype=int)
    return X, y
