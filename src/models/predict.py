"""
Inference module.

Loads the best registered model from MLflow (or a local .pkl) and generates
predictions for new observations.

Usage (CLI via make evaluate):
    python -m scripts.run_pipeline --predict --input path/to/new_data.csv

Usage (Python API):
    from src.models.predict import load_model, predict
    pipeline = load_model("xgboost")
    preds = predict(pipeline, new_df)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import settings
from src.data.preprocess import clean_raw

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_from_disk(model_name: str) -> object:
    """Load a pickled pipeline from models/<model_name>.pkl."""
    path = settings.models_path / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}. "
            "Run the training step first."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return pipeline


def load_model_from_mlflow(
    model_name: str,
    stage: str = "Production",
) -> object:
    """
    Load the latest model from the MLflow Model Registry.

    Falls back to 'None' stage (all versions) if no model is in Production.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/bank_marketing_{model_name}/{stage}"
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from MLflow: %s", model_uri)
    except Exception:
        # Fallback: load latest version regardless of stage
        model_uri = f"models:/bank_marketing_{model_name}/latest"
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded latest model version from MLflow: %s", model_uri)
    return pipeline


def load_model(model_name: str = "xgboost", prefer_mlflow: bool = False) -> object:
    """
    Load a pipeline, preferring MLflow registry if available.

    Falls back gracefully to the local .pkl file.
    """
    if prefer_mlflow:
        try:
            return load_model_from_mlflow(model_name)
        except Exception as exc:
            logger.warning("MLflow load failed (%s); falling back to disk.", exc)
    return load_model_from_disk(model_name)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    pipeline: object,
    raw_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate predictions for raw input data.

    Parameters
    ----------
    pipeline  : fitted sklearn Pipeline (preprocessor + classifier)
    raw_df    : raw feature DataFrame (same schema as training data, minus target)
    threshold : decision threshold for positive class

    Returns
    -------
    DataFrame with columns: predicted_label, predicted_proba
    """
    # Apply the same cleaning used during training (no target expected here)
    df = clean_raw(raw_df) if "subscribed" not in raw_df.columns else clean_raw(raw_df)

    # Drop target if accidentally included
    X = df.drop(columns=["subscribed"], errors="ignore")

    proba = pipeline.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)

    result = pd.DataFrame({
        "predicted_label": labels,
        "predicted_proba": proba.round(5),
    }, index=raw_df.index)

    n_pos = labels.sum()
    logger.info(
        "Predictions: %d rows, %d positive (%.1f%%), threshold=%.2f",
        len(result), n_pos, 100 * n_pos / len(result), threshold,
    )
    return result


def predict_from_csv(
    input_path: str | Path,
    model_name: str = "xgboost",
    output_path: str | Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    End-to-end: load CSV → clean → predict → optionally save results.
    """
    raw_df = pd.read_csv(input_path)
    pipeline = load_model(model_name)
    preds = predict(pipeline, raw_df, threshold=threshold)

    if output_path:
        preds.to_csv(output_path, index=False)
        logger.info("Predictions saved to %s", output_path)

    return preds
