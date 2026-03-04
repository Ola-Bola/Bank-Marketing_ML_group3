"""
Model evaluation utilities.

Provides:
  - compute_metrics() — classification metrics for a fitted pipeline
  - generate_evaluation_report() — full report on the test set for the best model
  - MetricsDict type alias

Also writes confusion-matrix and ROC-curve plots to reports/figures/.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import settings
from src.features.build_features import split_X_y

logger = logging.getLogger(__name__)


class MetricsDict(TypedDict):
    model_name: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str = "",
    threshold: float = 0.5,
) -> MetricsDict:
    """
    Compute classification metrics for a fitted pipeline.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X          : feature DataFrame (not yet preprocessed)
    y          : true labels
    model_name : label for logging
    threshold  : decision threshold for binary predictions (default 0.5)
    """
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics: MetricsDict = {
        "model_name": model_name,
        "roc_auc":    round(float(roc_auc_score(y, y_proba)), 4),
        "f1":         round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "precision":  round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "accuracy":   round(float(accuracy_score(y, y_pred)), 4),
    }
    return metrics


# ── Full evaluation report ────────────────────────────────────────────────────

def generate_evaluation_report(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    model_name: str,
    mlflow_run_id: str | None = None,
) -> dict:
    """
    Generate a comprehensive evaluation report on the held-out test set.

    Saves:
      - confusion_matrix_<model_name>.png
      - roc_curve_<model_name>.png
      - classification_report_<model_name>.txt
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings.ensure_paths()
    X_test, y_test = split_X_y(test_df)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(pipeline, X_test, y_test, model_name)

    # ── Classification report ─────────────────────────────────────────────────
    report_text = classification_report(
        y_test, y_pred, target_names=["no (0)", "yes (1)"]
    )
    report_path = settings.reports_path / f"classification_report_{model_name}.txt"
    report_path.write_text(report_text)
    logger.info("Classification report:\n%s", report_text)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}\n(test set)")
    fig.tight_layout()
    cm_path = settings.figures_path / f"confusion_matrix_{model_name}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", cm_path)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = settings.figures_path / f"roc_curve_{model_name}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s", roc_path)

    # ── Threshold analysis ────────────────────────────────────────────────────
    thresholds = np.arange(0.1, 0.9, 0.05)
    threshold_df = pd.DataFrame([
        {
            "threshold": t,
            "precision": precision_score(y_test, (y_proba >= t).astype(int), zero_division=0),
            "recall":    recall_score(y_test, (y_proba >= t).astype(int), zero_division=0),
            "f1":        f1_score(y_test, (y_proba >= t).astype(int), zero_division=0),
        }
        for t in thresholds
    ])
    threshold_path = settings.reports_path / f"threshold_analysis_{model_name}.csv"
    threshold_df.to_csv(threshold_path, index=False)

    # ── Save predictions to DB ────────────────────────────────────────────────
    try:
        from src.data.database import DatabaseClient
        db = DatabaseClient()
        preds_df = pd.DataFrame({
            "raw_id":          test_df.index + 1,   # SERIAL starts at 1
            "model_name":      model_name,
            "mlflow_run_id":   mlflow_run_id,
            "predicted_label": y_pred.astype(int),
            "predicted_proba": y_proba.round(5),
            "actual_label":    y_test.astype(int),
        })
        db.bulk_insert("predictions", preds_df)
        logger.info("Saved %d predictions to DB for '%s'", len(preds_df), model_name)
    except Exception as exc:
        logger.warning("Could not save predictions to DB: %s", exc)

    result = {
        "model_name": model_name,
        "metrics": metrics,
        "report_text": report_text,
        "cm_path": str(cm_path),
        "roc_path": str(roc_path),
        "threshold_analysis": threshold_df,
    }
    logger.info(
        "Evaluation complete — ROC-AUC=%.4f, F1=%.4f",
        metrics["roc_auc"], metrics["f1"],
    )
    return result
