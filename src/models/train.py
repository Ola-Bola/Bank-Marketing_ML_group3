"""
Model training with MLflow experiment tracking.

Three classifiers are trained and compared:
  1. Logistic Regression        (interpretable baseline)
  2. Random Forest              (ensemble, handles non-linearity)
  3. XGBoost Gradient Boosting  (usually best on tabular data)

Each model is wrapped in a full sklearn Pipeline (preprocessor + classifier)
so the artefact saved to MLflow is self-contained for inference.

Class imbalance (~11.7 % positive) is handled via class_weight='balanced'
(equivalent to SMOTE but computationally cheaper).

Usage
-----
    from src.models.train import train_all_models
    results = train_all_models(train_df, val_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import settings
from src.data.database import DatabaseClient
from src.features.build_features import build_preprocessor, split_X_y
from src.models.evaluate import compute_metrics, MetricsDict

logger = logging.getLogger(__name__)


# ── Model zoo ─────────────────────────────────────────────────────────────────

def _model_configs() -> list[dict[str, Any]]:
    """Return list of (name, estimator, params) dicts to train."""
    return [
        {
            "name": "logistic_regression",
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=settings.random_state,
            ),
            "params": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
            },
        },
        {
            "name": "random_forest",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=settings.random_state,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": "None",
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
        },
        {
            "name": "xgboost",
            "estimator": XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=7,  # approx. ratio of negatives to positives
                eval_metric="logloss",
                random_state=settings.random_state,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 7,
            },
        },
    ]


# ── Single model training ─────────────────────────────────────────────────────

def train_model(
    model_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Pipeline, MetricsDict, str]:
    """
    Train one model, log everything to MLflow, persist artefact locally.

    Returns
    -------
    pipeline   : fitted sklearn Pipeline (preprocessor + classifier)
    metrics    : evaluation metrics dict
    run_id     : MLflow run ID
    """
    name = model_cfg["name"]
    estimator = model_cfg["estimator"]
    params = model_cfg["params"]

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)

    preprocessor = build_preprocessor(train_df)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started — model=%s, run_id=%s", name, run_id)

        # Cross-validation on train set (gives honest in-sample estimate)
        cv = StratifiedKFold(
            n_splits=settings.cv_folds,
            shuffle=True,
            random_state=settings.random_state,
        )
        cv_results = cross_validate(
            pipeline,
            X_train, y_train,
            cv=cv,
            scoring=["roc_auc", "f1", "precision", "recall"],
            return_train_score=False,
            n_jobs=-1,
        )
        mlflow.log_metrics({
            f"cv_mean_{k.replace('test_', '')}": float(np.mean(v))
            for k, v in cv_results.items()
            if k.startswith("test_")
        })

        # Full fit on all training data
        pipeline.fit(X_train, y_train)

        # Validation metrics
        metrics = compute_metrics(pipeline, X_val, y_val, name)
        mlflow.log_metrics({
            "val_roc_auc":   metrics["roc_auc"],
            "val_f1":        metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall":    metrics["recall"],
            "val_accuracy":  metrics["accuracy"],
        })

        # Log params and tags
        mlflow.log_params(params)
        mlflow.set_tags({
            "model_type": name,
            "include_duration": str(settings.include_duration),
        })

        # Log the pipeline as a model artefact
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"bank_marketing_{name}",
        )

        # Save locally for quick access
        settings.ensure_paths()
        local_path = settings.models_path / f"{name}.pkl"
        joblib.dump(pipeline, local_path)
        mlflow.log_artifact(str(local_path))
        logger.info(
            "Model '%s' — val ROC-AUC=%.4f, val F1=%.4f",
            name, metrics["roc_auc"], metrics["f1"],
        )

    return pipeline, metrics, run_id


# ── Train all models ──────────────────────────────────────────────────────────

def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Train all models in the model zoo and return a summary list.

    Each element is a dict with keys: name, pipeline, metrics, run_id.
    """
    # Attempt DB connection once; if unavailable, skip gracefully
    try:
        db = DatabaseClient()
        db.ensure_schema()
    except Exception as exc:
        logger.warning("PostgreSQL unavailable — experiment results won't be saved to DB: %s", exc)
        db = None

    results = []
    for cfg in _model_configs():
        logger.info("=== Training: %s ===", cfg["name"])
        pipeline, metrics, run_id = train_model(cfg, train_df, val_df)
        results.append({
            "name": cfg["name"],
            "pipeline": pipeline,
            "metrics": metrics,
            "run_id": run_id,
        })

        # Persist to experiment_results table if DB is available
        if db is not None:
            try:
                import json
                db.save_experiment_result({
                    "mlflow_run_id":    run_id,
                    "model_name":       cfg["name"],
                    "roc_auc":          metrics["roc_auc"],
                    "f1_score":         metrics["f1"],
                    "precision_score":  metrics["precision"],
                    "recall_score":     metrics["recall"],
                    "accuracy":         metrics["accuracy"],
                    "params_json":      json.dumps(cfg["params"]),
                })
            except Exception as exc:
                logger.warning("Could not save experiment result to DB: %s", exc)

    # Identify best model by validation ROC-AUC
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    logger.info(
        "Best model: %s (val ROC-AUC=%.4f)",
        best["name"], best["metrics"]["roc_auc"],
    )

    # Save a summary CSV
    summary_df = pd.DataFrame([
        {"model": r["name"], "run_id": r["run_id"], **r["metrics"]}
        for r in results
    ])
    summary_path = settings.reports_path / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Training summary saved to %s", summary_path)

    return results
