"""
End-to-end ML pipeline orchestrator.

Usage
-----
Run one step at a time:
    python -m scripts.run_pipeline --ingest
    python -m scripts.run_pipeline --preprocess
    python -m scripts.run_pipeline --train
    python -m scripts.run_pipeline --evaluate

Run the complete pipeline:
    python -m scripts.run_pipeline --all

Or use the Makefile shortcuts:
    make ingest | make preprocess | make train | make evaluate | make pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Reconfigure stdout/stderr to UTF-8 on Windows (cp1252 chokes on MLflow emoji output)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import pandas as pd

from src.config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


# ── Step functions (callable from pyproject.toml [project.scripts]) ───────────

def run_ingest() -> pd.DataFrame:
    from src.data.ingest import run_ingest as _ingest
    logger.info("-- STEP 1: Ingest ------------------------------------------")
    t0 = time.perf_counter()
    df = _ingest(load_to_db=True)
    logger.info("Ingest complete in %.1fs", time.perf_counter() - t0)
    return df


def run_preprocess(raw_df: pd.DataFrame | None = None):
    from src.data.ingest import load_raw_from_disk
    from src.data.preprocess import run_preprocess as _preprocess

    logger.info("-- STEP 2: Preprocess ---------------------------------------")
    t0 = time.perf_counter()
    if raw_df is None:
        raw_df = load_raw_from_disk()
    train, val, test = _preprocess(raw_df)
    logger.info("Preprocess complete in %.1fs", time.perf_counter() - t0)
    return train, val, test


def run_train(train_df: pd.DataFrame | None = None, val_df: pd.DataFrame | None = None):
    from src.data.preprocess import load_processed
    from src.models.train import train_all_models

    logger.info("-- STEP 3: Train --------------------------------------------")
    t0 = time.perf_counter()
    if train_df is None or val_df is None:
        train_df, val_df, _ = load_processed()
    results = train_all_models(train_df, val_df)
    logger.info("Training complete in %.1fs", time.perf_counter() - t0)
    return results


def run_evaluate(training_results=None):
    from src.data.preprocess import load_processed
    from src.models.evaluate import generate_evaluation_report
    from src.visualization.plots import (
        plot_feature_importance,
        plot_model_comparison,
    )

    logger.info("-- STEP 4: Evaluate -----------------------------------------")
    t0 = time.perf_counter()

    _, _, test_df = load_processed()

    reports = []
    if training_results:
        for res in training_results:
            fitted_pipeline = res["pipeline"]
            report = generate_evaluation_report(
                fitted_pipeline, test_df, res["name"],
                mlflow_run_id=res.get("run_id"),
            )
            reports.append(report)
            try:
                plot_feature_importance(fitted_pipeline, res["name"])
            except Exception as exc:
                logger.warning("Feature importance plot failed: %s", exc)

        # Model comparison chart
        try:
            summary_path = settings.reports_path / "training_summary.csv"
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                plot_model_comparison(summary_df)
        except Exception as exc:
            logger.warning("Model comparison plot failed: %s", exc)
    else:
        # Load the best local model for evaluation
        from src.models.predict import load_model_from_disk
        model_files = list(settings.models_path.glob("*.pkl"))
        if not model_files:
            logger.error("No trained models found. Run --train first.")
            return []
        for model_path in model_files:
            name = model_path.stem
            try:
                pipeline = load_model_from_disk(name)
                report = generate_evaluation_report(pipeline, test_df, name)
                reports.append(report)
                plot_feature_importance(pipeline, name)
            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", name, exc)

    logger.info("Evaluation complete in %.1fs", time.perf_counter() - t0)
    return reports


def run_full_pipeline() -> None:
    """Execute all pipeline steps sequentially."""
    logger.info("============================================================")
    logger.info("  Bank-Marketing ML — Full Pipeline")
    logger.info("============================================================")
    t_start = time.perf_counter()

    raw_df = run_ingest()
    train_df, val_df, _ = run_preprocess(raw_df)
    results = run_train(train_df, val_df)
    run_evaluate(results)

    elapsed = time.perf_counter() - t_start
    logger.info("============================================================")
    logger.info("  Pipeline complete in %.1fs", elapsed)
    logger.info("  Reports: %s", settings.reports_path)
    logger.info("  Models:  %s", settings.models_path)
    logger.info("  MLflow:  %s", settings.mlflow_tracking_uri)
    logger.info("============================================================")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bank-Marketing ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",        action="store_true", help="Run full pipeline")
    group.add_argument("--ingest",     action="store_true", help="Run ingest step only")
    group.add_argument("--preprocess", action="store_true", help="Run preprocess step only")
    group.add_argument("--train",      action="store_true", help="Run training step only")
    group.add_argument("--evaluate",   action="store_true", help="Run evaluation step only")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.all:
        run_full_pipeline()
    elif args.ingest:
        run_ingest()
    elif args.preprocess:
        run_preprocess()
    elif args.train:
        run_train()
    elif args.evaluate:
        run_evaluate()
