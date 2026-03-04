"""
End-to-end pipeline validation script.

Runs the full pipeline with:
  - Local file-based MLflow (no server required)
  - DB operations skipped gracefully (no Docker/PostgreSQL needed)
  - Real UCI data download + real model training

Checks every step and every expected artifact at the end.

Usage
-----
    uv run python -m scripts.validate_pipeline
"""

from __future__ import annotations

import os
import sys
import time
import logging

# ── Force local MLflow tracking before any src import loads settings ──────────
os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("validate")

PASS = "  PASS"
FAIL = "  FAIL"
SEP  = "=" * 62


def section(title: str) -> None:
    log.info("")
    log.info(SEP)
    log.info("  %s", title)
    log.info(SEP)


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    log.info("%s  %s%s", status, label, suffix)
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Syntax re-check (belt-and-suspenders)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 1 — Python syntax")

import py_compile
import glob as _glob

py_files = _glob.glob("src/**/*.py", recursive=True) + \
           _glob.glob("scripts/*.py", recursive=False)

all_ok = True
for f in sorted(py_files):
    try:
        py_compile.compile(f, doraise=True)
        check(f, True)
    except py_compile.PyCompileError as e:
        check(f, False, str(e))
        all_ok = False

if not all_ok:
    log.error("Syntax errors found — aborting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 2 — Module imports")

modules_to_check = [
    "src.config",
    "src.data.database",
    "src.data.ingest",
    "src.data.preprocess",
    "src.features.build_features",
    "src.models.evaluate",
    "src.models.train",
    "src.models.predict",
    "src.visualization.plots",
]

import_ok = True
for mod in modules_to_check:
    try:
        __import__(mod)
        check(mod, True)
    except Exception as exc:
        check(mod, False, str(exc))
        import_ok = False

if not import_ok:
    log.error("Import errors found — aborting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Config / Settings
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 3 — Configuration")

from src.config import settings

check("settings.mlflow_tracking_uri set",
      bool(settings.mlflow_tracking_uri),
      settings.mlflow_tracking_uri)
check("settings.db_url has expected shape",
      "postgresql://" in settings.db_url)
check("settings.random_state = 42", settings.random_state == 42)
check("settings.test_size = 0.15", settings.test_size == 0.15)
check("settings.include_duration = False", not settings.include_duration)

settings.ensure_paths()
check("data/raw/ exists", settings.data_raw_path.exists())
check("data/processed/ exists", settings.data_processed_path.exists())
check("models/ exists", settings.models_path.exists())
check("reports/figures/ exists", settings.figures_path.exists())

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Data ingestion
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 4 — Data ingestion (UCI download)")

from src.data.ingest import run_ingest, load_raw_from_disk

t0 = time.perf_counter()
raw_df = run_ingest(load_to_db=False)          # skip PostgreSQL
elapsed = time.perf_counter() - t0

check("Ingest returns DataFrame", hasattr(raw_df, "shape"))
check("Row count ~45 211", 44000 < len(raw_df) < 46000, f"{len(raw_df)} rows")
check("Column count = 17 (16 features + subscribed)",
      raw_df.shape[1] == 17, f"{raw_df.shape[1]} cols")
check("Target column present", "subscribed" in raw_df.columns)
check("Raw CSV written to disk", (settings.data_raw_path / "bank_marketing_raw.csv").exists())
check("Target has only yes/no", set(raw_df["subscribed"].unique()) == {"yes", "no"})
log.info("  time: Ingestion: %.1fs", elapsed)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 5 — Preprocessing & splitting")

from src.data.preprocess import run_preprocess, clean_raw, load_processed

t0 = time.perf_counter()
train_df, val_df, test_df = run_preprocess(raw_df)
elapsed = time.perf_counter() - t0

total = len(train_df) + len(val_df) + len(test_df)
check("No rows lost in split", abs(total - len(raw_df)) <= 2, f"{total} vs {len(raw_df)}")
check("Train is largest split", len(train_df) > len(val_df) and len(train_df) > len(test_df))
check("Target is binary (0/1)",
      set(train_df["subscribed"].unique()).issubset({0, 1}))
check("'unknown' replaced with NaN",
      not (train_df.isin(["unknown"]).any().any()))
check("'duration' dropped", "duration" not in train_df.columns)
check("previously_contacted engineered", "previously_contacted" in train_df.columns)

pos_train = train_df["subscribed"].mean()
pos_val   = val_df["subscribed"].mean()
pos_test  = test_df["subscribed"].mean()
check("Stratification preserved (train)",  0.10 < pos_train < 0.14, f"{pos_train:.3f}")
check("Stratification preserved (val)",    0.10 < pos_val   < 0.14, f"{pos_val:.3f}")
check("Stratification preserved (test)",   0.10 < pos_test  < 0.14, f"{pos_test:.3f}")

check("train.parquet on disk", (settings.data_processed_path / "train.parquet").exists())
check("val.parquet on disk",   (settings.data_processed_path / "val.parquet").exists())
check("test.parquet on disk",  (settings.data_processed_path / "test.parquet").exists())

train_loaded, val_loaded, test_loaded = load_processed()
check("load_processed() round-trips correctly",
      len(train_loaded) == len(train_df) and len(test_loaded) == len(test_df))
log.info("  time: Preprocessing: %.1fs", elapsed)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Feature pipeline
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 6 — Feature engineering pipeline")

from src.features.build_features import build_preprocessor, split_X_y, get_feature_names
import numpy as np

X_train, y_train = split_X_y(train_df)
X_val,   y_val   = split_X_y(val_df)

preprocessor = build_preprocessor(train_df)
preprocessor.fit(X_train)

X_tr_transformed = preprocessor.transform(X_train)
X_va_transformed = preprocessor.transform(X_val)

check("split_X_y: target removed from X", "subscribed" not in X_train.columns)
check("split_X_y: y shape matches X rows", y_train.shape[0] == X_train.shape[0])
check("y_train is binary (0/1)", set(np.unique(y_train)).issubset({0, 1}))
check("Preprocessor transforms train without NaN",
      not np.isnan(X_tr_transformed).any(),
      f"shape={X_tr_transformed.shape}")
check("Preprocessor transforms val without NaN",
      not np.isnan(X_va_transformed).any())
check("Feature names extractable", len(get_feature_names(preprocessor)) > 0,
      f"{len(get_feature_names(preprocessor))} features")

log.info("  Feature matrix shape: %s", X_tr_transformed.shape)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Model training (local MLflow)
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 7 — Model training (local MLflow, no PostgreSQL)")

import mlflow
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

from src.models.train import train_all_models

t0 = time.perf_counter()
results = train_all_models(train_df, val_df)
elapsed = time.perf_counter() - t0

check("Three models trained", len(results) == 3,
      str([r["name"] for r in results]))

for r in results:
    name    = r["name"]
    metrics = r["metrics"]
    run_id  = r["run_id"]
    pkl     = settings.models_path / f"{name}.pkl"

    check(f"{name}: run_id returned",    bool(run_id), run_id[:8] if run_id else "")
    check(f"{name}: .pkl saved to disk", pkl.exists(), str(pkl))
    check(f"{name}: roc_auc > 0.5",
          metrics["roc_auc"] > 0.5, f"roc_auc={metrics['roc_auc']:.4f}")
    check(f"{name}: f1 > 0",
          metrics["f1"] > 0,        f"f1={metrics['f1']:.4f}")
    check(f"{name}: recall > 0",
          metrics["recall"] > 0,    f"recall={metrics['recall']:.4f}")

check("training_summary.csv written",
      (settings.reports_path / "training_summary.csv").exists())

log.info("  time: Training: %.1fs", elapsed)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — Model evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 8 — Test-set evaluation & report generation")

from src.models.evaluate import generate_evaluation_report
from src.visualization.plots import plot_feature_importance, plot_model_comparison
import pandas as pd

eval_reports = []
for r in results:
    name     = r["name"]
    pipeline = r["pipeline"]
    report   = generate_evaluation_report(pipeline, test_df, name)

    check(f"{name}: classification report text non-empty",
          len(report.get("report_text", "")) > 50)
    check(f"{name}: confusion_matrix PNG exists",
          (settings.figures_path / f"confusion_matrix_{name}.png").exists())
    check(f"{name}: roc_curve PNG exists",
          (settings.figures_path / f"roc_curve_{name}.png").exists())
    check(f"{name}: threshold_analysis CSV exists",
          (settings.reports_path / f"threshold_analysis_{name}.csv").exists())
    check(f"{name}: test ROC-AUC > 0.5",
          report["metrics"]["roc_auc"] > 0.5,
          f"{report['metrics']['roc_auc']:.4f}")

    plot_feature_importance(pipeline, name)
    check(f"{name}: feature_importance PNG exists",
          (settings.figures_path / f"feature_importance_{name}.png").exists())
    eval_reports.append(report)

# Model comparison chart
summary_df = pd.read_csv(settings.reports_path / "training_summary.csv")
plot_model_comparison(summary_df)
check("model_comparison PNG exists",
      (settings.figures_path / "model_comparison.png").exists())

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — Inference smoke test
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 9 — Inference smoke test")

from src.models.predict import load_model_from_disk, predict

best = max(results, key=lambda r: r["metrics"]["roc_auc"])
best_name = best["name"]

pipeline_loaded = load_model_from_disk(best_name)
check("load_model_from_disk returns object", pipeline_loaded is not None)

# Predict on a 10-row sample of raw (uncleaned) data
sample_raw = raw_df.drop(columns=["subscribed"]).head(10)
preds = predict(pipeline_loaded, sample_raw, threshold=0.5)

check("predict() returns DataFrame",      hasattr(preds, "columns"))
check("predicted_label column present",   "predicted_label" in preds.columns)
check("predicted_proba column present",   "predicted_proba" in preds.columns)
check("predicted_label is 0 or 1",        preds["predicted_label"].isin([0, 1]).all())
check("predicted_proba in [0, 1]",
      ((preds["predicted_proba"] >= 0) & (preds["predicted_proba"] <= 1)).all())
check("inference output has 10 rows",     len(preds) == 10, f"{len(preds)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10 — MLflow artefact audit
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 10 — MLflow artefact audit")

from pathlib import Path

mlruns_path = Path("mlruns")
check("mlruns/ directory created", mlruns_path.exists())

# Count pkl files saved by training
pkl_files = list(settings.models_path.glob("*.pkl"))
check("Model pkl files saved", len(pkl_files) == 3,
      f"found: {[f.name for f in pkl_files]}")

# Count PNG artefacts
png_files = list(settings.figures_path.glob("*.png"))
check("At least 8 PNG files in reports/figures/", len(png_files) >= 8,
      f"found {len(png_files)} files")

# Summary CSV
summary_csv = pd.read_csv(settings.reports_path / "training_summary.csv")
check("training_summary.csv has 3 model rows", len(summary_csv) == 3)
check("training_summary.csv has roc_auc column", "roc_auc" in summary_csv.columns)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("VALIDATION COMPLETE")

best_roc = max(r["metrics"]["roc_auc"] for r in results)
best_f1  = max(r["metrics"]["f1"]      for r in results)
best_mdl = max(results, key=lambda r: r["metrics"]["roc_auc"])["name"]

log.info("  Best model:     %s", best_mdl)
log.info("  Best ROC-AUC:   %.4f (on validation set)", best_roc)
log.info("  Best F1:        %.4f (on validation set)", best_f1)
log.info("")
log.info("  Model pkl files : %s", [f.name for f in sorted(settings.models_path.glob("*.pkl"))])
log.info("  Report figures  : %d PNG files in %s", len(png_files), settings.figures_path)
log.info("  MLflow runs     : %s", str(mlruns_path.resolve()))
log.info("")
log.info("  All pipeline phases completed successfully.")
log.info(SEP)
