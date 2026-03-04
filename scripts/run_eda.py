"""
Exploratory Data Analysis (EDA) script.

Loads the raw dataset (downloading it if needed), generates descriptive
statistics and all EDA plots, and saves them to reports/figures/.

Usage
-----
    python -m scripts.run_eda
    # or
    make eda
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eda")


def main() -> None:
    from src.config import settings
    from src.data.ingest import fetch_raw_data, save_raw_to_disk
    from src.data.preprocess import clean_raw
    from src.visualization.plots import (
        plot_age_by_subscription,
        plot_categorical_subscription_rate,
        plot_correlation_heatmap,
        plot_numeric_distributions,
        plot_target_distribution,
    )

    settings.ensure_paths()

    # ── Load data ─────────────────────────────────────────────────────────────
    raw_csv = settings.data_raw_path / "bank_marketing_raw.csv"
    if raw_csv.exists():
        logger.info("Loading raw data from disk...")
        raw_df = pd.read_csv(raw_csv)
    else:
        logger.info("Raw data not found — downloading from UCI...")
        X, y = fetch_raw_data()
        save_raw_to_disk(X, y)
        raw_df = X.copy()
        raw_df["subscribed"] = y.values

    logger.info("Dataset shape: %s", raw_df.shape)
    logger.info("Columns: %s", list(raw_df.columns))

    # ── Clean for EDA (keeps raw structure, just encodes target) ──────────────
    eda_df = clean_raw(raw_df)

    # ── Descriptive statistics ────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("DESCRIPTIVE STATISTICS")
    logger.info("%s", "=" * 60)

    logger.info("\nTarget distribution:\n%s", eda_df["subscribed"].value_counts())
    logger.info(
        "\nSubscription rate: %.2f%%",
        100 * eda_df["subscribed"].mean(),
    )

    num_stats = eda_df.select_dtypes(include="number").describe().T
    logger.info("\nNumeric feature statistics:\n%s", num_stats.to_string())

    cat_cols = eda_df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        logger.info(
            "\nValue counts for '%s':\n%s",
            col,
            eda_df[col].value_counts(dropna=False).head(10),
        )

    # Save descriptive stats to CSV
    stats_path = settings.reports_path / "descriptive_stats.csv"
    num_stats.to_csv(stats_path)
    logger.info("\nDescriptive stats saved to %s", stats_path)

    # ── Missing value analysis ─────────────────────────────────────────────────
    missing = eda_df.isnull().sum()
    missing_pct = (missing / len(eda_df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0]
    if not missing_df.empty:
        logger.info("\nMissing values (after 'unknown' → NaN):\n%s", missing_df)
        missing_path = settings.reports_path / "missing_values.csv"
        missing_df.to_csv(missing_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("GENERATING PLOTS  →  %s", settings.figures_path)
    logger.info("%s", "=" * 60)

    plot_target_distribution(eda_df)
    plot_numeric_distributions(eda_df)
    plot_categorical_subscription_rate(eda_df)
    plot_correlation_heatmap(eda_df)
    plot_age_by_subscription(eda_df)

    logger.info("\nEDA complete. All figures saved to %s", settings.figures_path)


if __name__ == "__main__":
    main()
