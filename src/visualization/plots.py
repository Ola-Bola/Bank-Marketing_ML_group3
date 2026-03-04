"""
EDA and reporting visualisations for the Bank-Marketing dataset.

All functions save PNG files to reports/figures/ and return the figure object
so they can be embedded in notebooks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.config import settings

logger = logging.getLogger(__name__)

# ── Style defaults ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQ = (7, 6)
DPI = 150


def _save(fig: plt.Figure, filename: str) -> Path:
    settings.ensure_paths()
    path = settings.figures_path / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    logger.info("Saved figure: %s", path)
    return path


# ── Target distribution ───────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing class imbalance in the target variable."""
    counts = df["subscribed"].value_counts()
    labels = counts.index.map({1: "Yes (subscribed)", 0: "No"})

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Count plot
    axes[0].bar(labels, counts.values, color=["#4878cf", "#d65f5f"])
    axes[0].set_title("Target Distribution — Absolute Count")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}",
            ha="center", fontsize=10,
        )

    # Percentage pie
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4878cf", "#d65f5f"],
    )
    axes[1].set_title("Target Distribution — Proportion")

    fig.suptitle("Class Imbalance: Term Deposit Subscription", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "target_distribution.png")
    return fig


# ── Numeric feature distributions ────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    """Histograms for all numeric features, coloured by target class."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "subscribed"]

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        for label, colour in [(0, "#d65f5f"), (1, "#4878cf")]:
            subset = df.loc[df["subscribed"] == label, col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=colour,
                    label=("Yes" if label == 1 else "No"), density=True)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Target Class", fontsize=13)
    fig.tight_layout()
    _save(fig, "numeric_distributions.png")
    return fig


# ── Categorical feature counts ────────────────────────────────────────────────

def plot_categorical_subscription_rate(df: pd.DataFrame) -> plt.Figure:
    """
    For each categorical feature, plot the subscription rate per category.
    Helps identify which groups are most/least likely to subscribe.
    """
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Also include categorical columns stored as string (from raw CSV)
    if not cat_cols:
        cat_cols = [
            c for c in ["job", "marital", "education", "contact", "poutcome",
                         "month", "day_of_week"]
            if c in df.columns
        ]

    n_cols = 2
    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        rate = (
            df.groupby(col)["subscribed"]
            .mean()
            .sort_values(ascending=False)
        )
        bars = ax.barh(rate.index, rate.values, color="#4878cf")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"Subscription Rate by {col}", fontsize=10)
        ax.set_xlabel("% subscribed")
        for bar, val in zip(bars, rate.values):
            ax.text(
                val + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center", fontsize=8,
            )

    for ax in axes[len(cat_cols):]:
        ax.set_visible(False)

    fig.suptitle("Subscription Rate by Categorical Feature", fontsize=13)
    fig.tight_layout()
    _save(fig, "categorical_subscription_rates.png")
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=13)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")
    return fig


# ── Age distribution ──────────────────────────────────────────────────────────

def plot_age_by_subscription(df: pd.DataFrame) -> plt.Figure:
    """Box-plot of age split by subscription outcome."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    df_plot = df.copy()
    df_plot["Subscribed"] = df_plot["subscribed"].map({1: "Yes", 0: "No"})
    sns.boxplot(data=df_plot, x="Subscribed", y="age", palette=["#d65f5f", "#4878cf"], ax=ax)
    ax.set_title("Age Distribution by Subscription Outcome")
    ax.set_xlabel("Subscribed to Term Deposit")
    ax.set_ylabel("Age")
    fig.tight_layout()
    _save(fig, "age_by_subscription.png")
    return fig


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 20) -> plt.Figure:
    """
    Horizontal bar chart of the top-N most important features.
    Works for tree-based models (feature_importances_) and Logistic Regression (coef_).
    """
    from src.features.build_features import get_feature_names

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning("Model %s has no feature importances.", model_name)
        return None

    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#4878cf")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    _save(fig, f"feature_importance_{model_name}.png")
    return fig


# ── Model comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart comparing multiple models on key metrics.

    Parameters
    ----------
    summary_df : DataFrame with columns [model, roc_auc, f1, precision, recall, accuracy]
    """
    metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    x = np.arange(len(summary_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, summary_df[metric], width, label=metric)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(summary_df["model"], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Validation Metrics")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout()
    _save(fig, "model_comparison.png")
    return fig
