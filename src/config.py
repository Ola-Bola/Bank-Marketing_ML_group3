"""
Centralised project configuration.

Settings are loaded from environment variables or a .env file.
Override any value by setting the corresponding environment variable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is two levels above this file  (src/config.py → repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunable knobs for the project, sourced from env / .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "bank_marketing"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "bank_marketing_classification"

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_raw_path: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_path: Path = PROJECT_ROOT / "data" / "processed"
    models_path: Path = PROJECT_ROOT / "models"
    reports_path: Path = PROJECT_ROOT / "reports"
    figures_path: Path = PROJECT_ROOT / "reports" / "figures"

    # ── ML training ───────────────────────────────────────────────────────────
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    cv_folds: int = 5

    # ── Feature flags ─────────────────────────────────────────────────────────
    # 'duration' is only known after the call ends; set False for realistic model
    include_duration: bool = False

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_paths(self) -> None:
        """Create all output directories if they do not exist."""
        for path in (
            self.data_raw_path,
            self.data_processed_path,
            self.models_path,
            self.reports_path,
            self.figures_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
