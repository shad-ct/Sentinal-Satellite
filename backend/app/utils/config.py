"""
Configuration management for the Sentinel-2 Change Detection System.

Loads all settings from environment variables (via .env file).
Fails fast with clear messages if required credentials are missing.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Copernicus Credentials ──────────────────────────────────────────────
    copernicus_client_id: Optional[str] = None
    copernicus_client_secret: Optional[str] = None
    copernicus_username: Optional[str] = None
    copernicus_password: Optional[str] = None

    # ── Sentinel Hub (optional) ────────────────────────────────────────────
    sentinelhub_client_id: Optional[str] = None
    sentinelhub_client_secret: Optional[str] = None

    # ── API URLs ───────────────────────────────────────────────────────────
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    cdse_catalog_url: str = "https://catalogue.dataspace.copernicus.eu"
    cdse_s3_endpoint: str = "https://eodata.dataspace.copernicus.eu"
    sentinelhub_base_url: str = "https://services.sentinel-hub.com"

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-to-a-random-secret-key"
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Data directories ───────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")

    # ── Search defaults ────────────────────────────────────────────────────
    default_max_cloud_coverage: float = 20.0
    default_search_window_days: int = 15
    default_resolution: int = 10  # metres

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @model_validator(mode="after")
    def ensure_data_dirs_exist(self) -> "Settings":
        """Create data directories if they do not already exist."""
        for sub in ["raw/sentinel2", "processed", "masks", "results", "thumbnails"]:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        for sub in ["pretrained", "checkpoints"]:
            (self.models_dir / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def has_copernicus_credentials(self) -> bool:
        """True if OAuth2 machine-to-machine credentials are available."""
        return bool(self.copernicus_client_id and self.copernicus_client_secret)

    @property
    def has_copernicus_user_credentials(self) -> bool:
        """True if username/password credentials are available."""
        return bool(self.copernicus_username and self.copernicus_password)

    @property
    def has_sentinelhub_credentials(self) -> bool:
        """True if Sentinel Hub credentials are configured."""
        return bool(self.sentinelhub_client_id and self.sentinelhub_client_secret)

    def require_copernicus_credentials(self) -> None:
        """
        Raise a RuntimeError with a helpful message if Copernicus
        credentials are not configured.
        """
        if not self.has_copernicus_credentials and not self.has_copernicus_user_credentials:
            raise RuntimeError(
                "\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  Missing Copernicus Data Space Ecosystem credentials.\n"
                "\n"
                "  To download Sentinel-2 imagery you need to:\n"
                "\n"
                "  1. Register a free account at:\n"
                "     https://dataspace.copernicus.eu/\n"
                "\n"
                "  2. In the dashboard, create OAuth2 client credentials\n"
                "     (S2PP application).\n"
                "\n"
                "  3. Copy .env.example to .env and fill in:\n"
                "     COPERNICUS_CLIENT_ID=<your-client-id>\n"
                "     COPERNICUS_CLIENT_SECRET=<your-client-secret>\n"
                "\n"
                "  API credentials are NEVER logged or printed.\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    def __repr__(self) -> str:
        """Safe repr — never exposes secrets."""
        return (
            f"Settings(env={self.app_env!r}, "
            f"copernicus={'configured' if self.has_copernicus_credentials else 'NOT configured'}, "
            f"data_dir={str(self.data_dir)!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Cached so environment is only read once per process.
    """
    return Settings()
