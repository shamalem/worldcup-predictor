"""Application settings, read from environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Use SQLite by default for zero-config local dev; override with a Postgres
    # URL (e.g. postgresql+psycopg://user:pass@db:5432/worldcup) in Docker.
    database_url: str = "sqlite:///./worldcup.db"

    # Comma-separated list of allowed CORS origins for the React dev server.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    app_name: str = "Explainable World Cup Predictor"
    api_version: str = "1.0.0"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
