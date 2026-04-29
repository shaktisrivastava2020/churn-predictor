"""
Configuration module for Churn Predictor.
Loads environment variables and provides typed settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env and environment."""

    # GCP / Cloud SQL (reusing Day 1 infrastructure)
    PROJECT_ID: str = "project-c5185bee-a238-4d53-b9b"
    REGION: str = "asia-south1"
    INSTANCE_NAME: str = "quickshop-db"
    INSTANCE_CONNECTION_NAME: str = (
        "project-c5185bee-a238-4d53-b9b:asia-south1:quickshop-db"
    )
    DB_NAME: str = "quickshop"
    DB_USER: str = "postgres"
    DB_PASSWORD: str  # loaded from .env

    # API
    API_TITLE: str = "Churn Predictor API"
    API_VERSION: str = "0.1.0"

    # Model
    MODEL_PATH: str = "models/churn_model.pth"
    CHURN_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
