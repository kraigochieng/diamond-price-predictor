from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mlflow_tracking_uri: str
    mlflow_tracking_host: str
    mlflow_tracking_port: str
    
    mlflow_experiment_name: str

    client_url: str

    model_config = SettingsConfigDict(
        extra="allow", env_file=Path(__file__).resolve().parents[3] / ".env"
    )


settings = Settings()
