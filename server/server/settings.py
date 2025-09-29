from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mlflow_experiment_name: str
    mlflow_experiment_path: str
    mlflow_artifact_path: str
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_model_version: str

    databricks_host: str
    databricks_token: str

    client_url: str

    server_url: str

    model_config = SettingsConfigDict(
        extra="allow", env_file=Path(__file__).resolve().parent.parent.parent / ".env"
    )


settings = Settings()
