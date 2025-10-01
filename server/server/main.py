from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mlflow.pyfunc import PyFuncModel
from databricks.sdk import WorkspaceClient
from server.schemas import DiamondRaw
from server.settings import settings
from apscheduler.schedulers.background import BackgroundScheduler
from server.utils import ping_self
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Authenticate databricks
    os.environ["DATABRICKS_HOST"] = settings.databricks_host
    os.environ["DATABRICKS_TOKEN"] = settings.databricks_token

    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri

    # Set tracking URI
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if mlflow.get_experiment_by_name(settings.mlflow_experiment_path) is None:
        mlflow.create_experiment(
            name=settings.mlflow_experiment_path,
            artifact_location=settings.mlflow_artifact_path,
        )

    mlflow.set_experiment(settings.mlflow_experiment_path)

    model_uri = f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_version}"

    app.state.ml_model = mlflow.pyfunc.load_model(model_uri)

    # Scheduler to keep app alive
    scheduler = BackgroundScheduler()
    scheduler.add_job(ping_self, "interval", minutes=5)  # every 10 min
    scheduler.start()

    print("Scheduler started")

    yield

    scheduler.shutdown()
    print("Scheduler stopped")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.client_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"message": "up"}


@app.post("/predict")
def predict(request: Request, diamond: DiamondRaw):
    data = diamond.model_dump()  # converts to dict
    df = pd.DataFrame([data])

    # cleaner = DiamondFeatureEngineer()
    # df = cleaner.transform(df)

    if df.shape[0] == 0:
        raise ValueError("No valid rows left after cleaning. Check input data.")

    model: PyFuncModel = request.app.state.ml_model

    preds = model.predict(df)

    print(preds)

    # Now you can clean, feature engineer, and pass to MLflow model
    return {
        "message": "validated",
        "data": data,
        "df": df.to_json(),
        "preds": float(preds[0]),
    }
