from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI

from ml.features.transformer import DiamondFeatureEngineer
from server.schemas import DiamondRaw
from server.settings import settings
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    yield


app = FastAPI()

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
def predict(diamond: DiamondRaw):
    data = diamond.model_dump()  # converts to dict
    df = pd.DataFrame([data])

    # cleaner = DiamondFeatureEngineer()
    # df = cleaner.transform(df)

    if df.shape[0] == 0:
        raise ValueError("No valid rows left after cleaning. Check input data.")

    model_uri = "models:/best_model/6"

    model = mlflow.pyfunc.load_model(model_uri)

    preds = model.predict(df)

    print(preds)

    # Now you can clean, feature engineer, and pass to MLflow model
    return {
        "message": "validated",
        "data": data,
        "df": df.to_json(),
        "preds": float(preds[0]),
    }
