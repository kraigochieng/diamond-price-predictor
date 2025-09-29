from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mlflow.pyfunc import PyFuncModel

from server.schemas import DiamondRaw
from server.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set tracking URI
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_version}"

    app.state.ml_model = mlflow.pyfunc.load_model(model_uri)
    yield


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
