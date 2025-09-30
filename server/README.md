## Server (FastAPI)

FastAPI service that loads a registered MLflow model at startup and exposes prediction endpoints.

### Prerequisites

-   Python 3.11
-   Poetry 2.x

### Install

```bash
poetry install
```

### Environment

Configure `.env` at the monorepo root (copy from `env.example`). Key variables consumed here:

-   `MLFLOW_TRACKING_URI`
-   `MLFLOW_EXPERIMENT_PATH`
-   `MLFLOW_ARTIFACT_PATH`
-   `MLFLOW_MODEL_NAME`
-   `MLFLOW_MODEL_VERSION`
-   `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (when using Databricks)
-   `CLIENT_URL` (CORS allow‑origin)

Settings are loaded via Pydantic Settings:

```19:25:/home/kraigochieng/projects/diamond-price-predictor/server/server/settings.py
settings = Settings()
```

On startup, the app authenticates with Databricks, configures MLflow, ensures the experiment exists, and loads the model:

```15:38:/home/kraigochieng/projects/diamond-price-predictor/server/server/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["DATABRICKS_HOST"] = settings.databricks_host
    os.environ["DATABRICKS_TOKEN"] = settings.databricks_token
    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(settings.mlflow_experiment_path) is None:
        mlflow.create_experiment(
            name=settings.mlflow_experiment_path,
            artifact_location=settings.mlflow_artifact_path,
        )
    mlflow.set_experiment(settings.mlflow_experiment_path)
    model_uri = f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_version}"
    app.state.ml_model = mlflow.pyfunc.load_model(model_uri)
```

### Run (dev)

```bash
poetry run fastapi dev server/main.py --port 8000
```

### Endpoints

-   `GET /` – health
-   `POST /predict` – returns prediction

Example request (current minimal schema):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"carat": 0.23}'
```

Response:

```json
{
	"message": "validated",
	"data": { "carat": 0.23 },
	"df": "...",
	"preds": 1234.56
}
```

Note: `server/schemas.py` contains enumerations for `cut`, `color`, `clarity`. As the model evolves, extend `DiamondRaw` accordingly and update the client contract.

### Deployment

-   Docker: use the monorepo `docker-compose.yml` to build and run the server container. Ensure `.env` is present with the MLflow and Databricks values.
-   CORS: set `CLIENT_URL` to the public client origin.

### Troubleshooting

-   Model load errors: verify `MLFLOW_MODEL_NAME` and `MLFLOW_MODEL_VERSION` exist in the registry and the token has permission.
-   CORS issues: check `CLIENT_URL` matches the requesting origin.
