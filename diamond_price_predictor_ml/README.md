# Diamond Price Predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ML package for data preparation, feature engineering, model training, experiment tracking with MLflow, and model serving.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## Quickstart

### Prerequisites

-   Python 3.11
-   Poetry 2.x

### Install

```bash
poetry install
```

### Environment

This package can log to a remote MLflow tracking server (e.g., Databricks). Configure these in the monorepo `.env` (see `env.example`):

-   `MLFLOW_TRACKING_URI`
-   `MLFLOW_EXPERIMENT_PATH`
-   `MLFLOW_ARTIFACT_PATH`
-   (Optional) `DATABRICKS_HOST`, `DATABRICKS_TOKEN` if using Databricks

### Pipelines

Convenience `Makefile` targets:

```1:39:/home/kraigochieng/projects/diamond-price-predictor/diamond_price_predictor_ml/Makefile
PORT ?= 5001
MODEL_NAME ?= best_model
MODEL_VERSION ?= 1

.PHONY: all data features train figures clean-data clean-models clean-figures clean-all serve

# Default target
all: data features train figures

# === Pipelines ===
data:
    poetry run python ml/data/make_dataset.py

features: data
    poetry run python ml/features/build_features.py

train:
    poetry run python -m diamond_price_predictor_ml.modeling.train_model

figures: data features
    poetry run python ml/visualisation/eda_plots.py

# === Cleaning ===
clean-data:
    rm -rf data/interim/*.csv data/processed/*.csv

clean-models:
    rm -rf models/*.pkl mlruns/

clean-figures:
    rm -rf reports/figures/*.png

clean-all: clean-data clean-models clean-figures

serve:
    poetry run mlflow models serve \
        -m models:/$(MODEL_NAME)/$(MODEL_VERSION) \
        -p $(PORT) \
        --env-manager=local
```

Common flows:

-   Prepare data: `make data`
-   Build features: `make features`
-   Train and log to MLflow: `make train`
-   Generate EDA figures: `make figures`

### Serving a registered model locally

```bash
make serve PORT=5001 MODEL_NAME=best_model MODEL_VERSION=1
```

This relies on the model being in the MLflow Model Registry referenced by your `MLFLOW_TRACKING_URI`.

### Training entry point

The main training script is:

```1:200:/home/kraigochieng/projects/diamond-price-predictor/diamond_price_predictor_ml/diamond_price_predictor_ml/modeling/train_model.py
# refer to source for full details
```

### Documentation

MkDocs project is available under `docs/`:

```1:13:/home/kraigochieng/projects/diamond-price-predictor/diamond_price_predictor_ml/docs/README.md
Generating the docs
```

### Notes

-   Artifacts and runs are stored under `mlruns/` locally if not using a remote tracking server.
-   When promoted, the best model should be registered as `MLFLOW_MODEL_NAME` that the API consumes.
