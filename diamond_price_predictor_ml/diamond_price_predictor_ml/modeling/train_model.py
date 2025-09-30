import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from diamond_price_predictor_ml.config import (
    CATEGORY_ORDERS,
    MODELS_DIR,
    PROCESSED_FEATURES_REDUCED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    logger,
    settings,
)
from diamond_price_predictor_ml.data.transformer import DiamondDataCleaner
from diamond_price_predictor_ml.features.transformer import (
    CaratOnly,
    DropLowValueFeatures,
    DropMulticollinear,
    FeatureEncoder,
    ToFloat64,
)

# Authenticate Databricks
# w = WorkspaceClient(host=settings.databricks_host, token=settings.databricks_token)
os.environ["DATABRICKS_HOST"] = settings.databricks_host
os.environ["DATABRICKS_TOKEN"] = settings.databricks_token
# Set MLFlow Tracking URI
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# Create MLFlow Experiment if not created
if mlflow.get_experiment_by_name(settings.mlflow_experiment_path) is None:
    mlflow.create_experiment(
        name=settings.mlflow_experiment_path,
        artifact_location=settings.mlflow_artifact_path,
    )

# Set MLFlow Experiment if not created
mlflow.set_experiment(settings.mlflow_experiment_path)


def train_model(df_path: str):
    mlflow.autolog()
    logger.info("Enabled autolog")

    df = pd.read_csv(df_path)
    logger.info(f"Dataset shape: {df.shape}")

    df = df[df["price"] > 0]
    logger.info(f"Dataset shape after removing price of 0: {df.shape}")

    cleaner = DiamondDataCleaner()
    df = cleaner.transform(df)

    logger.info(f"Dataset shape after cleaning dataset(row removal): {df.shape}")

    # Carat is the only valuable column
    X = df[["carat"]]
    y = df[TARGET_COLUMN].astype("float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # === Define pipeline: Polynomial expansion + Ridge regression ===
    # pipeline = Pipeline(
    #     [
    #         ("encode_categoricals", FeatureEncoder(category_orders=CATEGORY_ORDERS)),
    #         ("drop_multicollinear", DropMulticollinear()),
    #         ("drop_low_value", DropLowValueFeatures()),
    #         ("to_float", ToFloat64()),
    #         ("poly", PolynomialFeatures(include_bias=False)),
    #         # ("scaler", StandardScaler()),
    #         ("ridge", Ridge()),
    #     ]
    # )

    pipeline = Pipeline(
        [
            ("to_float", ToFloat64()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )

    # === Hyperparameter space ===
    param_dist = {
        "poly__degree": [3],
        "ridge__alpha": np.logspace(-3, 3, 20),  # 0.001 → 1000
    }

    # === Randomized Search ===
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,  # number of random configs
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )

    with mlflow.start_run():
        logger.info("Starting model training with RandomizedSearchCV")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Best Params: {search.best_params_}")
        logger.info(f"R²={r2:.4f}, MSE={mse:.2f}")

        input_example = X_test.iloc[:1]
        mlflow.sklearn.log_model(
            best_model,
            name="ridge_model_random_search",
            input_example=input_example,
            signature=infer_signature(X_test, y_pred),
        )

        logger.success(
            "Model training completed with RandomizedSearchCV and logged to MLflow"
        )


if __name__ == "__main__":
    train_model(RAW_DATA_FILE)
