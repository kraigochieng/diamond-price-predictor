import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml.config import MODELS_DIR, PROCESSED_FEATURES_REDUCED_DATA_FILE, logger


def train_model(df_path: str, model_out: str):
    logger.info("Starting model training with RandomizedSearchCV")

    df = pd.read_csv(df_path)
    logger.info(f"Dataset shape: {df.shape}")

    X = df.drop(columns=["price", "cut", "color", "clarity"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # === Define pipeline: Polynomial expansion + Ridge regression ===
    pipeline = Pipeline(
        [
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
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Best Params: {search.best_params_}")
        logger.info(f"R²={r2:.4f}, MSE={mse:.2f}")

        # Log best params & metrics
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mse", mse)

        # Ensure input_example is float64
        input_example = X_test.iloc[:1].astype("float64")

        mlflow.sklearn.log_model(
            best_model,
            name="ridge_model_random_search",
            input_example=input_example,
            signature=infer_signature(X_test.astype("float64"), y_pred.astype("float64")),
        )

        logger.success("Model training completed with RandomizedSearchCV and logged to MLflow")


if __name__ == "__main__":
    train_model(PROCESSED_FEATURES_REDUCED_DATA_FILE, f"{MODELS_DIR}/ridge_model.pkl")
