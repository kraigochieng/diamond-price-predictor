from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from diamond_price_predictor_ml.config import logger


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features using a provided mapping."""

    def __init__(self, category_orders: Dict[str, Dict] = None):
        self.category_orders = category_orders

    def fit(self, X: pd.DataFrame, y=None):
        # No fitting required
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        logger.info("Encoding categorical features")

        if self.category_orders:
            for col, mapping in self.category_orders.items():
                df[col + "_encoded"] = df[col].map(mapping)

        logger.info("Finished encoding categorical features")
        return df


class DropMulticollinear(BaseEstimator, TransformerMixin):
    """Drop predefined highly collinear features."""

    def __init__(self, columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = ["x", "y", "z", "volume"]
            # columns_to_drop = ["x", "y", "z"]
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        df = df.drop(columns=self.columns_to_drop, errors="ignore")
        logger.info(f"Dropped multicollinear features: {self.columns_to_drop}")
        return df


class DropLowValueFeatures(BaseEstimator, TransformerMixin):
    """
    Drops columns identified as low-value for prediction.
    """

    def __init__(self, columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = ["cut", "color", "clarity"]
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        df = df.drop(columns=self.columns_to_drop, errors="ignore")
        logger.info(f"Dropped low-value features: {self.columns_to_drop}")
        return df


class ToFloat64(BaseEstimator, TransformerMixin):
    """Convert all numeric columns to float64 to prevent MLflow schema issues."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=["int", "float"]).columns
        df[numeric_cols] = df[numeric_cols].astype("float64")
        return df


class DiamondFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Row-preserving feature engineer for diamonds dataset.
    Computes derived features without dropping rows.
    Safe to use at inference.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to learn, just return self
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        logger.info("Applying row-preserving feature engineering")

        # Derived features
        df["volume"] = df["x"] * df["y"] * df["z"]
        df["aspect_ratio"] = df["x"] / df["y"]
        df["carat_per_volume"] = df["carat"] / df["volume"]
        df["depth_table_ratio"] = df["depth"] / df["table"]

        return df


class CaratOnly(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        logger.info("Using column carats only since it is the only valuable feature")

        # Double brackets since it needs to be a dataframe
        df = df[["carat"]]

        return df
