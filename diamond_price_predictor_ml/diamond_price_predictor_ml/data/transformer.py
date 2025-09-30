import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from diamond_price_predictor_ml.config import logger


class DiamondDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for cleaning diamond dataset and creating derived features.
    """

    def __init__(self, drop_columns: list[str] = None):
        self.drop_columns = drop_columns if drop_columns else ["Unnamed: 0"]

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to fit for this transformer
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()
        logger.info("Starting data cleaning in transformer")

        # Drop columns and duplicates
        df = df.drop(columns=self.drop_columns, errors="ignore").drop_duplicates()

        # Derived features
        # df["volume"] = df["x"] * df["y"] * df["z"]
        # df["aspect_ratio"] = df["x"] / df["y"]
        # df["carat_per_volume"] = df["carat"] / df["volume"]
        # df["depth_table_ratio"] = df["depth"] / df["table"]

        # Replace inf/-inf and drop NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"After dropping NaNs: {df.shape}")

        # Filters
        # df = df[
        #     (df["x"] > 0)
        #     & (df["y"] > 0)
        #     & (df["z"] > 0)
        #     & (df["carat"] > 0)
        #     # & (df["carat_per_volume"] > 0.006)
        # ]

        df = df[df["carat"] > 0]

        # df = df[
        #     df["depth"].between(50, 75)
        #     & df["table"].between(50, 75)
        #     & df["aspect_ratio"].between(0.96, 1.04)
        #     & df["depth_table_ratio"].between(0.8, 1.3)
        # ]

        # # Remove outliers (top 1%)
        # df = df[df["volume"] < df["volume"].quantile(0.99)]
        # df = df[df["carat_per_volume"] < df["carat_per_volume"].quantile(0.99)]

        logger.info(f"After cleaning and feature engineering: {df.shape}")

        return df
