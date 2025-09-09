from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import INTERIM_CLEANED_DATA_FILE, RAW_DATA_FILE, logger


def load_raw_data(input_filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {input_filepath}")
    df = pd.read_csv(input_filepath)
    logger.info(f"Raw data shape: {df.shape}")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data and creating derived features")
    # Derived features
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["aspect_ratio"] = df["x"] / df["y"]
    df["carat_per_volume"] = df["carat"] / df["volume"]
    df["depth_table_ratio"] = df["depth"] / df["table"]

    # Replace inf/-inf, drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"After dropping NaNs: {df.shape}")

    # Filters
    df = df[
        (df["x"] > 0)
        & (df["y"] > 0)
        & (df["z"] > 0)
        & (df["carat"] > 0)
        & (df["carat_per_volume"] > 0.006)
        & (df["price"] > 0)
    ]

    logger.info(f"After filtering invalid values: {df.shape}")

    df = df[
        df["depth"].between(50, 75)
        & df["table"].between(50, 75)
        & df["aspect_ratio"].between(0.96, 1.04)
        & df["depth_table_ratio"].between(0.8, 1.3)
    ]

    logger.info(f"After filtering ranges: {df.shape}")

    df = df[df["volume"] < df["volume"].quantile(0.99)]
    df = df[df["carat_per_volume"] < df["carat_per_volume"].quantile(0.99)]

    logger.info(f"After removing outliers: {df.shape}")

    return df


def main(input_filepath: str, output_filepath: str):
    logger.info("Starting make_dataset pipeline")

    df = load_raw_data(input_filepath)
    df = clean_data(df)
    df.to_csv(output_filepath, index=False)

    logger.success(f"Saved cleaned dataset to {output_filepath}")


if __name__ == "__main__":
    main(RAW_DATA_FILE, INTERIM_CLEANED_DATA_FILE)
