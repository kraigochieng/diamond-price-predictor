import pandas as pd

from ml.config import (
    CATEGORY_ORDERS,
    INTERIM_CLEANED_DATA_FILE,
    PROCESSED_FEATURES_DATA_FILE,
    PROCESSED_FEATURES_REDUCED_DATA_FILE,
    logger,
)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features")

    for col, mapping in CATEGORY_ORDERS.items():
        df[col + "_encoded"] = df[col].map(mapping)
    # df = df.drop(columns=["x", "y", "z", "volume"])  # drop multicollinear

    logger.info("Dropped multicollinear features")

    return df


def drop_multicollinear(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified as highly collinear."""
    high_multicollinearity_columns_to_drop = ["x", "y", "z", "volume"]
    df = df.drop(columns=high_multicollinearity_columns_to_drop, errors="ignore")
    logger.info("Dropped multicollinear features: %s", high_multicollinearity_columns_to_drop)
    return df


def main(input_filepath: str, output_encoded: str, output_reduced: str):
    logger.info("Starting build_features pipeline")

    df = pd.read_csv(input_filepath)
    logger.info(f"Loaded dataset: {df.shape}")

    # Encode features
    df_encoded = encode_features(df)
    df_encoded.to_csv(output_encoded, index=False)
    logger.success(f"Saved encoded dataset to {output_encoded}")

    # Drop multicollinear features
    df_reduced = drop_multicollinear(df_encoded.copy())
    df_reduced.to_csv(output_reduced, index=False)
    logger.success(f"Saved reduced dataset to {output_reduced}")


if __name__ == "__main__":
    main(
        INTERIM_CLEANED_DATA_FILE,
        PROCESSED_FEATURES_DATA_FILE,  # encoded-only
        PROCESSED_FEATURES_REDUCED_DATA_FILE,  # encoded + no multicollinearity
    )
