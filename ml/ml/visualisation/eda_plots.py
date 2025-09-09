from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.config import (
    CATEGORICAL_COLUMNS,
    CATEGORY_ORDERS,
    FIGURES_DIR,
    INTERIM_CLEANED_DATA_FILE,
    NUMERICAL_COLUMNS,
    PROCESSED_FEATURES_DATA_FILE,
    TARGET_COLUMN,
    logger,
)


def plot_numerical(df, numerical_columns, target_column, figures_dir):
    logger.info("Plotting histograms for numerical features")
    ax = df[numerical_columns].hist(bins=30, figsize=(15, 10))
    plt.suptitle("Numerical Features")
    plt.savefig(FIGURES_DIR / "eda_numerical_histograms.png", bbox_inches="tight")
    plt.close()

    logger.info("Plotting histograms for target column")
    plt.figure(figsize=(8, 5))
    plt.hist(df[target_column], bins=30, edgecolor="black")
    plt.title("Price Distribution")
    plt.savefig(figures_dir / "eda_target_price_distribution.png", bbox_inches="tight")
    plt.close()

    logger.info("Plotting scatterplots with price")
    for col in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=col, y=target_column, data=df, alpha=0.5)
        plt.title(f"{col} vs {target_column}")
        plt.savefig(
            figures_dir / f"eda_numerical_scatter_{col}_vs_{target_column}.png",
            bbox_inches="tight",
        )
        plt.close()


def plot_correlations(df, numerical_columns, target_column, figures_dir):
    logger.info("Plotting correlation heatmaps")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_columns + [target_column]].corr(), annot=True, cmap="coolwarm")
    plt.title("Pearson Correlation")
    plt.savefig(figures_dir / "correlation_pearson.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[numerical_columns + [target_column]].corr(method="spearman"),
        annot=True,
        cmap="coolwarm",
    )
    plt.title("Spearman Correlation")
    plt.savefig(figures_dir / "correlation_spearman.png", bbox_inches="tight")
    plt.close()


def plot_categorical(df, categorical_columns, category_orders, target_column, figures_dir):
    logger.info("Plotting categorical features vs price")
    for col in categorical_columns:
        order = sorted(category_orders[col], key=category_orders[col].get)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.countplot(x=col, data=df, order=order, ax=axes[0])
        axes[0].set_title(f"{col} Count")

        sns.boxplot(x=col, y=target_column, data=df, order=order, ax=axes[1])
        axes[1].set_title(f"{col} vs Price (Boxplot)")

        sns.barplot(
            x=col,
            y=target_column,
            data=df,
            order=order,
            estimator="mean",
            errorbar="sd",
            ax=axes[2],
        )
        axes[2].set_title(f"Mean Price by {col}")

        plt.tight_layout()
        plt.savefig(figures_dir / f"eda_categorical_{col}.png", bbox_inches="tight")
        plt.close()


def main():
    df = pd.read_csv(PROCESSED_FEATURES_DATA_FILE)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_numerical(df, NUMERICAL_COLUMNS, TARGET_COLUMN, FIGURES_DIR)
    plot_correlations(df, NUMERICAL_COLUMNS, TARGET_COLUMN, FIGURES_DIR)
    plot_categorical(df, CATEGORICAL_COLUMNS, CATEGORY_ORDERS, TARGET_COLUMN, FIGURES_DIR)

    logger.success(f"All EDA plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
