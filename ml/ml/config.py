from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "diamonds.csv"
INTERIM_CLEANED_DATA_FILE = INTERIM_DATA_DIR / "diamonds_cleaned.csv"
PROCESSED_FEATURES_DATA_FILE = PROCESSED_DATA_DIR / "diamonds_features.csv"
PROCESSED_FEATURES_REDUCED_DATA_FILE = PROCESSED_DATA_DIR / "diamonds_features_reduced.csv"
# Column
CATEGORICAL_COLUMNS = ["cut", "color", "clarity"]
NUMERICAL_COLUMNS = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "volume",
    "aspect_ratio",
    "carat_per_volume",
    "depth_table_ratio",
    "cut_encoded",
    "color_encoded",
    "clarity_encoded",
]
TARGET_COLUMN = "price"

# Encoding
CATEGORY_ORDERS = {
    "clarity": {
        "I1": 1,
        "SI2": 2,
        "SI1": 3,
        "VS2": 4,
        "VS1": 5,
        "VVS2": 6,
        "VVS1": 7,
        "IF": 8,
    },
    "cut": {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5},
    "color": {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7},
}


logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="INFO",
    colorize=True,
    enqueue=True,
    backtrace=True,
    diagnose=True,
)