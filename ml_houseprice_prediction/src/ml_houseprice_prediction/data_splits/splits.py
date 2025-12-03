"""
data_splits.py

Splits a cleaned dataset into train and test sets and saves them to disk.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import sys
import argparse
import logging

from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
#  Global Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASTORE_DIR = PROJECT_ROOT / "datastores"

OUTPUT_DIR = DATASTORE_DIR / "splits_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATASTORE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
#  Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "data_splits.log", mode="a"),
    ],
)
logger = logging.getLogger("data_splits")

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")


# -------------------------------------------------------------------
#  Function: Load Data
# -------------------------------------------------------------------
def load_data(input_data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        input_data_path (Union[str, Path]): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger.info("Loading cleaned dataset for splitting...")
    try:
        input_path = Path(input_data_path)
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


# -------------------------------------------------------------------
#  Function: Split Dataset
# -------------------------------------------------------------------
def splits_data(df: pd.DataFrame):
    """
    Split dataset into train and test sets.

    Args:
        df (pd.DataFrame): The clean dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and test DataFrames.
    """
    logger.info("Splitting dataset into train and test...")

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )
    logger.info(f"Train shape: {df_train.shape} | Test shape: {df_test.shape}")

    return df_train, df_test


# -------------------------------------------------------------------
#  Function: Save Output Files
# -------------------------------------------------------------------
def save_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Saves the split datasets as CSV files in the standardized data directory.

    Args:
        df_train (pd.DataFrame): Train dataset to save.
        df_test (pd.DataFrame): Test dataset to save.
    """

    logger.info("Saving split datasets to disk...")

    file_paths = {
        "train_data.csv": df_train,
        "test_data.csv": df_test,
    }
    for filename, df in file_paths.items():
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved split data: {filename} into datastores at {output_path}.")


# -------------------------------------------------------------------
#  CLI Interface
# -------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments for the splitting step.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Split cleaned dataset into train and test sets"
    )

    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to the cleaned input CSV file (e.g., ../datastores/clean_data/clean_housing.csv).",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
#  Main Entry Point
# -------------------------------------------------------------------
def main() -> None:
    """
    Main function for CLI execution.
    Loads cleaned data, splits it, and saves train/test CSVs.
    """
    args = parse_arguments()

    df_clean = load_data(args.input_data_path)
    df_train, df_test = splits_data(df_clean)
    save_data(df_train, df_test)


if __name__ == "__main__":
    main()



