"""
Data loading functionality for the Spam Email Detection project.

This module provides functions to load and preprocess the spam email dataset.
"""

import pandas as pd
from typing import Optional, Union, Tuple
from pathlib import Path
import os

from src.utils.config import DATA_FILE
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the spam dataset from a CSV file.

    Args:
        filepath (str or Path, optional): Path to the CSV file.
            If None, uses the default path from config.
            Defaults to None.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if filepath is None:
        filepath = DATA_FILE

    # Convert to Path object
    filepath = Path(filepath)

    logger.info(f"Loading data from {filepath}")

    try:
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found at path: {filepath}")

        # Try different encodings if latin-1 fails
        try:
            df = pd.read_csv(filepath, encoding="latin-1")
        except UnicodeDecodeError:
            logger.warning("latin-1 encoding failed, trying utf-8")
            df = pd.read_csv(filepath, encoding="utf-8")

        logger.info(
            f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the spam dataset by cleaning and transforming the data.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logger.info("Preprocessing data")

    try:
        # Make a copy to avoid modifying the original
        df_processed = df.copy()

        # Drop unnecessary columns if they exist
        unnecessary_columns = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
        existing_unnecessary = [
            col for col in unnecessary_columns if col in df_processed.columns
        ]

        if existing_unnecessary:
            df_processed = df_processed.drop(columns=existing_unnecessary)
            logger.info(f"Dropped unnecessary columns: {existing_unnecessary}")

        # Rename columns for clarity if they have cryptic names
        if "v1" in df_processed.columns and "v2" in df_processed.columns:
            df_processed = df_processed.rename(columns={"v1": "label", "v2": "message"})
            logger.info("Renamed columns: v1 -> label, v2 -> message")

        # Encode 'spam' as 1 and 'ham' as 0 if label column contains strings
        if "label" in df_processed.columns and df_processed["label"].dtype == "object":
            df_processed["label"] = df_processed["label"].map({"spam": 1, "ham": 0})
            logger.info("Encoded 'spam' as 1 and 'ham' as 0")

        # Check for missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(
                f"Found missing values in the dataset: {missing_values[missing_values > 0]}"
            )

            # Drop rows with missing values in important columns
            important_columns = (
                ["label", "message"]
                if "label" in df_processed.columns
                else df_processed.columns
            )
            df_processed = df_processed.dropna(subset=important_columns)
            logger.info(
                f"Dropped rows with missing values in columns: {important_columns}"
            )

        logger.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


def load_and_preprocess(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load and preprocess the spam dataset in one step.

    Args:
        filepath (str or Path, optional): Path to the CSV file.
            If None, uses the default path from config.
            Defaults to None.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = load_data(filepath)
    return preprocess_data(df)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Split the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
        Tuple[pd.Series, pd.Series]: The features and target series.
    """
    logger.info("Splitting features and target")

    try:
        # Check if the DataFrame has the expected columns
        if "message" in df.columns and "label" in df.columns:
            X = df["message"]
            y = df["label"]
        else:
            # Assume the first column is the target and the second is the features
            logger.warning(
                "Expected column names 'message' and 'label' not found. Using default column positions."
            )
            y = df.iloc[:, 0]
            X = df.iloc[:, 1]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error during feature-target split: {str(e)}")
        raise
