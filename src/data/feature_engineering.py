"""
Feature engineering functionality for the Spam Email Detection project.

This module provides functions for text preprocessing, vectorization, and feature generation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.config import (
    VECTORIZER_MAX_FEATURES,
    VECTORIZER_MIN_DF,
    VECTORIZER_STOP_WORDS,
    RANDOM_STATE,
    TEST_SIZE,
    UPSAMPLING,
)
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading NLTK resources")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.

    Args:
        text (str): The raw text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text (str): The cleaned text.

    Returns:
        List[str]: List of tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Tokens with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply stemming to tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Stemmed tokens.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: Lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(
    text: str,
    remove_stop: bool = True,
    stemming: bool = False,
    lemmatization: bool = True,
) -> str:
    """
    Apply full text preprocessing pipeline.

    Args:
        text (str): The raw text.
        remove_stop (bool, optional): Whether to remove stopwords.
            Defaults to True.
        stemming (bool, optional): Whether to apply stemming.
            Defaults to False.
        lemmatization (bool, optional): Whether to apply lemmatization.
            Defaults to True.

    Returns:
        str: The preprocessed text.
    """
    # Clean text
    text = clean_text(text)

    # Tokenize
    tokens = tokenize_text(text)

    # Remove stopwords
    if remove_stop:
        tokens = remove_stopwords(tokens)

    # Apply stemming or lemmatization
    if stemming:
        tokens = stem_tokens(tokens)
    elif lemmatization:
        tokens = lemmatize_tokens(tokens)

    # Join tokens back into text
    return " ".join(tokens)


def vectorize_text(
    texts: Union[List[str], pd.Series],
    method: str = "tfidf",
    max_features: int = VECTORIZER_MAX_FEATURES,
    min_df: int = VECTORIZER_MIN_DF,
    stop_words: Optional[str] = VECTORIZER_STOP_WORDS,
) -> Tuple[np.ndarray, Any]:
    """
    Convert text data into numerical vectors.

    Args:
        texts (List[str] or pd.Series): The preprocessed text data.
        method (str, optional): Vectorization method ('tfidf' or 'count').
            Defaults to 'tfidf'.
        max_features (int, optional): Maximum number of features.
            Defaults to value from config.
        min_df (int, optional): Minimum document frequency.
            Defaults to value from config.
        stop_words (str, optional): Stop words strategy.
            Defaults to value from config.

    Returns:
        Tuple[np.ndarray, Any]: The vectorized text and the vectorizer object.
    """
    logger.info(f"Vectorizing text using {method} method")

    try:
        if method.lower() == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=max_features, min_df=min_df, stop_words=stop_words
            )
        else:  # Use CountVectorizer by default
            vectorizer = CountVectorizer(
                max_features=max_features, min_df=min_df, stop_words=stop_words
            )

        X_vectorized = vectorizer.fit_transform(texts)
        logger.info(f"Text vectorized with shape: {X_vectorized.shape}")

        return X_vectorized, vectorizer

    except Exception as e:
        logger.error(f"Error during text vectorization: {str(e)}")
        raise


def balance_classes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution using upsampling.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Balanced features and target.
    """
    if not UPSAMPLING:
        logger.info("Skipping class balancing as it is disabled in config")
        return X, y

    logger.info("Balancing class distribution")

    try:
        # Get indices for each class
        negative_indices = np.where(y == 0)[0]
        positive_indices = np.where(y == 1)[0]

        # Count samples in each class
        n_negative = len(negative_indices)
        n_positive = len(positive_indices)

        logger.info(
            f"Class distribution before balancing - Negative: {n_negative}, Positive: {n_positive}"
        )

        # Identify majority and minority classes
        if n_negative > n_positive:
            majority_indices = negative_indices
            minority_indices = positive_indices
            majority_class = 0
            minority_class = 1
        else:
            majority_indices = positive_indices
            minority_indices = negative_indices
            majority_class = 1
            minority_class = 0

        # Perform upsampling of minority class
        minority_indices_upsampled = resample(
            minority_indices,
            replace=True,
            n_samples=len(majority_indices),
            random_state=RANDOM_STATE,
        )

        # Combine indices
        balanced_indices = np.concatenate(
            [majority_indices, minority_indices_upsampled]
        )

        # Create balanced dataset
        X_balanced = X[balanced_indices]
        y_balanced = np.array(
            [majority_class] * len(majority_indices)
            + [minority_class] * len(minority_indices_upsampled)
        )

        logger.info(
            f"Class distribution after balancing - Class {majority_class}: {len(majority_indices)}, Class {minority_class}: {len(minority_indices_upsampled)}"
        )

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Error during class balancing: {str(e)}")
        raise


def prepare_data_for_training(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training by splitting into train and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test.
    """
    logger.info("Splitting data into train and test sets")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(
            f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error during train-test split: {str(e)}")
        raise


def extract_text_features(texts: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Extract additional features from text data.

    Args:
        texts (List[str] or pd.Series): The raw text data.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    logger.info("Extracting additional text features")

    try:
        features = pd.DataFrame()

        # Text length
        features["text_length"] = [len(str(text)) for text in texts]

        # Word count
        features["word_count"] = [len(str(text).split()) for text in texts]

        # Character count per word
        features["char_per_word"] = features["text_length"] / (
            features["word_count"] + 1
        )  # Add 1 to avoid division by zero

        # Count special characters
        features["special_char_count"] = [
            len(re.findall(r"[^\w\s]", str(text))) for text in texts
        ]

        # Count uppercase characters
        features["uppercase_count"] = [
            sum(1 for c in str(text) if c.isupper()) for text in texts
        ]

        # Count digits
        features["digit_count"] = [
            sum(1 for c in str(text) if c.isdigit()) for text in texts
        ]

        # Count URLs
        features["url_count"] = [
            len(re.findall(r"https?://\S+|www\.\S+", str(text))) for text in texts
        ]

        logger.info(f"Extracted {features.shape[1]} additional features")

        return features

    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        raise
