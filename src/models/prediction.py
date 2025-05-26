"""
Prediction functionality for the Spam Email Detection project.

This module provides functions for making predictions with trained models.
"""

from typing import Any, Union, List, Dict, Optional
import numpy as np
import pandas as pd

from src.data.feature_engineering import preprocess_text, vectorize_text
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def predict_single_email(
    model: Any, vectorizer: Any, text: str, preprocess: bool = True
) -> Dict[str, Any]:
    """
    Predict whether a single email is spam or ham.

    Args:
        model (Any): Trained model.
        vectorizer (Any): Fitted vectorizer.
        text (str): Email text.
        preprocess (bool, optional): Whether to preprocess the text.
            Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary with prediction results.
    """
    logger.info("Making prediction for single email")

    try:
        # Preprocess text if required
        if preprocess:
            processed_text = preprocess_text(text)
            logger.debug(f"Preprocessed text: {processed_text[:100]}...")
        else:
            processed_text = text

        # Vectorize text
        X = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(X)[0]

        # Get probability if available
        try:
            probability = model.predict_proba(X)[
                0, 1
            ]  # Probability of positive class (spam)
            has_proba = True
        except (AttributeError, IndexError):
            logger.warning("Model does not support probability predictions")
            probability = None
            has_proba = False

        # Create result dictionary
        result = {
            "text": text,
            "preprocessed_text": processed_text,
            "prediction": int(prediction),
            "label": "spam" if prediction == 1 else "ham",
        }

        if has_proba:
            result["probability"] = float(probability)
            result["confidence"] = float(max(probability, 1 - probability))

        logger.info(
            f"Prediction result: {result['label']} (confidence: {result.get('confidence', 'N/A')})"
        )

        return result

    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        raise


def predict_batch(
    model: Any, vectorizer: Any, texts: List[str], preprocess: bool = True
) -> List[Dict[str, Any]]:
    """
    Predict whether multiple emails are spam or ham.

    Args:
        model (Any): Trained model.
        vectorizer (Any): Fitted vectorizer.
        texts (List[str]): List of email texts.
        preprocess (bool, optional): Whether to preprocess the texts.
            Defaults to True.

    Returns:
        List[Dict[str, Any]]: List of dictionaries with prediction results.
    """
    logger.info(f"Making batch predictions for {len(texts)} emails")

    try:
        results = []

        # Process each text
        processed_texts = []
        for text in texts:
            if preprocess:
                processed_text = preprocess_text(text)
            else:
                processed_text = text
            processed_texts.append(processed_text)

        # Vectorize texts
        X = vectorizer.transform(processed_texts)

        # Make predictions
        predictions = model.predict(X)

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(X)[
                :, 1
            ]  # Probability of positive class (spam)
            has_proba = True
        except (AttributeError, IndexError):
            logger.warning("Model does not support probability predictions")
            probabilities = [None] * len(texts)
            has_proba = False

        # Create result dictionaries
        for i, (text, processed_text, prediction) in enumerate(
            zip(texts, processed_texts, predictions)
        ):
            result = {
                "text": text,
                "preprocessed_text": processed_text,
                "prediction": int(prediction),
                "label": "spam" if prediction == 1 else "ham",
            }

            if has_proba:
                probability = float(probabilities[i])
                result["probability"] = probability
                result["confidence"] = float(max(probability, 1 - probability))

            results.append(result)

        # Count predictions
        spam_count = sum(1 for result in results if result["prediction"] == 1)
        ham_count = len(results) - spam_count

        logger.info(f"Batch prediction results: {spam_count} spam, {ham_count} ham")

        return results

    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise


def extract_top_features(
    vectorizer: Any, model: Any, n_features: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract top features for spam and ham classification.

    Args:
        vectorizer (Any): Fitted vectorizer.
        model (Any): Trained model with feature importance or coefficients.
        n_features (int, optional): Number of top features to extract.
            Defaults to 20.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with top features for spam and ham.

    Raises:
        ValueError: If model does not have feature importance or coefficients.
    """
    logger.info(f"Extracting top {n_features} features for spam and ham")

    try:
        # Get feature names from vectorizer
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            try:
                feature_names = vectorizer.get_feature_names()
            except AttributeError:
                logger.error("Vectorizer does not have get_feature_names method")
                raise ValueError("Vectorizer does not have get_feature_names method")

        # Get feature importance or coefficients from model
        if hasattr(model, "feature_importances_"):
            # For tree-based models
            feature_importances = model.feature_importances_

            # Sort features by importance
            indices = np.argsort(feature_importances)[::-1]

            # Extract top features
            top_features = []
            for i in range(min(n_features, len(indices))):
                idx = indices[i]
                top_features.append(
                    {
                        "feature": feature_names[idx],
                        "importance": float(feature_importances[idx]),
                        "rank": i + 1,
                    }
                )

            logger.info(
                f"Extracted top {len(top_features)} features based on importance"
            )

            return {"top_features": top_features}

        elif hasattr(model, "coef_"):
            # For linear models
            coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_

            # Get indices of top positive (spam) and negative (ham) coefficients
            top_positive_indices = np.argsort(coefficients)[::-1][:n_features]
            top_negative_indices = np.argsort(coefficients)[:n_features]

            # Extract top spam features
            top_spam_features = []
            for i, idx in enumerate(top_positive_indices):
                top_spam_features.append(
                    {
                        "feature": feature_names[idx],
                        "coefficient": float(coefficients[idx]),
                        "rank": i + 1,
                    }
                )

            # Extract top ham features
            top_ham_features = []
            for i, idx in enumerate(top_negative_indices):
                top_ham_features.append(
                    {
                        "feature": feature_names[idx],
                        "coefficient": float(coefficients[idx]),
                        "rank": i + 1,
                    }
                )

            logger.info(
                f"Extracted top {len(top_spam_features)} features for spam and {len(top_ham_features)} for ham"
            )

            return {
                "spam_features": top_spam_features,
                "ham_features": top_ham_features,
            }

        else:
            error_msg = "Model does not have feature_importances_ or coef_ attribute"
            logger.error(error_msg)
            raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"Error extracting top features: {str(e)}")
        raise


def create_predictor(model: Any, vectorizer: Any) -> Dict[str, Any]:
    """
    Create a predictor object that encapsulates model and vectorizer.

    Args:
        model (Any): Trained model.
        vectorizer (Any): Fitted vectorizer.

    Returns:
        Dict[str, Any]: Predictor object.
    """
    logger.info("Creating predictor object")

    predictor = {
        "model": model,
        "vectorizer": vectorizer,
        "predict": lambda text, preprocess=True: predict_single_email(
            model, vectorizer, text, preprocess
        ),
        "predict_batch": lambda texts, preprocess=True: predict_batch(
            model, vectorizer, texts, preprocess
        ),
    }

    # Add feature extraction if model supports it
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        predictor["extract_top_features"] = lambda n_features=20: extract_top_features(
            vectorizer, model, n_features
        )

    return predictor
