"""
Model training and evaluation functionality for the Spam Email Detection project.

This module provides functions for training, evaluating, and comparing models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import time
import pickle
import os
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)

from src.utils.logger import setup_logger
from src.utils.config import RANDOM_STATE, CV_FOLDS, MODELS_DIR

# Set up logger
logger = setup_logger(__name__)


def train_model(
    model: Any, X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, float]:
    """
    Train a model on the given data.

    Args:
        model (Any): The model to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.

    Returns:
        Tuple[Any, float]: The trained model and training time in seconds.
    """
    logger.info(f"Training model: {type(model).__name__}")

    # Measure training time
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Calculate training time
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    return model, training_time


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model (Any): The trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating model: {type(model).__name__}")

    # Make predictions
    try:
        y_pred = model.predict(X_test)

        # Get probability predictions if available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            has_proba = True
        except (AttributeError, IndexError):
            logger.warning("Model does not support probability predictions")
            y_prob = None
            has_proba = False

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        # Add metrics that require probability predictions
        if has_proba:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            metrics["average_precision"] = average_precision_score(y_test, y_prob)

            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            # Precision-recall curve data
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            metrics["pr_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
            }

        logger.info(
            f"Evaluation results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )

        return metrics

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


def cross_validate_model(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS, scoring: str = "f1"
) -> Dict[str, Any]:
    """
    Perform cross-validation for a model.

    Args:
        model (Any): The model to validate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        cv (int, optional): Number of cross-validation folds.
            Defaults to value from config.
        scoring (str, optional): Scoring metric to use.
            Defaults to 'f1'.

    Returns:
        Dict[str, Any]: Dictionary of cross-validation results.
    """
    logger.info(
        f"Performing {cv}-fold cross-validation for model: {type(model).__name__}"
    )

    try:
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        # Calculate statistics
        results = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "scores": scores.tolist(),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
        }

        logger.info(
            f"Cross-validation results: Mean {scoring}={results['mean_score']:.4f} (±{results['std_score']:.4f})"
        )

        return results

    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise


def tune_model_hyperparameters(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int = CV_FOLDS,
    scoring: str = "f1",
) -> Dict[str, Any]:
    """
    Tune model hyperparameters using grid search.

    Args:
        model (Any): The model to tune.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        param_grid (Dict[str, List[Any]]): Parameter grid for search.
        cv (int, optional): Number of cross-validation folds.
            Defaults to value from config.
        scoring (str, optional): Scoring metric to use.
            Defaults to 'f1'.

    Returns:
        Dict[str, Any]: Dictionary with best parameters and results.
    """
    logger.info(f"Tuning hyperparameters for model: {type(model).__name__}")

    try:
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )

        # Fit grid search
        grid_search.fit(X, y)

        # Get results
        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": pd.DataFrame(grid_search.cv_results_).to_dict("records"),
        }

        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best {scoring} score: {results['best_score']:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise


def train_and_evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple models.

    Args:
        models (Dict[str, Any]): Dictionary of models to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of model results.
    """
    logger.info(f"Training and evaluating {len(models)} models")

    results = {}

    for model_name, model in models.items():
        logger.info(f"Processing model: {model_name}")

        try:
            # Train the model
            trained_model, training_time = train_model(model, X_train, y_train)

            # Evaluate the model
            evaluation_metrics = evaluate_model(trained_model, X_test, y_test)

            # Store results
            results[model_name] = {
                "model": trained_model,
                "training_time": training_time,
                "evaluation": evaluation_metrics,
            }

            logger.info(f"Completed training and evaluation for {model_name}")

        except Exception as e:
            logger.error(f"Error processing model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}

    logger.info(f"Completed training and evaluation for all models")

    return results


def save_model(
    model: Any, model_name: str, model_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Save a trained model to disk.

    Args:
        model (Any): The model to save.
        model_name (str): Name of the model.
        model_dir (str or Path, optional): Directory to save the model in.
            If None, uses the default directory from config.
            Defaults to None.

    Returns:
        str: Path to the saved model file.
    """
    if model_dir is None:
        model_dir = MODELS_DIR

    # Convert to Path object
    model_dir = Path(model_dir)

    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Create a valid filename
    valid_model_name = model_name.replace(" ", "_").lower()
    model_path = model_dir / f"{valid_model_name}.pkl"

    logger.info(f"Saving model {model_name} to {model_path}")

    try:
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Successfully saved model to {model_path}")
        return str(model_path)

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a model from disk.

    Args:
        model_path (str or Path): Path to the model file.

    Returns:
        Any: The loaded model.

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    # Convert to Path object
    model_path = Path(model_path)

    logger.info(f"Loading model from {model_path}")

    # Check if file exists
    if not model_path.exists():
        error_msg = f"Model file not found at path: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def get_best_model(
    results: Dict[str, Dict[str, Any]], metric: str = "f1"
) -> Tuple[str, Any]:
    """
    Get the best model based on the specified metric.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of model results.
        metric (str, optional): Metric to use for comparison.
            Defaults to 'f1'.

    Returns:
        Tuple[str, Any]: Name and instance of the best model.

    Raises:
        ValueError: If results dictionary is empty or no valid models found.
    """
    if not results:
        error_msg = "Results dictionary is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Filter out models with errors
    valid_results = {name: res for name, res in results.items() if "error" not in res}

    if not valid_results:
        error_msg = "No valid models found in results"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Find best model
    best_score = -float("inf")
    best_model_name = None

    for model_name, model_result in valid_results.items():
        if metric in model_result["evaluation"]:
            score = model_result["evaluation"][metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name

    if best_model_name is None:
        error_msg = f"No models found with metric: {metric}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Best model: {best_model_name} with {metric}={best_score:.4f}")

    return best_model_name, valid_results[best_model_name]["model"]
