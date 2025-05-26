"""
Model selection functionality for the Spam Email Detection project.

This module provides functions for selecting and configuring models.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger
from src.utils.config import RANDOM_STATE

# Set up logger
logger = setup_logger(__name__)


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get a dictionary of model configurations.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to configuration dictionaries.
    """
    model_configs = {
        "Gaussian Naive Bayes": {
            "model": GaussianNB,
            "requires_scaling": False,
            "params": {
                "default": {},
                "grid_search": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
            },
        },
        "Multinomial Naive Bayes": {
            "model": MultinomialNB,
            "requires_scaling": False,
            "params": {
                "default": {"alpha": 1.0},
                "grid_search": {"alpha": [0.1, 0.5, 1.0, 2.0]},
            },
        },
        "Bernoulli Naive Bayes": {
            "model": BernoulliNB,
            "requires_scaling": False,
            "params": {
                "default": {"alpha": 1.0},
                "grid_search": {"alpha": [0.1, 0.5, 1.0, 2.0]},
            },
        },
        "Logistic Regression": {
            "model": LogisticRegression,
            "requires_scaling": True,
            "params": {
                "default": {
                    "max_iter": 1000,
                    "solver": "liblinear",
                    "random_state": RANDOM_STATE,
                    "C": 1.0,
                },
                "grid_search": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                    "class_weight": [None, "balanced"],
                },
            },
        },
        "Support Vector Machine": {
            "model": SVC,
            "requires_scaling": True,
            "params": {
                "default": {
                    "kernel": "linear",
                    "C": 1.0,
                    "probability": True,
                    "random_state": RANDOM_STATE,
                },
                "grid_search": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto", 0.1, 1.0],
                    "class_weight": [None, "balanced"],
                },
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "requires_scaling": False,
            "params": {
                "default": {"n_estimators": 100, "random_state": RANDOM_STATE},
                "grid_search": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": [None, "balanced", "balanced_subsample"],
                },
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier,
            "requires_scaling": False,
            "params": {
                "default": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "random_state": RANDOM_STATE,
                },
                "grid_search": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier,
            "requires_scaling": False,
            "params": {
                "default": {"random_state": RANDOM_STATE},
                "grid_search": {
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],
                    "class_weight": [None, "balanced"],
                },
            },
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier,
            "requires_scaling": True,
            "params": {
                "default": {"n_neighbors": 5},
                "grid_search": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],  # 1 for Manhattan, 2 for Euclidean
                },
            },
        },
        "Neural Network": {
            "model": MLPClassifier,
            "requires_scaling": True,
            "params": {
                "default": {
                    "hidden_layer_sizes": (100,),
                    "max_iter": 1000,
                    "random_state": RANDOM_STATE,
                },
                "grid_search": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate_init": [0.001, 0.01],
                },
            },
        },
    }

    return model_configs


def create_model(
    model_name: str, params: Optional[Dict[str, Any]] = None
) -> Union[Any, Pipeline]:
    """
    Create a model instance with the given parameters.

    Args:
        model_name (str): Name of the model to create.
        params (Dict[str, Any], optional): Parameters for the model.
            If None, uses default parameters. Defaults to None.

    Returns:
        Union[Any, Pipeline]: Model instance or pipeline.

    Raises:
        ValueError: If the model name is not recognized.
    """
    # Get model configurations
    model_configs = get_model_configs()

    # Check if model name is valid
    if model_name not in model_configs:
        logger.error(f"Unknown model name: {model_name}")
        raise ValueError(f"Unknown model name: {model_name}")

    # Get model configuration
    config = model_configs[model_name]

    # Determine parameters
    if params is None:
        params = config["params"]["default"].copy()

    # Create model instance
    model_class = config["model"]
    model = model_class(**params)

    # Wrap in pipeline if scaling is required
    if config["requires_scaling"]:
        logger.info(f"Creating {model_name} with scaling")
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        logger.info(f"Creating {model_name} without scaling")
        return model


def get_models(
    selected_models: Optional[List[str]] = None,
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Get a dictionary of model instances.

    Args:
        selected_models (List[str], optional): List of model names to include.
            If None, includes all available models. Defaults to None.
        custom_params (Dict[str, Dict[str, Any]], optional): Custom parameters for models.
            Dictionary mapping model names to parameter dictionaries.
            If None, uses default parameters. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances.
    """
    # Get model configurations
    model_configs = get_model_configs()

    # Determine which models to include
    if selected_models is None:
        selected_models = list(model_configs.keys())
    else:
        # Validate selected models
        for model_name in selected_models:
            if model_name not in model_configs:
                logger.warning(f"Unknown model name: {model_name}, will be skipped")

        # Filter to valid model names
        selected_models = [
            model_name for model_name in selected_models if model_name in model_configs
        ]

    # Initialize custom parameters if None
    if custom_params is None:
        custom_params = {}

    # Create dictionary of model instances
    models = {}
    for model_name in selected_models:
        # Get parameters (custom or default)
        params = custom_params.get(
            model_name, model_configs[model_name]["params"]["default"].copy()
        )

        # Create model
        try:
            models[model_name] = create_model(model_name, params)
            logger.info(f"Added model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {str(e)}")

    return models


def get_recommended_models_for_text_classification() -> List[str]:
    """
    Get a list of recommended models for text classification tasks.

    Returns:
        List[str]: List of recommended model names.
    """
    return [
        "Multinomial Naive Bayes",
        "Bernoulli Naive Bayes",
        "Logistic Regression",
        "Support Vector Machine",
    ]
