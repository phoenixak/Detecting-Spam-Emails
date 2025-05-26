"""
Visualization functionality for the Spam Email Detection project.

This module provides functions for visualizing data and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from wordcloud import WordCloud

from src.utils.logger import setup_logger
from src.utils.config import PLOT_STYLE, FIGSIZE, DPI, RESULTS_DIR

# Set up logger
logger = setup_logger(__name__)

# Set the plot style
plt.style.use(PLOT_STYLE)


def save_figure(
    fig: plt.Figure, filename: str, results_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Save a figure to disk.

    Args:
        fig (plt.Figure): The figure to save.
        filename (str): Name of the file.
        results_dir (str or Path, optional): Directory to save the figure in.
            If None, uses the default directory from config.
            Defaults to None.

    Returns:
        str: Path to the saved figure.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # Convert to Path object
    results_dir = Path(results_dir)

    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Add file extension if not present
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        filename += ".png"

    # Create file path
    filepath = results_dir / filename

    logger.info(f"Saving figure to {filepath}")

    try:
        # Save figure
        fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Successfully saved figure to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        raise


def plot_data_distribution(
    df: pd.DataFrame, label_col: str = "label", save: bool = True
) -> Optional[str]:
    """
    Plot the distribution of spam vs. ham in the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        label_col (str, optional): Name of the label column.
            Defaults to 'label'.
        save (bool, optional): Whether to save the figure.
            Defaults to True.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info("Plotting data distribution")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Count values
        value_counts = df[label_col].value_counts()

        # Create plot
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)

        # Add labels and title
        ax.set_title("Distribution of Spam vs. Ham Emails")

        if df[label_col].dtype == "object":
            # String labels
            ax.set_xlabel("Email Type")
        else:
            # Numeric labels
            ax.set_xlabel("Email Type")
            ax.set_xticklabels(["Ham (0)", "Spam (1)"])

        ax.set_ylabel("Count")

        # Add values on top of bars
        for i, count in enumerate(value_counts.values):
            ax.text(
                i,
                count + 0.1 * max(value_counts.values),
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add percentage
        total = value_counts.sum()
        for i, count in enumerate(value_counts.values):
            percentage = count / total * 100
            ax.text(
                i,
                count * 0.5,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        # Save figure if requested
        if save:
            return save_figure(fig, "data_distribution")

        return None

    except Exception as e:
        logger.error(f"Error plotting data distribution: {str(e)}")
        raise


def plot_wordcloud(
    texts: List[str],
    title: str = "Word Cloud",
    save: bool = True,
    filename: str = "wordcloud",
) -> Optional[str]:
    """
    Create a word cloud from texts.

    Args:
        texts (List[str]): List of text strings.
        title (str, optional): Title of the plot.
            Defaults to 'Word Cloud'.
        save (bool, optional): Whether to save the figure.
            Defaults to True.
        filename (str, optional): Name of the file if saving.
            Defaults to 'wordcloud'.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info(f"Creating word cloud for {title}")

    try:
        # Join texts
        text = " ".join(texts)

        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200,
            contour_width=3,
            contour_color="steelblue",
        ).generate(text)

        # Display word cloud
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")

        # Save figure if requested
        if save:
            return save_figure(fig, filename)

        return None

    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}")
        raise


def plot_spam_ham_wordclouds(
    df: pd.DataFrame,
    text_col: str = "message",
    label_col: str = "label",
    save: bool = True,
) -> Optional[Tuple[str, str]]:
    """
    Create separate word clouds for spam and ham emails.

    Args:
        df (pd.DataFrame): The dataset.
        text_col (str, optional): Name of the text column.
            Defaults to 'message'.
        label_col (str, optional): Name of the label column.
            Defaults to 'label'.
        save (bool, optional): Whether to save the figures.
            Defaults to True.

    Returns:
        Optional[Tuple[str, str]]: Paths to the saved figures if save is True, None otherwise.
    """
    logger.info("Creating word clouds for spam and ham emails")

    try:
        # Split texts by label
        if df[label_col].dtype == "object":
            # String labels
            spam_texts = df[df[label_col] == "spam"][text_col].tolist()
            ham_texts = df[df[label_col] == "ham"][text_col].tolist()
        else:
            # Numeric labels
            spam_texts = df[df[label_col] == 1][text_col].tolist()
            ham_texts = df[df[label_col] == 0][text_col].tolist()

        logger.info(f"Creating word cloud for {len(spam_texts)} spam emails")
        spam_path = plot_wordcloud(
            spam_texts,
            title="Spam Emails Word Cloud",
            save=save,
            filename="spam_wordcloud",
        )

        logger.info(f"Creating word cloud for {len(ham_texts)} ham emails")
        ham_path = plot_wordcloud(
            ham_texts,
            title="Ham Emails Word Cloud",
            save=save,
            filename="ham_wordcloud",
        )

        if save:
            return spam_path, ham_path

        return None

    except Exception as e:
        logger.error(f"Error creating spam/ham word clouds: {str(e)}")
        raise


def plot_confusion_matrix(matrix: np.ndarray, save: bool = True) -> Optional[str]:
    """
    Plot a confusion matrix.

    Args:
        matrix (np.ndarray): Confusion matrix.
        save (bool, optional): Whether to save the figure.
            Defaults to True.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info("Plotting confusion matrix")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Create plot
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

        # Add labels and title
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Set tick labels
        ax.set_xticklabels(["Ham (0)", "Spam (1)"])
        ax.set_yticklabels(["Ham (0)", "Spam (1)"])

        # Save figure if requested
        if save:
            return save_figure(fig, "confusion_matrix")

        return None

    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save: bool = True
) -> Optional[str]:
    """
    Plot a ROC curve.

    Args:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        roc_auc (float): Area under the ROC curve.
        save (bool, optional): Whether to save the figure.
            Defaults to True.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info("Plotting ROC curve")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )

        # Plot random guess line
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        # Set axis limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Save figure if requested
        if save:
            return save_figure(fig, "roc_curve")

        return None

    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise


def plot_precision_recall_curve(
    precision: np.ndarray, recall: np.ndarray, avg_precision: float, save: bool = True
) -> Optional[str]:
    """
    Plot a precision-recall curve.

    Args:
        precision (np.ndarray): Precision values.
        recall (np.ndarray): Recall values.
        avg_precision (float): Average precision score.
        save (bool, optional): Whether to save the figure.
            Defaults to True.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info("Plotting precision-recall curve")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot precision-recall curve
        ax.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"Precision-Recall curve (AP = {avg_precision:.2f})",
        )

        # Set axis limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Save figure if requested
        if save:
            return save_figure(fig, "precision_recall_curve")

        return None

    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {str(e)}")
        raise


def plot_feature_importance(
    features: List[Dict[str, Any]],
    title: str = "Feature Importance",
    save: bool = True,
    filename: str = "feature_importance",
) -> Optional[str]:
    """
    Plot feature importance or coefficients.

    Args:
        features (List[Dict[str, Any]]): List of feature dictionaries.
        title (str, optional): Title of the plot.
            Defaults to 'Feature Importance'.
        save (bool, optional): Whether to save the figure.
            Defaults to True.
        filename (str, optional): Name of the file if saving.
            Defaults to 'feature_importance'.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info(f"Plotting {title}")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Extract feature names and values
        if "importance" in features[0]:
            # For models with feature_importances_
            feature_names = [f["feature"] for f in features]
            values = [f["importance"] for f in features]
            value_label = "Importance"
        elif "coefficient" in features[0]:
            # For linear models with coefficients
            feature_names = [f["feature"] for f in features]
            values = [f["coefficient"] for f in features]
            value_label = "Coefficient"
        else:
            raise ValueError(
                "Features list does not contain 'importance' or 'coefficient' keys"
            )

        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = ax.barh(range(len(feature_names)), values, color=colors)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = max(width * 0.05, 0.01)
            ax.text(
                label_position,
                bar.get_y() + bar.get_height() / 2,
                f"{values[i]:.4f}",
                va="center",
            )

        # Set labels and title
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel(value_label)
        ax.set_title(title)

        # Adjust layout
        plt.tight_layout()

        # Save figure if requested
        if save:
            return save_figure(fig, filename)

        return None

    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]], metric: str = "f1", save: bool = True
) -> Optional[str]:
    """
    Plot a comparison of models based on a metric.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of model results.
        metric (str, optional): Metric to use for comparison.
            Defaults to 'f1'.
        save (bool, optional): Whether to save the figure.
            Defaults to True.

    Returns:
        Optional[str]: Path to the saved figure if save is True, None otherwise.
    """
    logger.info(f"Plotting model comparison using {metric} metric")

    try:
        # Filter out models with errors
        valid_results = {
            name: res for name, res in results.items() if "error" not in res
        }

        if not valid_results:
            logger.warning("No valid models found for comparison")
            return None

        # Extract model names and metric values
        model_names = []
        metric_values = []

        for model_name, model_result in valid_results.items():
            if metric in model_result["evaluation"]:
                model_names.append(model_name)
                metric_values.append(model_result["evaluation"][metric])

        if not model_names:
            logger.warning(f"No models found with metric: {metric}")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Create plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, metric_values, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        # Set labels and title
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Model Comparison - {metric.capitalize()} Score")

        # Rotate x labels for readability
        plt.xticks(rotation=45, ha="right")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")

        # Adjust layout
        plt.tight_layout()

        # Save figure if requested
        if save:
            return save_figure(fig, f"model_comparison_{metric}")

        return None

    except Exception as e:
        logger.error(f"Error plotting model comparison: {str(e)}")
        raise


def visualize_model_results(
    results: Dict[str, Dict[str, Any]], model_name: str
) -> Dict[str, str]:
    """
    Visualize the results of a specific model.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of model results.
        model_name (str): Name of the model to visualize.

    Returns:
        Dict[str, str]: Dictionary mapping plot names to file paths.
    """
    logger.info(f"Visualizing results for model: {model_name}")

    if model_name not in results:
        error_msg = f"Model {model_name} not found in results"
        logger.error(error_msg)
        raise ValueError(error_msg)

    model_result = results[model_name]

    if "error" in model_result:
        error_msg = f"Model {model_name} has an error: {model_result['error']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    visualization_paths = {}

    # Plot confusion matrix
    if "confusion_matrix" in model_result["evaluation"]:
        logger.info("Plotting confusion matrix")
        matrix = np.array(model_result["evaluation"]["confusion_matrix"])
        path = plot_confusion_matrix(matrix)
        visualization_paths["confusion_matrix"] = path

    # Plot ROC curve
    if (
        "roc_curve" in model_result["evaluation"]
        and "roc_auc" in model_result["evaluation"]
    ):
        logger.info("Plotting ROC curve")
        fpr = np.array(model_result["evaluation"]["roc_curve"]["fpr"])
        tpr = np.array(model_result["evaluation"]["roc_curve"]["tpr"])
        roc_auc = model_result["evaluation"]["roc_auc"]
        path = plot_roc_curve(fpr, tpr, roc_auc)
        visualization_paths["roc_curve"] = path

    # Plot precision-recall curve
    if (
        "pr_curve" in model_result["evaluation"]
        and "average_precision" in model_result["evaluation"]
    ):
        logger.info("Plotting precision-recall curve")
        precision = np.array(model_result["evaluation"]["pr_curve"]["precision"])
        recall = np.array(model_result["evaluation"]["pr_curve"]["recall"])
        avg_precision = model_result["evaluation"]["average_precision"]
        path = plot_precision_recall_curve(precision, recall, avg_precision)
        visualization_paths["precision_recall_curve"] = path

    return visualization_paths
