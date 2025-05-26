"""
Main script for the Spam Email Detection project.

This script serves as the entry point for the project, orchestrating the entire pipeline from
data loading and preprocessing to model training, evaluation, and visualization.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time
from typing import Dict, List, Any, Tuple, Optional

# Import modules from the project
from src.data.data_loader import load_and_preprocess, split_features_target
from src.data.feature_engineering import (
    preprocess_text,
    vectorize_text,
    balance_classes,
    prepare_data_for_training,
    extract_text_features,
)
from src.models.model_selection import (
    get_recommended_models_for_text_classification,
    get_models,
)
from src.models.model_training import (
    train_and_evaluate_models,
    get_best_model,
    save_model,
)
from src.models.prediction import create_predictor
from src.visualization.visualize import (
    plot_data_distribution,
    plot_spam_ham_wordclouds,
    plot_model_comparison,
    visualize_model_results,
)
from src.utils.logger import setup_logger
from src.utils.config import MODELS_DIR, RESULTS_DIR

# Set up logger
logger = setup_logger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Spam Email Detection")

    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to the data file. If not provided, uses the default from config.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save models. If not provided, uses the default from config.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. If not provided, uses the default from config.",
    )

    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=["tfidf", "count"],
        default="tfidf",
        help="Vectorization method: tfidf or count. Default: tfidf",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of models to train. If not provided, uses recommended models for text classification.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to use for model comparison. Default: f1",
    )

    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization."
    )

    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Save only the best model instead of all models.",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without saving any files."
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    return parser.parse_args()


def run_pipeline(args):
    """
    Run the complete spam detection pipeline.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        Dict[str, Any]: Dictionary with pipeline results.
    """
    start_time = time.time()
    logger.info("Starting spam detection pipeline")

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    df = load_and_preprocess(args.data_file)

    # Display dataset information
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {df.columns.tolist()}")

    # Visualize data distribution
    if not args.no_visualize:
        logger.info("Visualizing data distribution")
        plot_data_distribution(df, save=not args.dry_run)

        # Create word clouds for spam and ham emails
        logger.info("Creating word clouds")
        plot_spam_ham_wordclouds(df, save=not args.dry_run)

    # Split features and target
    logger.info("Splitting features and target")
    X, y = split_features_target(df)

    # Vectorize text
    logger.info(f"Vectorizing text using {args.vectorizer} method")
    X_vectorized, vectorizer = vectorize_text(X, method=args.vectorizer)

    # Balance classes if needed
    logger.info("Balancing classes")
    X_balanced, y_balanced = balance_classes(X_vectorized, y)

    # Prepare data for training
    logger.info("Preparing data for training")
    X_train, X_test, y_train, y_test = prepare_data_for_training(X_balanced, y_balanced)

    # Get models to train
    if args.models:
        logger.info(f"Using specified models: {args.models}")
        model_names = args.models
    else:
        logger.info("Using recommended models for text classification")
        model_names = get_recommended_models_for_text_classification()

    models = get_models(model_names)

    # Train and evaluate models
    logger.info("Training and evaluating models")
    model_results = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)

    # Visualize model comparison
    if not args.no_visualize:
        logger.info("Visualizing model comparison")
        plot_model_comparison(model_results, metric=args.metric, save=not args.dry_run)

    # Get best model
    logger.info(f"Getting best model based on {args.metric}")
    best_model_name, best_model = get_best_model(model_results, metric=args.metric)

    # Visualize best model results
    if not args.no_visualize:
        logger.info(f"Visualizing results for best model: {best_model_name}")
        visualize_model_results(model_results, best_model_name)

    # Save models
    if not args.dry_run:
        if args.save_best_only:
            logger.info(f"Saving best model: {best_model_name}")
            model_path = save_model(best_model, best_model_name, args.model_dir)

            # Save vectorizer alongside the model
            vectorizer_path = os.path.join(
                args.model_dir or MODELS_DIR,
                f"{best_model_name.replace(' ', '_').lower()}_vectorizer.pkl",
            )
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved vectorizer to {vectorizer_path}")
        else:
            logger.info("Saving all models")
            model_paths = {}

            for model_name, model_result in model_results.items():
                if "error" not in model_result:
                    model_path = save_model(
                        model_result["model"], model_name, args.model_dir
                    )
                    model_paths[model_name] = model_path

            # Save vectorizer
            vectorizer_path = os.path.join(
                args.model_dir or MODELS_DIR, "vectorizer.pkl"
            )
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved vectorizer to {vectorizer_path}")

    # Create a predictor with the best model
    logger.info("Creating predictor with best model")
    predictor = create_predictor(best_model, vectorizer)

    # Save results summary
    if not args.dry_run:
        results_summary = {
            "best_model": best_model_name,
            "metrics": {
                model_name: {
                    "accuracy": result["evaluation"]["accuracy"],
                    "precision": result["evaluation"]["precision"],
                    "recall": result["evaluation"]["recall"],
                    "f1": result["evaluation"]["f1"],
                    "training_time": result["training_time"],
                }
                for model_name, result in model_results.items()
                if "error" not in result
            },
            "dataset_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "spam_count": int((y == 1).sum()),
                "ham_count": int((y == 0).sum()),
            },
        }

        # Save summary to JSON file
        summary_path = os.path.join(
            args.results_dir or RESULTS_DIR, "results_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=4)

        logger.info(f"Saved results summary to {summary_path}")

    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"Pipeline completed in {execution_time:.2f} seconds")

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": vectorizer,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "predictor": predictor,
        "execution_time": execution_time,
    }


def main():
    """
    Main function to run the spam detection pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Run the pipeline
        results = run_pipeline(args)
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}", exc_info=args.verbose)
        raise


if __name__ == "__main__":
    main()
