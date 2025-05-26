"""
Command-line interface for making predictions with trained models.

This script provides a CLI for making predictions on new emails.
"""

import argparse
import json
import os
import joblib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from src.models.prediction import predict_single_email, predict_batch
from src.utils.logger import setup_logger
from src.utils.config import MODELS_DIR

# Set up logger
logger = setup_logger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Spam Email Prediction")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the model file. If not provided, uses the best model from the default directory.",
    )

    parser.add_argument(
        "--vectorizer",
        type=str,
        default=None,
        help="Path to the vectorizer file. If not provided, looks for a vectorizer in the same directory as the model.",
    )

    parser.add_argument("--email", type=str, help="Text of the email to classify.")

    parser.add_argument(
        "--email-file", type=str, help="Path to a file containing the email text."
    )

    parser.add_argument(
        "--emails-file",
        type=str,
        help="Path to a file containing multiple email texts, one per line.",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output file for batch predictions. If not provided, prints to stdout.",
    )

    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format."
    )

    parser.add_argument(
        "--no-preprocess", action="store_true", help="Disable text preprocessing."
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    return parser.parse_args()


def load_model_and_vectorizer(
    model_path: Optional[str] = None, vectorizer_path: Optional[str] = None
) -> tuple:
    """
    Load a model and vectorizer from disk.

    Args:
        model_path (str, optional): Path to the model file.
            If None, looks for the best model in the default directory.
        vectorizer_path (str, optional): Path to the vectorizer file.
            If None, looks for a vectorizer in the same directory as the model.

    Returns:
        tuple: (model, vectorizer)

    Raises:
        FileNotFoundError: If model or vectorizer file is not found.
    """
    # Determine model path
    if model_path is None:
        # Look for a model in the default directory
        models_dir = Path(MODELS_DIR)

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Find model files
        model_files = list(models_dir.glob("*.pkl"))

        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {models_dir}")

        # Exclude vectorizer files
        model_files = [f for f in model_files if "vectorizer" not in f.name]

        if not model_files:
            raise FileNotFoundError(
                f"No model files found (excluding vectorizer files) in directory: {models_dir}"
            )

        # Sort by modification time (most recent first)
        model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Use the most recent model
        model_path = str(model_files[0])
        logger.info(f"Using most recent model: {model_path}")

    # Load model
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    # Determine vectorizer path
    if vectorizer_path is None:
        # Look for a vectorizer in the same directory as the model
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace(".pkl", "")

        # Check for model-specific vectorizer
        specific_vectorizer_path = os.path.join(
            model_dir, f"{model_name}_vectorizer.pkl"
        )

        if os.path.exists(specific_vectorizer_path):
            vectorizer_path = specific_vectorizer_path
        else:
            # Check for generic vectorizer
            generic_vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

            if os.path.exists(generic_vectorizer_path):
                vectorizer_path = generic_vectorizer_path
            else:
                raise FileNotFoundError(
                    f"Vectorizer file not found for model: {model_path}"
                )

        logger.info(f"Using vectorizer: {vectorizer_path}")

    # Load vectorizer
    try:
        logger.info(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        logger.error(f"Error loading vectorizer: {str(e)}")
        raise

    return model, vectorizer


def format_result(result: Dict[str, Any], json_format: bool = False) -> str:
    """
    Format prediction result for output.

    Args:
        result (Dict[str, Any]): Prediction result.
        json_format (bool, optional): Whether to format as JSON.
            Defaults to False.

    Returns:
        str: Formatted result.
    """
    if json_format:
        return json.dumps(result, indent=2)

    # Format as plain text
    text = result.get("text", "")
    label = result.get("label", "")
    confidence = result.get("confidence", None)

    # Truncate text if too long
    if len(text) > 100:
        text = text[:97] + "..."

    if confidence is not None:
        return f"Classification: {label.upper()} (Confidence: {confidence:.2f})\nText: {text}"
    else:
        return f"Classification: {label.upper()}\nText: {text}"


def main():
    """
    Main function to run prediction.
    """
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer(args.model, args.vectorizer)

        # Get email text
        if args.email:
            # Single email from command line
            email_text = args.email
            batch_mode = False
        elif args.email_file:
            # Single email from file
            with open(args.email_file, "r", encoding="utf-8") as f:
                email_text = f.read()
            batch_mode = False
        elif args.emails_file:
            # Multiple emails from file
            with open(args.emails_file, "r", encoding="utf-8") as f:
                email_texts = [line.strip() for line in f if line.strip()]
            batch_mode = True
        else:
            logger.error(
                "No email input provided. Use --email, --email-file, or --emails-file."
            )
            sys.exit(1)

        # Make predictions
        if batch_mode:
            logger.info(f"Making batch predictions for {len(email_texts)} emails")
            results = predict_batch(
                model, vectorizer, email_texts, preprocess=not args.no_preprocess
            )

            # Output results
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    if args.json:
                        json.dump(results, f, indent=2)
                    else:
                        for result in results:
                            f.write(format_result(result) + "\n\n")
                logger.info(f"Saved batch predictions to {args.output}")
            else:
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    for i, result in enumerate(results):
                        print(f"Email {i+1}:")
                        print(format_result(result))
                        print()
        else:
            logger.info("Making single prediction")
            result = predict_single_email(
                model, vectorizer, email_text, preprocess=not args.no_preprocess
            )

            # Output result
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(format_result(result, json_format=args.json))
                logger.info(f"Saved prediction to {args.output}")
            else:
                print(format_result(result, json_format=args.json))

        logger.info("Prediction completed successfully")

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
