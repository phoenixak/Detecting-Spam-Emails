"""
Logging configuration for the Spam Email Detection project.

This module sets up logging for the project with appropriate handlers and formatters.
"""

import logging
import os
import sys
from typing import Optional

from src.utils.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name (str, optional): Name of the logger. If None, uses the root logger.
            Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    try:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")

    return logger
