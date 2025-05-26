"""
Configuration settings for the Spam Email Detection project.

This module contains all the configuration parameters used throughout the project,
making it easier to modify settings in one central location.
"""

import os
from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data settings
DATA_FILE = DATA_DIR / "spam.csv"

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

# Preprocessing settings
UPSAMPLING = True
VECTORIZER_MAX_FEATURES = 5000
VECTORIZER_MIN_DF = 5
VECTORIZER_STOP_WORDS = "english"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "spam_detection.log"

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (12, 8)
DPI = 100
