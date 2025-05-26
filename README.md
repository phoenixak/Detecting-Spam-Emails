# Spam Email Detection

A machine learning project for detecting spam emails with high accuracy.

## Overview

This project provides a comprehensive solution for detecting spam emails using various machine learning algorithms. It includes:

- Data preprocessing and feature engineering
- Multiple classification models for comparison
- Model evaluation and visualization
- Command-line interface for making predictions

## Features

- Clean and modular code structure
- Comprehensive logging
- Model comparison and selection
- Visualization of results
- Easy-to-use command-line interface
- Unit tests

## Project Structure

```
Detecting-Spam-Emails/
├── dataset/                  # Dataset directory
│   └── spam.csv              # SMS Spam Collection dataset
├── src/                      # Source code
│   ├── data/                 # Data handling modules
│   │   ├── data_loader.py    # Data loading and preprocessing
│   │   └── feature_engineering.py # Feature extraction and engineering
│   ├── models/               # Model-related modules
│   │   ├── model_selection.py # Model configurations
│   │   ├── model_training.py  # Training and evaluation
│   │   └── prediction.py      # Making predictions
│   ├── utils/                # Utility modules
│   │   ├── config.py         # Configuration settings
│   │   └── logger.py         # Logging setup
│   ├── visualization/        # Visualization modules
│   │   └── visualize.py      # Data and result visualization
│   ├── main.py               # Main script for running the pipeline
│   └── predict.py            # CLI for making predictions
├── tests/                    # Unit tests
│   ├── test_data_loader.py   # Tests for data loading
│   └── test_feature_engineering.py # Tests for feature engineering
├── models/                   # Saved models (created at runtime)
├── results/                  # Results and visualizations (created at runtime)
└── logs/                     # Log files (created at runtime)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Detecting-Spam-Emails.git
   cd Detecting-Spam-Emails
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Pipeline

To run the complete pipeline (data loading, preprocessing, model training, evaluation, and visualization):

```bash
python -m src.main
```

Optional arguments:
- `--data-file PATH`: Path to the data file (default: dataset/spam.csv)
- `--vectorizer {tfidf,count}`: Vectorization method (default: tfidf)
- `--models MODEL [MODEL ...]`: List of models to train
- `--metric METRIC`: Metric for model comparison (default: f1)
- `--no-visualize`: Disable visualization
- `--save-best-only`: Save only the best model
- `--dry-run`: Run without saving any files
- `--verbose`: Enable verbose output

### Making Predictions

To make predictions on new emails:

```bash
python -m src.predict --email "Win a free iPhone now!"
```

Or from a file:
```bash
python -m src.predict --email-file path/to/email.txt
```

For batch predictions:
```bash
python -m src.predict --emails-file path/to/emails.txt
```

Optional arguments:
- `--model PATH`: Path to the model file
- `--vectorizer PATH`: Path to the vectorizer file
- `--output PATH`: Output file for predictions
- `--json`: Output results in JSON format
- `--no-preprocess`: Disable text preprocessing
- `--verbose`: Enable verbose output

## Dataset

The project uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), a public dataset of SMS messages that have been labeled as spam or ham (non-spam).

## Models

The project includes various classification models:
- Naive Bayes (Multinomial, Bernoulli, Gaussian)
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors
- Neural Network

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/) 