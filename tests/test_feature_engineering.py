"""
Test module for the feature_engineering module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from scipy.sparse import spmatrix

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.feature_engineering import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
    preprocess_text,
    vectorize_text,
    balance_classes,
    prepare_data_for_training,
    extract_text_features,
)


class TestFeatureEngineering(unittest.TestCase):
    """
    Test case for the feature_engineering module.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample texts
        self.texts = [
            "Hello, how are you doing today? I hope you're well!",
            "Buy now! Limited time offer. 50% off on all products. www.example.com",
            "This is a test message with some numbers 123456 and special chars @#$%",
            "URGENT: You have won a prize of $1000. Call now to claim it!",
        ]

        # Create sample features and target
        self.X = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24],
            ]
        )

        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_clean_text(self):
        """
        Test the clean_text function.
        """
        # Clean the texts
        cleaned_texts = [clean_text(text) for text in self.texts]

        # Check that the texts are cleaned correctly
        self.assertEqual(
            cleaned_texts[0], "hello how are you doing today i hope youre well"
        )
        self.assertTrue("buy now limited time offer" in cleaned_texts[1])
        self.assertFalse("www.example.com" in cleaned_texts[1])
        self.assertFalse("123456" in cleaned_texts[2])
        self.assertFalse("@#$%" in cleaned_texts[2])
        self.assertFalse("$1000" in cleaned_texts[3])

    def test_tokenize_text(self):
        """
        Test the tokenize_text function.
        """
        # Tokenize a sample text
        tokens = tokenize_text("hello how are you")

        # Check that the text is tokenized correctly
        self.assertEqual(tokens, ["hello", "how", "are", "you"])

    def test_remove_stopwords(self):
        """
        Test the remove_stopwords function.
        """
        # Remove stopwords from sample tokens
        tokens = ["hello", "how", "are", "you", "today", "is", "a", "beautiful", "day"]
        filtered_tokens = remove_stopwords(tokens)

        # Check that stopwords are removed correctly
        self.assertNotIn("how", filtered_tokens)
        self.assertNotIn("are", filtered_tokens)
        self.assertNotIn("you", filtered_tokens)
        self.assertNotIn("is", filtered_tokens)
        self.assertNotIn("a", filtered_tokens)
        self.assertIn("hello", filtered_tokens)
        self.assertIn("today", filtered_tokens)
        self.assertIn("beautiful", filtered_tokens)
        self.assertIn("day", filtered_tokens)

    def test_stem_tokens(self):
        """
        Test the stem_tokens function.
        """
        # Stem sample tokens
        tokens = ["running", "jumps", "jumped", "jumping", "runs", "ran"]
        stemmed_tokens = stem_tokens(tokens)

        # Check that tokens are stemmed correctly
        self.assertEqual(stemmed_tokens[0], "run")
        self.assertEqual(stemmed_tokens[1], "jump")

    def test_lemmatize_tokens(self):
        """
        Test the lemmatize_tokens function.
        """
        # Lemmatize sample tokens
        tokens = [
            "running",
            "jumps",
            "jumped",
            "jumping",
            "runs",
            "ran",
            "better",
            "best",
            "good",
        ]
        lemmatized_tokens = lemmatize_tokens(tokens)

        # Check that tokens are lemmatized correctly
        self.assertEqual(
            lemmatized_tokens[0], "running"
        )  # NLTK's lemmatizer needs POS tags for verbs
        self.assertEqual(lemmatized_tokens[8], "good")

    def test_preprocess_text(self):
        """
        Test the preprocess_text function.
        """
        # Preprocess a sample text
        text = "Hello, how are you doing today? I hope you're well!"
        preprocessed_text = preprocess_text(text)

        # Check that the text is preprocessed correctly
        self.assertIsInstance(preprocessed_text, str)
        self.assertTrue(len(preprocessed_text) < len(text))

    def test_vectorize_text(self):
        """
        Test the vectorize_text function.
        """
        # Vectorize sample texts
        X_vectorized, vectorizer = vectorize_text(self.texts)

        # Check that the texts are vectorized correctly
        self.assertIsInstance(X_vectorized, spmatrix)
        self.assertEqual(X_vectorized.shape[0], len(self.texts))

    def test_balance_classes(self):
        """
        Test the balance_classes function.
        """
        # Balance classes
        X_balanced, y_balanced = balance_classes(self.X, self.y)

        # Check that classes are balanced correctly
        self.assertEqual(np.sum(y_balanced == 0), np.sum(y_balanced == 1))
        self.assertEqual(X_balanced.shape[0], len(y_balanced))

    def test_prepare_data_for_training(self):
        """
        Test the prepare_data_for_training function.
        """
        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_data_for_training(self.X, self.y)

        # Check that data is split correctly
        self.assertEqual(X_train.shape[0], len(y_train))
        self.assertEqual(X_test.shape[0], len(y_test))
        self.assertTrue(X_train.shape[0] > X_test.shape[0])

    def test_extract_text_features(self):
        """
        Test the extract_text_features function.
        """
        # Extract features from sample texts
        features = extract_text_features(self.texts)

        # Check that features are extracted correctly
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertTrue("text_length" in features.columns)
        self.assertTrue("word_count" in features.columns)
        self.assertTrue("char_per_word" in features.columns)


if __name__ == "__main__":
    unittest.main()
