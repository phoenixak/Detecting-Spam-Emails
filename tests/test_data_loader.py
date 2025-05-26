"""
Test module for the data_loader module.
"""

import unittest
import pandas as pd
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import (
    load_data,
    preprocess_data,
    load_and_preprocess,
    split_features_target,
)


class TestDataLoader(unittest.TestCase):
    """
    Test case for the data_loader module.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.temp_file_path = Path(self.temp_file.name)

        # Create a test DataFrame
        self.test_df = pd.DataFrame(
            {
                "v1": ["ham", "spam", "ham", "spam"],
                "v2": [
                    "This is a test message",
                    "Buy now!",
                    "Hello, how are you?",
                    "Win a prize!",
                ],
                "Unnamed: 2": [None, None, None, None],
                "Unnamed: 3": [None, None, None, None],
            }
        )

        # Save the DataFrame to the temporary file
        self.test_df.to_csv(self.temp_file_path, index=False)
        self.temp_file.close()

    def tearDown(self):
        """
        Tear down test fixtures.
        """
        # Remove the temporary file
        os.unlink(self.temp_file_path)

    def test_load_data(self):
        """
        Test the load_data function.
        """
        # Load the data
        df = load_data(self.temp_file_path)

        # Check that the data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (4, 4))
        self.assertEqual(df.columns.tolist(), ["v1", "v2", "Unnamed: 2", "Unnamed: 3"])

    def test_preprocess_data(self):
        """
        Test the preprocess_data function.
        """
        # Preprocess the data
        df_processed = preprocess_data(self.test_df)

        # Check that unnecessary columns are dropped
        self.assertEqual(df_processed.shape, (4, 2))
        self.assertEqual(df_processed.columns.tolist(), ["label", "message"])

        # Check that labels are encoded correctly
        self.assertTrue((df_processed["label"] == pd.Series([0, 1, 0, 1])).all())

    def test_load_and_preprocess(self):
        """
        Test the load_and_preprocess function.
        """
        # Load and preprocess the data
        df = load_and_preprocess(self.temp_file_path)

        # Check that the data is loaded and preprocessed correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (4, 2))
        self.assertEqual(df.columns.tolist(), ["label", "message"])
        self.assertTrue((df["label"] == pd.Series([0, 1, 0, 1])).all())

    def test_split_features_target(self):
        """
        Test the split_features_target function.
        """
        # Preprocess the data
        df_processed = preprocess_data(self.test_df)

        # Split features and target
        X, y = split_features_target(df_processed)

        # Check that features and target are split correctly
        self.assertIsInstance(X, pd.Series)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        self.assertTrue((y == pd.Series([0, 1, 0, 1])).all())
        self.assertTrue(
            (
                X
                == pd.Series(
                    [
                        "This is a test message",
                        "Buy now!",
                        "Hello, how are you?",
                        "Win a prize!",
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
