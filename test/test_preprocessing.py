"""
Unit tests for preprocessing utilities in the lvdpy package.

This module provides unit tests for functions that generate metadata and preprocess audio datasets.

Author: Danilo Ristic
Date created: 2025-08-07
"""

# pylint: disable=C0413, C0411

# Root setup
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)

# Imports
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from lvdpy import preprocessing


class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for preprocessing functions in the lvdpy package.

    This class provides unit tests for functions that generate metadata
                                            and preprocess audio datasets.
    """

    def setUp(self):
        """
        Set up a minimal mock metadata DataFrame for use in tests.
        """
        self.metadata = pd.DataFrame(
            {
                "file_path": [MagicMock(), MagicMock()],
                "filename": ["a.wav", "b.wav"],
                "sampling_rate": [16000, 8000],
                "duration": [2.0, 3.5],
                "category": ["cat1", "cat2"],
            }
        )

    def test_get_unique_sample_rates(self):
        """
        Test that get_unique_sample_rates returns all unique sample rates
                                                from the metadata DataFrame.
        """
        rates = preprocessing.get_unique_sample_rates(self.metadata)
        self.assertIn(16000, rates)
        self.assertIn(8000, rates)

    def test_remove_underrepresented_categories(self):
        """
        Test that remove_underrepresented_categories returns a DataFrame and removes as expected.
        """
        df = preprocessing.remove_underrepresented_categories(self.metadata, percent=0.5)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_remove_files_by_duration(self):
        """
        Test that remove_files_by_duration filters files by duration thresholds.
        """
        df = preprocessing.remove_files_by_duration(
            self.metadata, lower_threshold=2.5, upper_threshold=4.0
        )
        self.assertTrue((df["duration"] >= 2.5).all())
        self.assertTrue((df["duration"] <= 4.0).all())

    @patch("lvdpy.preprocessing.create_metadata")
    def test_add_random_noise(self, mock_create_metadata):
        """
        Test that add_random_noise calls sf.write for at least one file, using mocks for I/O.
        """
        mock_create_metadata.return_value = self.metadata
        with patch("lvdpy.preprocessing.librosa.load", return_value=(np.zeros(10), 16000)), patch(
            "lvdpy.preprocessing.sf.write"
        ) as mock_write:
            preprocessing.add_random_noise("input", "output", noise_level=0.01, seed=27, n_files=1)
            mock_write.assert_called()

    @patch("lvdpy.preprocessing.create_metadata")
    def test_extract_cepstral_coefficients(self, mock_create_metadata):
        """
        Test that extract_cepstral_coefficients returns a DataFrame with 13 MFCC columns.
        """
        mock_create_metadata.return_value = self.metadata
        with patch(
            "lvdpy.preprocessing.librosa.load", return_value=(np.zeros(1000), 16000)
        ), patch("lvdpy.preprocessing.librosa.feature.mfcc", return_value=np.ones((13, 10))):
            df = preprocessing.extract_cepstral_coefficients(self.metadata)
            self.assertEqual(df.shape[1], 13)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
