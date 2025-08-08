"""
Unit tests for visualization utilities in the lvdpy package.

This module provides unit tests for functions that visualize audio dataset metadata.

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

import pandas as pd

from lvdpy import visuals


class TestVisuals(unittest.TestCase):
    """
    Unit tests for visualization functions in the lvdpy package.

    This class provides unit tests for functions that visualize audio dataset metadata.
    """

    def setUp(self):
        """
        Set up a minimal mock metadata DataFrame for use in visualization tests.
        """
        self.metadata = pd.DataFrame({"duration": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]})

    def test_plot_duration_histogram_runs(self):
        """
        Test that plot_duration_histogram runs without error for a valid DataFrame.
        """
        visuals.plot_duration_histogram(self.metadata)

    def test_plot_category_distribution_runs(self):
        """
        Test that plot_category_distribution runs without error for a valid DataFrame.
        """
        visuals.plot_category_distribution(self.metadata)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
