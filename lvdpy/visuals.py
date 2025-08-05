"""
Visualization utilities for the lvdpy package.

This module provides functions for visualizing audio dataset metadata.

Author: Danilo Ristic
Date created: 2025-08-05
"""

# Imports
import matplotlib.pyplot as plt


# Functions
def plot_duration_histogram(metadata_df):
    """
    Plot a histogram of audio durations from the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame): DataFrame containing metadata with a 'duration' column.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(metadata_df["duration"], bins=30, color="blue", alpha=0.7)
    plt.title("Histogram of Audio Durations")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_category_distribution(metadata_df):
    """
    Plot the distribution of audio categories in the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame): DataFrame containing metadata with a 'category' column.
    """
    category_counts = metadata_df["category"].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(category_counts.index, category_counts.values, color="skyblue", alpha=0.7)
    plt.title("Distribution of Audio Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()
