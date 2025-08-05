"""
Preprocessing utilities for the lvdpy package.

This module provides functions for generating metadata and preprocessing audio datasets.

Author: Danilo Ristic
Date created: 2025-08-05
"""

# Imports
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# Functions
def create_metadata(input_path):
    """
    Generate a metadata DataFrame for all .wav files in the specified directory.

    Args:
        directory (str): Path to the directory containing .wav files.

    Returns:
        pandas.DataFrame: DataFrame containing file path, filename,
                    sampling rate, duration, and category for each .wav file.
    """
    metadata = []
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                file_path = Path(os.path.join(dirname, filename))
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                sr_info = librosa.get_samplerate(file_path)
                category = file_path.parent.name
                metadata.append(
                    {
                        "file_path": file_path,
                        "filename": filename,
                        "sampling_rate": sr_info,
                        "duration": duration,
                        "category": category,
                    }
                )
    return pd.DataFrame(metadata)


def get_unique_sample_rates(metadata_df):
    """
    Get all unique sample rates from the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with a 'sampling_rate' column.

    Returns:
        list: List of unique sample rates.
    """
    return metadata_df["sampling_rate"].unique().tolist()


def upscale_sample_rate(input_path, output_path):
    """
    Upscale all audio files in the metadata DataFrame to the
                    highest sample rate found in the DataFrame.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with 'file_path' and 'sampling_rate' columns.
    """
    metadata_df = create_metadata(input_path)
    max_sr = metadata_df["sampling_rate"].max()
    for _, row in metadata_df.iterrows():
        if row["sampling_rate"] < max_sr:
            y, sr = librosa.load(row["file_path"], sr=row["sampling_rate"])
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=max_sr)
            path = os.path.join(output_path, row["file_path"].name)
            sf.write(path, y_resampled, max_sr)


def remove_underrepresented_categories(metadata_df, percent=0.5):
    """
    Remove categories with less than some percent of the median number of samples.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with a 'category' column.

    Returns:
        pandas.DataFrame: Updated metadata DataFrame after removing underrepresented categories.
    """
    category_counts = metadata_df["category"].value_counts()
    median_count = category_counts.median()
    threshold = percent * median_count
    categories_to_keep = category_counts[category_counts >= threshold].index
    return metadata_df[metadata_df["category"].isin(categories_to_keep)]


def add_random_noise(input_path, output_path, noise_level=0.01, seed=27):
    """
    Add random noise to audio files in the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with 'file_path' column.
        noise_level (float): Standard deviation of the noise to be added.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    metadata_df = create_metadata(input_path)
    for _, row in metadata_df.iterrows():
        y, sr = librosa.load(row["file_path"], sr=None)
        noise = np.random.normal(0, noise_level, y.shape)
        y_noisy = y + noise
        filename = row["file_path"].name.replace(".wav", "_noise_" + str(noise_level) + ".wav")
        path = os.path.join(output_path, filename)
        sf.write(path, y_noisy, sr)


def extract_cepstral_coefficients(metadata_df):
    """
    Extract cepstral coefficients from audio files in the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with 'file_path' column.

    Returns:
        pandas.DataFrame: DataFrame with cepstral coefficients for each audio file.
    """
    coefficients = []
    for _, row in metadata_df.iterrows():
        y, sr = librosa.load(row["file_path"], sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        coefficients.append(mfccs.mean(axis=1))
    return pd.DataFrame(coefficients, columns=[f"mfcc_{i + 1}" for i in range(13)])


# def preprocess_data(data=None, import_data=False, directory="Baby Cry Dataset/"):
#     # make make sense
#     if import_data:
#         data = create_metadata(directory)
#     if not data:
#         return data
#     data = remove_underrepresented_categories(upscale_sample_rate(data))
#     return data
