"""
Preprocessing utilities for the lvdpy package.

This module provides functions for generating metadata and preprocessing audio datasets.

Author: Danilo Ristic
Date created: 2025-08-05
"""

# System imports
import os
import shutil
from pathlib import Path

# Other imports
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold, train_test_split


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
        path = os.path.join(output_path, row["file_path"].parent.name, row["file_path"].name)
        if row["sampling_rate"] < max_sr:
            y, sr = librosa.load(row["file_path"], sr=row["sampling_rate"])
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=max_sr)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, y_resampled, max_sr)
        else:
            y, sr = librosa.load(row["file_path"], sr=None)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, y, sr)


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
        parentname = row["file_path"].parent.name
        filename = row["file_path"].name.replace(".wav", "_noise_" + str(noise_level) + ".wav")
        path = os.path.join(output_path, parentname, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
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


def split_and_organize_files(
    input_dir,
    output_dir,
    test_size=0.2,
    val_size=0.1,
    n_splits=5,
    random_state=42,
    stratify_col="category",
):
    """
    Split files into train, test, validation, and k-fold folders, and move/copy
                                                    them to appropriate directories.

    Args:
        metadata_df (pandas.DataFrame): DataFrame with at least 'file_path' and 'category' columns.
        output_dir (str): Path to the output directory where folders will be created.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        n_splits (int): Number of folds for K-Fold cross-validation.
        random_state (int): Random seed for reproducibility.
        stratify_col (str): Column name to use for stratification.
    """
    # Split into train+val and test
    metadata_df = create_metadata(input_dir)

    trainval_df, test_df = train_test_split(
        metadata_df,
        test_size=test_size,
        random_state=random_state,
        stratify=metadata_df[stratify_col] if stratify_col in metadata_df else None,
    )

    # Split train+val into train and val
    if val_size > 0:
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_size,
            random_state=random_state,
            stratify=trainval_df[stratify_col] if stratify_col in trainval_df else None,
        )
    else:
        train_df, val_df = trainval_df, pd.DataFrame()

    # Helper to copy files
    def copy_files(df, split_name):
        for _, row in df.iterrows():
            category = row["category"]
            dest_dir = os.path.join(output_dir, split_name, category)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, row["file_path"].name)
            shutil.copy2(row["file_path"], dest_path)

    # Copy files to train, val, test folders
    copy_files(train_df, "train")
    if not val_df.empty:
        copy_files(val_df, "val")
    copy_files(test_df, "test")

    # K-Fold split on train+val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = trainval_df.index
    y = trainval_df[stratify_col] if stratify_col in trainval_df else None
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # fold_dir = os.path.join(output_dir, f"kfold_{fold + 1}")
        # Train fold
        fold_train_df = trainval_df.iloc[train_idx]
        copy_files(fold_train_df, os.path.join(f"kfold_{fold + 1}", "train"))
        # Validation fold
        fold_val_df = trainval_df.iloc[val_idx]
        copy_files(fold_val_df, os.path.join(f"kfold_{fold + 1}", "val"))
