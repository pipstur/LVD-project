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
        input_path (str): Path to the directory containing .wav files.

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


def upscale_sample_rate(metadata_df):
    """
    Resample all audio files in metadata_df to the max sample rate,
    overwriting the files in-place in the same folder structure under input_path.

    Args:
        metadata_df (pandas.DataFrame): Must have 'file_path' and 'sampling_rate' columns.
    """
    max_sr = metadata_df["sampling_rate"].max()

    for idx, row in metadata_df.iterrows():
        file_path = row["file_path"]
        current_sr = row["sampling_rate"]

        if current_sr < max_sr:
            full_path = file_path
            if not os.path.exists(full_path):
                print(f"[WARN] File not found: {full_path}")
                continue

            # Load and resample
            y, sr = librosa.load(full_path, sr=current_sr)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=max_sr)

            # Overwrite the file
            sf.write(full_path, y_resampled, max_sr)

            # Optional: update the metadata
            metadata_df.at[idx, "sampling_rate"] = max_sr

            print(f"[OK] Upsampled: {file_path} → {max_sr}Hz")


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


def remove_files_by_duration(metadata_df, lower_threshold=0, upper_threshold=10):
    """
    Remove files from the metadata DataFrame that exceed a specified duration.

    Args:
        metadata_df (pandas.DataFrame):
            DataFrame containing metadata with a 'duration' column.
        lower_threshold (float): Lower duration threshold in seconds.
        upper_threshold (float): Upper duration threshold in seconds.

    Returns:
        pandas.DataFrame: Updated metadata DataFrame after removing files
                                                over the duration threshold.
    """
    mask = (metadata_df["duration"] >= lower_threshold) & (
        metadata_df["duration"] <= upper_threshold
    )
    filtered_df = metadata_df[mask]
    return filtered_df


def add_random_noise(input_path, output_path, noise_level=0.01, seed=27, n_files=-1):
    """
    Add random noise to audio files in the metadata DataFrame.

    Args:
        input_dir (str): Path to the input directory to create the dataframe.
        noise_level (float): Standard deviation of the noise to be added.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    metadata_df = create_metadata(input_path)
    if n_files >= 0 and n_files <= metadata_df.size:
        metadata_df = metadata_df.sample(n=n_files, random_state=27)
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
    output_dir,
    metadata_df,
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
        input_dir (str): Path to the input directory to create the dataframe.
        output_dir (str): Path to the output directory where folders will be created.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        n_splits (int): Number of folds for K-Fold cross-validation.
        random_state (int): Random seed for reproducibility.
        stratify_col (str): Column name to use for stratification.
        oversample (bool): Whether to oversample the minority class
                                                in the train sets of stratified splits.
    """
    # Split into train+val and test
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
        fold_path_train = os.path.join(f"kfold_{fold + 1}", "train")
        copy_files(fold_train_df, fold_path_train)

        # Validation fold
        fold_val_df = trainval_df.iloc[val_idx]
        fold_path_val = os.path.join(f"kfold_{fold + 1}", "val")
        copy_files(fold_val_df, fold_path_val)


def augmentation(
    path,
    target_duration=5.0,
    sample_rate=44100,
    overlap=0.3,
    noise_level=0.005,
    extra_noise_level=0.02,
):

    window_size = int(target_duration * sample_rate)
    hop_length = int(window_size * (1 - overlap))
    max_padding_allowed = int(3 * sample_rate)

    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    category_counts = {}

    for cat in categories:
        cat_dir = os.path.join(path, cat)
        wav_files = [f for f in os.listdir(cat_dir) if f.endswith(".wav")]
        category_counts[cat] = len(wav_files)

    max_count = max(category_counts.values())

    for cat in categories:
        print(f"Processando '{cat}'...")
        cat_dir = os.path.join(path, cat)
        wav_files = [f for f in os.listdir(cat_dir) if f.endswith(".wav")]

        if cat.lower() == "hungry":
            for f in wav_files:
                file_path = os.path.join(cat_dir, f)
                y, sr = librosa.load(file_path, sr=sample_rate)

                if len(y) < window_size:
                    y = np.pad(y, (0, window_size - len(y)))
                elif len(y) > window_size:
                    start = (len(y) - window_size) // 2
                    y = y[start : start + window_size]

                y_noisy = y + np.random.normal(0, noise_level, size=y.shape)

                sf.write(file_path, y_noisy, sample_rate)

            print(f"Total: {len(wav_files)} samples")
            continue

        new_samples = []
        for f in wav_files:
            file_path = os.path.join(cat_dir, f)
            y, sr = librosa.load(file_path, sr=sample_rate)

            for start in range(0, len(y), hop_length):
                end = start + window_size
                window = y[start:end]

                if len(window) < window_size:
                    padding_len = window_size - len(window)
                    window = np.pad(window, (0, padding_len))
                else:
                    padding_len = 0

                if padding_len >= max_padding_allowed:
                    continue

                noisy_window = window + np.random.normal(0, noise_level, size=window.shape)
                new_samples.append(noisy_window)

            os.remove(file_path)

        current_count = len(new_samples)
        i = 0
        while len(new_samples) < max_count:
            original = new_samples[i % current_count]
            duplicate = original + np.random.normal(0, extra_noise_level, size=original.shape)
            new_samples.append(duplicate)
            i += 1

        for idx, sample in enumerate(new_samples):
            new_name = f"{cat}_sample_{idx + 1}.wav"
            new_path = os.path.join(cat_dir, new_name)
            sf.write(new_path, sample, sample_rate)

        print(f"Total: {len(new_samples)} samples")

    print("Augmentation completed.")


def apply_augmentation(splits_path):
    for split_dir in os.listdir(splits_path):
        split_path = os.path.join(splits_path, split_dir)
        if not os.path.isdir(split_path):
            continue

        # Caso comum (train, test, val)
        if split_dir in ["train", "test", "val"]:
            print(f"🔹 Rodando augmentation para: {split_dir}")
            augmentation(split_path)

        # Caso especial: kfold
        elif split_dir.lower().startswith("kfold"):
            for sub_dir in os.listdir(split_path):
                sub_path = os.path.join(split_path, sub_dir)
                if os.path.isdir(sub_path):
                    print(f"🔹 Rodando augmentation para: {split_dir}/{sub_dir}")
                    augmentation(sub_path)
