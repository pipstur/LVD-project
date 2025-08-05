# preprocessing functions for lvdpy package
# Author: Danilo Ristic
# Date created: 2025-08-05

# Imports
import os
from pathlib import Path
import librosa
import soundfile as sf
import pandas as pd

# Functions
def create_metadata(directory = 'Baby Cry Dataset/'):
    """
    Generate a metadata DataFrame for all .wav files in the specified directory.

    Args:
        directory (str): Path to the directory containing .wav files.

    Returns:
        pandas.DataFrame: DataFrame containing file path, filename, sampling rate, duration, and category for each .wav file.
    """
    metadata = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = Path(os.path.join(dirname, filename))
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                sr_info = librosa.get_samplerate(file_path)
                category = file_path.parent.name
                metadata.append({
                    'file_path': file_path,
                    'filename': filename,
                    'sampling_rate': sr_info,
                    'duration': duration,
                    'category': category
                })
    return pd.DataFrame(metadata)

def get_unique_sample_rates(metadata_df):
    """
    Get all unique sample rates from the metadata DataFrame.

    Args:
        metadata_df (pandas.DataFrame): DataFrame containing metadata with a 'sampling_rate' column.

    Returns:
        list: List of unique sample rates.
    """
    return metadata_df['sampling_rate'].unique().tolist()

def upscale_sample_rate(metadata_df):
    """
    Upscale all audio files in the metadata DataFrame to the highest sample rate found in the DataFrame.

    Args:
        metadata_df (pandas.DataFrame): DataFrame containing metadata with 'file_path' and 'sampling_rate' columns.

    Returns:
        pandas.DataFrame: Updated metadata DataFrame after resampling.
    """
    max_sr = metadata_df['sampling_rate'].max()
    for _, row in metadata_df.iterrows():
        if row['sampling_rate'] < max_sr:
            y, sr = librosa.load(row['file_path'], sr=row['sampling_rate'])
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=max_sr)
            sf.write(row['file_path'], y_resampled, max_sr)
    return create_metadata()