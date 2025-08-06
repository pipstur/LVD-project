# lvdpy Library

## Overview

`lvdpy` is a Python package for preprocessing and visualizing audio datasets, with a focus on tasks such as metadata extraction, audio augmentation, dataset splitting, and feature extraction for machine learning workflows.

---

## Modules

### `__init__.py`

Initializes the `lvdpy` package and exposes core preprocessing utilities for easy import.

**Example usage:**
```python
from lvdpy.preprocessing import create_metadata
```

---

### `preprocessing.py`

Contains functions for preparing and augmenting audio datasets.

#### Main Functions

- **create_metadata(input_path):**
  - Scans a directory for `.wav` files and returns a DataFrame with file paths, sampling rates, durations, and categories.

- **get_unique_sample_rates(metadata_df):**
  - Returns a list of unique sample rates in the dataset.

- **upscale_sample_rate(input_path, output_path):**
  - Resamples all audio files to the highest sample rate found in the dataset and saves them to the output directory.

- **remove_underrepresented_categories(metadata_df, percent=0.5):**
  - Removes categories with less than a specified percentage of the median sample count.

- **remove_files_by_duration(metadata_df, lower_threshold=0, upper_threshold=10):**
  - Filters out files outside the specified duration range.

- **add_random_noise(input_path, output_path, noise_level=0.01, seed=27, n_files=-1):**
  - Adds random noise to audio files for augmentation.

- **extract_cepstral_coefficients(metadata_df):**
  - Extracts MFCC features from audio files and returns them as a DataFrame.

- **split_and_organize_files(...)**
  - Splits files into train, validation, test, and k-fold folders, and optionally oversamples minority classes.

---

### `visuals.py`

Provides visualization utilities for audio dataset metadata.

#### Main Functions

- **plot_duration_histogram(metadata_df):**
  - Plots a histogram of audio durations.

- **plot_category_distribution(metadata_df):**
  - Plots the distribution of audio categories.

---

## Example Workflow

```python
from lvdpy.preprocessing import create_metadata, extract_cepstral_coefficients
from lvdpy.visuals import plot_duration_histogram, plot_category_distribution

metadata = create_metadata('data/dataset/train')
plot_duration_histogram(metadata)
plot_category_distribution(metadata)

mfcc_df = extract_cepstral_coefficients(metadata)
```

---

## Author

Danilo Ristic
Created: 2025-08-05
