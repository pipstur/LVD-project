import kagglehub
import shutil
import os

# Download the dataset (downloads to kagglehub's internal cache)
cached_path = kagglehub.dataset_download("mennaahmed23/baby-cry-sense-dataset")

# Define your relative target path
relative_target_path = "data/baby_cry"

# Ensure target directory exists
os.makedirs(relative_target_path, exist_ok=True)

# Copy files from kagglehub cache to your local relative directory
for item in os.listdir(cached_path):
    s = os.path.join(cached_path, item)
    d = os.path.join(relative_target_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset copied to:", os.path.abspath(relative_target_path))
