import os
import shutil
import argparse
import kagglehub


def download_dataset(dataset_id: str) -> str:
    """
    Downloads the dataset from KaggleHub and returns the cached path.

    Args:
        dataset_id (str): The KaggleHub dataset identifier (e.g., "user/dataset-name").

    Returns:
        str: Path to the downloaded dataset in the kagglehub cache directory.
    """
    print(f"Downloading dataset: {dataset_id}")
    return kagglehub.dataset_download(dataset_id)


def ensure_directory(path: str) -> None:
    """
    Ensures the target directory exists. Creates it if it doesn't.

    Args:
        path (str): Directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)


def copy_dataset_to_target(source_path: str, target_path: str) -> None:
    """
    Copies dataset contents from source path to target path.

    Args:
        source_path (str): Path to the downloaded dataset (source).
        target_path (str): Destination directory where data will be copied.
    """
    for item in os.listdir(source_path):
        s = os.path.join(source_path, item)
        d = os.path.join(target_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download a KaggleHub dataset and copy it to a target directory."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="KaggleHub dataset ID, e.g., 'mennaahmed23/baby-cry-sense-dataset'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/baby_cry",
        help="Relative path where the dataset will be copied (default: data/baby_cry)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to download a KaggleHub dataset and copy it to a user-specified relative path.
    """
    args = parse_args()
    cached_path = download_dataset(args.dataset_id)
    ensure_directory(args.output_dir)
    copy_dataset_to_target(cached_path, args.output_dir)
    print("✅ Dataset copied to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
