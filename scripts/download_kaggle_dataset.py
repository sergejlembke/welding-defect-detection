import argparse
from pathlib import Path
import sys


def download_and_extract_kaggle_dataset(
    dataset_slug: str, target_dir: Path
) -> None:
    """Download and extract a Kaggle dataset using the official Kaggle API.

    Args:
        dataset_slug: Kaggle identifier in 'username/dataset-name' format.
        target_dir: Target directory path where raw images will be extracted.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "[Error] Package 'kaggle' is not installed. "
            "Please run 'pip install kaggle' first."
        )
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
        print("[Info] Kaggle API authentication successful.")
    except Exception as e:
        print(
            f"[Error] Kaggle API authentication failed: {e}\n"
            "Please ensure your Kaggle credentials are configured correctly.\n"
            "Supported auth methods:\n"
            "  1. Token file ('access_token' or 'kaggle.json') in '~/.kaggle/' (Linux/Mac) "
            "or '%USERPROFILE%\\.kaggle\\' (Windows)\n"
            "  2. Environment variable 'KAGGLE_API_TOKEN' (or 'KAGGLE_USERNAME' & 'KAGGLE_KEY')"
        )
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Downloading Kaggle dataset '{dataset_slug}'...")

    try:
        api.dataset_download_files(
            dataset_slug, path=str(target_dir), unzip=True
        )
        print(
            f"[Success] Dataset downloaded and extracted to: '{target_dir}'"
        )
    except Exception as e:
        print(f"[Error] Failed to download dataset: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point for downloading Kaggle datasets."""
    parser = argparse.ArgumentParser(
        description="Download and extract raw Kaggle datasets for welding inspection."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sukmaadhiwijaya/welding-defect-object-detection",
        help="Kaggle dataset slug (format: username/dataset-name)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_welding_dataset",
        help="Directory to store extracted raw dataset files",
    )
    args = parser.parse_args()

    download_and_extract_kaggle_dataset(
        args.dataset, Path(args.output_dir)
    )


if __name__ == "__main__":
    main()