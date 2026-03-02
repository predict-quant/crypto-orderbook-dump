import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import kaggle

from crypto_orderbook_dump.download_and_convert import process_symbol

DEFAULT_OUTPUT_DIR = Path("data/parquet")
DEFAULT_TEMP_DIR = Path("data/tmp/orderbook")


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def parse_args() -> argparse.Namespace:
    """Configure CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Download Bybit order book archives and emit interval-normalized Parquet files in one pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python crypto_orderbook_dump,main --symbol=BTCUSDT
        """,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to download (e.g. BTCUSDT)",
    )

    return parser.parse_args()


def download_latest_metadata(dataset_slug, upload_dir):
    """Download the dataset metadata from Kaggle."""
    try:
        kaggle.api.dataset_metadata(dataset_slug, path=upload_dir)
        return True
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return False


def download_latest_dataset(dataset_slug, upload_dir):
    """Download the latest dataset from Kaggle."""
    # Use Kaggle Python API to download the dataset directly to memory
    kaggle.api.dataset_download_files(dataset_slug, path=upload_dir, unzip=True)


def create_dataset_on_kaggle(symbol: str, upload_dir: Path):
    """Create a new dataset on Kaggle."""
    try:
        dataset_slug = f"gyroflaw/bybit-{symbol.lower()}-orderbook-snapshots"  # Kaggle dataset slug
        metadata = {
            "title": f"Bybit {symbol.upper()} Order Book Snapshots",
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        # Save metadata to JSON file
        metadata_path = upload_dir / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            import json

            json.dump(metadata, f)

        kaggle.api.dataset_create_new(
            folder=upload_dir,
            public=True,
            convert_to_csv=False,
            dir_mode="zip",
        )
        print(f"Dataset {dataset_slug} created successfully on Kaggle.")
    except Exception as e:
        print(f"Error creating dataset: {e}")


def main():

    args = parse_args()
    symbol = args.symbol.upper() if args.symbol else "BTCUSDT"

    dataset_slug = (
        f"gyroflaw/bybit-{symbol.lower()}-orderbook-snapshots"  # Kaggle dataset slug
    )
    upload_dir = DEFAULT_OUTPUT_DIR / symbol

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Step 1: Download the latest dataset and metadata from Kaggle

    print("Downloading dataset metadata from Kaggle...")
    dataset_exist = download_latest_metadata(dataset_slug, upload_dir)

    if dataset_exist:
        print("Downloading dataset from Kaggle...")
        download_latest_dataset(dataset_slug, upload_dir)

    start_date: date | None = None
    if os.path.exists(upload_dir):
        # List all files in the output directory for the symbol, and find the latest date
        files = list((upload_dir).glob("*.parquet"))
        if len(files) > 0:
            # File name format
            # data/parquet/BTCUSDT/2025-06-01_BTCUSDT_ob200_1m_depth10.parquet
            def extract_date(file_path: Path) -> date:
                date_str = file_path.stem.split("_")[0]
                return date.fromisoformat(date_str)

            latest_file = max(files, key=lambda f: f.stem)

            start_date = extract_date(latest_file) + timedelta(days=1)

    if start_date is None:
        start_date = date(2025, 5, 1)

    end_date = start_date + timedelta(days=3)

    for single_date in daterange(start_date, end_date):
        print(f"processing {single_date.strftime('%Y-%m-%d')}")
        process_symbol(
            "BTCUSDT",
            single_date,
            end_date,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_TEMP_DIR,
            6000,
            "1m",
            10,
            50_000,
            True,
            4,
            False,
        )

    if not dataset_exist:
        print("Creating dataset on Kaggle...")
        create_dataset_on_kaggle(symbol, upload_dir)


if __name__ == "__main__":
    main()
