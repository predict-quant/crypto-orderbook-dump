import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, upload_file

load_dotenv()


orderbook_path = Path("data/parquet/binance_ws")

# List sub directory to get symbols


def upload_symbol(symbol: str):
    symbol_path = orderbook_path / symbol
    for file in symbol_path.glob("*.parquet"):
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=f"{symbol}/{file.name}",  # Desired path in the repo
            repo_id="predict-quant/binance-orderbook",
            repo_type="dataset",
        )


def main():
    orderbook_path = Path("data/parquet/binance_ws")

    symbols = [folder.name for folder in orderbook_path.iterdir() if folder.is_dir()]

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    for symbol in symbols:
        upload_symbol(symbol)


if __name__ == "__main__":
    main()
