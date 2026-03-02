#!/usr/bin/env python3
"""
Bybit Order Book downloader.
Fetches historical order book snapshots (200 levels) from the public archive.
Supports multiple symbols.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import requests


def download_file(url: str, filepath: Path, max_retries: int = 3) -> Tuple[bool, str]:
    """
    Download a file by URL with retries and an atomic write.

    params:
        url: Source URL
        filepath: Destination path
        max_retries: Number of attempts
    return:
        Tuple (success, message)
    """
    temp_path = filepath.with_suffix(".tmp")

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return False, "not_found"
                r.raise_for_status()

                total_size = int(r.headers.get("content-length", 0))

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if total_size > 0 and temp_path.stat().st_size != total_size:
                    raise IOError("Incomplete download")

                os.replace(temp_path, filepath)

                size_mb = total_size / 1024 / 1024
                return True, f"{size_mb:.1f} MB"

        except requests.exceptions.Timeout:
            time.sleep(5)
        except Exception:
            time.sleep(2)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return False, "failed"


def daterange(start: datetime, end: datetime):
    """
    Generate dates from start to end inclusive.

    params:
        start: Start date
        end: End date
    return:
        Date iterator
    """
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    workers: int,
    dry_run: bool,
) -> dict:
    """
    Download Order Book snapshots for a single symbol.

    params:
        symbol: Trading pair
        start: Start date
        end: End date
        output_dir: Destination directory
        workers: Number of parallel downloads
        dry_run: Only print URLs
    return:
        Stats {success, failed, skipped}
    """
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = 0

    for date in daterange(start, end):
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{date_str}_{symbol}_ob200.data.zip"
        url = f"https://quote-saver.bycsi.com/orderbook/spot/{symbol}/{filename}"
        filepath = symbol_dir / filename

        if dry_run:
            print(f"  {url}")
            continue

        if filepath.exists():
            skipped += 1
            continue

        tasks.append((url, filepath))

    if dry_run:
        return {"success": 0, "failed": 0, "skipped": 0}

    print(f"  To download: {len(tasks)}, Skipped: {skipped}")

    if not tasks:
        return {"success": 0, "failed": 0, "skipped": skipped}

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, url, path): path for url, path in tasks
        }
        for future in as_completed(futures):
            path = futures[future]
            ok, msg = future.result()
            if ok:
                print(f"    ✓ {path.name} ({msg})")
                success += 1
            elif msg == "not_found":
                print(f"    - {path.name} (not available)")
            else:
                print(f"    ✗ {path.name} ({msg})")
                failed += 1

    return {"success": success, "failed": failed, "skipped": skipped}


def main() -> None:
    """
    Entry point.

    params:
        None
    return:
        None
    """
    parser = argparse.ArgumentParser(
        description="Download Bybit Order Book data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_orderbook.py BTCUSDT --start-date 2025-05-01 --end-date 2025-05-31
  python download_orderbook.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --start-date 2025-05-01 --end-date 2025-05-31
  python download_orderbook.py ETHUSDT --start-date 2025-05-01 --end-date 2025-05-07 --workers 10
        """,
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Trading pair (or use --symbols)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list: BTCUSDT,ETHUSDT,SOLUSDT",
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/orderbook",
        help="Directory for downloads",
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="Number of parallel downloads"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print URLs without downloading"
    )

    args = parser.parse_args()

    # Determine the list of symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = ["BTCUSDT"]

    output_dir = Path(args.output_dir)
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1

    print("Bybit Order Book Downloader")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.start_date} to {args.end_date} ({total_days} days)")
    print(f"Output: {output_dir}")
    print("=" * 50)

    total_stats = {"success": 0, "failed": 0, "skipped": 0}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        print("-" * 30)

        stats = download_symbol(
            symbol, start, end, output_dir, args.workers, args.dry_run
        )

        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
        total_stats["skipped"] += stats["skipped"]

    print("\n" + "=" * 50)
    print(
        f"TOTAL: {total_stats['success']} downloaded, {total_stats['failed']} errors, {total_stats['skipped']} skipped"
    )


if __name__ == "__main__":
    main()
