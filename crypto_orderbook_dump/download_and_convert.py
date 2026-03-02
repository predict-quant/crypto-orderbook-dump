#!/usr/bin/env python3
"""
Bybit Order Book downloader with on-the-fly Parquet conversion.
Downloads archive files, replays snapshot/delta streams into full snapshots,
optionally down-samples to coarser intervals, and keeps only the desired
top-of-book depth per side before persisting to Parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import polars as pl
import requests

BASE_URL = "https://quote-saver.bycsi.com/orderbook/spot"
DEFAULT_OUTPUT_DIR = Path("data/parquet")
DEFAULT_TEMP_DIR = Path("data/tmp/orderbook")
BASE_INTERVAL_MS = 200  # Bybit snapshots default to 200ms cadence


def _is_zero(value: str) -> bool:
    """Return True when a quantity string represents zero."""

    try:
        return float(value) == 0.0
    except ValueError:
        return False


class OrderBookSide:
    """Mutable representation of one side of the book."""

    def __init__(self, descending: bool) -> None:
        self.descending = descending
        self.levels: Dict[str, str] = {}

    def reset(self, levels: Optional[List[List[str]]]) -> None:
        self.levels.clear()
        self.apply(levels)

    def apply(self, updates: Optional[List[List[str]]]) -> None:
        if not updates:
            return
        for price, qty in updates:
            qty_str = str(qty)
            price_str = str(price)
            if _is_zero(qty_str):
                self.levels.pop(price_str, None)
            else:
                self.levels[price_str] = qty_str

    def top(self, depth: int) -> List[List[str]]:
        if depth <= 0:
            return []

        def sort_key(item: Tuple[str, str]) -> float:
            try:
                return float(item[0])
            except ValueError:
                return 0.0

        ordered = sorted(
            self.levels.items(),
            key=sort_key,
            reverse=self.descending,
        )
        return [[price, qty] for price, qty in ordered[:depth]]


class OrderBookState:
    """Tracks book state across snapshot/delta events."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.bids = OrderBookSide(descending=True)
        self.asks = OrderBookSide(descending=False)
        self._ready = False

    def apply(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_type = message.get("type")
        payload_raw = message.get("data") or {}
        payload = payload_raw if isinstance(payload_raw, dict) else {}
        bids = payload.get("b")
        asks = payload.get("a")

        if event_type == "snapshot":
            self.bids.reset(bids)
            self.asks.reset(asks)
            self._ready = True
        elif event_type == "delta":
            if not self._ready:
                return None
            self.bids.apply(bids)
            self.asks.apply(asks)
        else:
            return None

        if not self._ready:
            return None

        view = {
            "ts": message.get("ts"),
            "cts": message.get("cts"),
            "type": "snapshot",
            "u": payload.get("u"),
            "seq": payload.get("seq"),
            "bids": json.dumps(self.bids.top(self.depth)),
            "asks": json.dumps(self.asks.top(self.depth)),
            "depth": self.depth,
        }
        return view


def parse_interval(value: str) -> Tuple[int, str]:
    """Convert interval strings like '1m' or '15m' to milliseconds plus label."""

    raw = value.strip().lower()
    if raw in {"", "raw"}:
        return BASE_INTERVAL_MS, "250ms"

    units = [("ms", 1), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)]

    for suffix, multiplier in units:
        if raw.endswith(suffix):
            number_part = raw[: -len(suffix)] or "0"
            try:
                amount = float(number_part)
            except ValueError as exc:  # pragma: no cover - arg validation
                raise ValueError(f"Invalid interval '{value}'.") from exc

            interval_ms = int(amount * multiplier)
            if interval_ms <= 0:
                raise ValueError("Interval must be greater than zero.")

            label = f"{str(amount).rstrip('0').rstrip('.') if '.' in str(amount) else int(amount)}{suffix}"
            return interval_ms, label

    raise ValueError(
        "Unsupported interval suffix. Use one of: ms, s, m, h (e.g. 250ms, 1m, 15m)."
    )


def daterange(start: date, end: date) -> Iterable[datetime]:
    """Yield each day between start and end inclusive."""

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_filename(symbol: str, date: datetime) -> str:
    """Filename used by the Bybit archive for a symbol/date pair."""

    date_str = date.strftime("%Y-%m-%d")
    return f"{date_str}_{symbol}_ob200.data.zip"


def build_url(symbol: str, date: datetime) -> str:
    """Construct the download URL for a given symbol/date."""

    return f"{BASE_URL}/{symbol}/{build_filename(symbol, date)}"


def build_output_path(
    symbol: str, date: datetime, interval_label: str, depth: int, base_dir: Path
) -> Path:
    """Destination path for the converted Parquet file."""

    date_str = date.strftime("%Y-%m-%d")
    safe_interval = interval_label.replace("/", "-")
    depth_tag = f"depth{depth}"
    return (
        base_dir
        / symbol
        / f"{date_str}_{symbol}_ob200_{safe_interval}_{depth_tag}.parquet"
    )


def download_to_temp(
    url: str, temp_dir: Path, retries: int = 3, timeout: int = 120
) -> Tuple[str, Optional[Path]]:
    """Download a URL to a temporary ZIP file and return status plus path."""

    temp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        fd, tmp_path = tempfile.mkstemp(prefix="ob_", suffix=".zip", dir=temp_dir)
        os.close(fd)
        temp_file = Path(tmp_path)
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code == 404:
                    temp_file.unlink(missing_ok=True)
                    return "not_found", None
                resp.raise_for_status()

                with open(temp_file, "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)

            return "success", temp_file
        except requests.exceptions.Timeout:
            temp_file.unlink(missing_ok=True)
            time.sleep(5)
        except Exception:
            temp_file.unlink(missing_ok=True)
            time.sleep(2)

    return "failed", None


def normalize_interval(
    df: pl.DataFrame, interval_ms: int, interval_label: str
) -> pl.DataFrame:
    """Downsample snapshots to the requested interval using the latest snapshot per bucket."""

    bucket_col = "bucket_ts"
    normalized = (
        df.sort("ts")
        .with_columns(((pl.col("ts") // interval_ms) * interval_ms).alias(bucket_col))
        .group_by(bucket_col, maintain_order=True)
        .agg(
            pl.col("ts").last().alias("source_ts"),
            pl.col("cts").last(),
            pl.col("type").last(),
            pl.col("u").last(),
            pl.col("seq").last(),
            pl.col("bids").last(),
            pl.col("asks").last(),
        )
        .select(
            [
                pl.col(bucket_col).alias("ts"),
                "source_ts",
                "cts",
                "type",
                "u",
                "seq",
                "bids",
                "asks",
            ]
        )
    )

    return normalized


def convert_zip_to_parquet(
    zip_path: Path,
    output_path: Path,
    interval_ms: int,
    interval_label: str,
    depth: int,
    batch_size: int,
    verify: bool,
) -> Dict[str, object]:
    """Convert a single ZIP archive into a Parquet dataset."""

    batches: List[pl.DataFrame] = []
    current_batch: List[Dict[str, object]] = []
    total = 0
    errors = 0
    state = OrderBookState(depth)

    with zipfile.ZipFile(zip_path, "r") as archive:
        inner_files = archive.namelist()
        if not inner_files:
            raise ValueError("Archive is empty.")
        with archive.open(inner_files[0]) as payload:
            for line in payload:
                try:
                    raw = json.loads(line.decode("utf-8").strip())
                except Exception:
                    errors += 1
                    continue

                snapshot = state.apply(raw)
                if snapshot is None:
                    continue

                current_batch.append(snapshot)

                if len(current_batch) >= batch_size:
                    batches.append(pl.DataFrame(current_batch))
                    total += len(current_batch)
                    current_batch = []

            if current_batch:
                batches.append(pl.DataFrame(current_batch))
                total += len(current_batch)

    if not batches:
        raise ValueError("No snapshots reconstructed from archive.")

    df = pl.concat(batches) if len(batches) > 1 else batches[0]

    if interval_ms > BASE_INTERVAL_MS:
        df = normalize_interval(df, interval_ms, interval_label)
    else:
        df = df.sort("ts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd", compression_level=19)

    if verify:
        saved = pl.read_parquet(output_path)
        if len(saved) != len(df):
            raise ValueError("Verification mismatch after writing Parquet.")

    size_mb = output_path.stat().st_size / 1024 / 1024
    return {
        "status": "success",
        "records": len(df),
        "size_mb": size_mb,
        "errors": errors,
    }


def process_day(
    symbol: str,
    date: datetime,
    output_dir: Path,
    temp_dir: Path,
    interval_ms: int,
    interval_label: str,
    depth: int,
    batch_size: int,
    verify: bool,
) -> Dict[str, object]:
    """Download and convert a single day for a symbol."""

    url = build_url(symbol, date)
    output_path = build_output_path(symbol, date, interval_label, depth, output_dir)
    status, temp_path = download_to_temp(url, temp_dir)
    if status == "not_found":
        return {"status": "not_found", "output": output_path}
    if status != "success" or temp_path is None:
        return {"status": "failed", "output": output_path, "reason": status}

    try:
        result = convert_zip_to_parquet(
            temp_path,
            output_path,
            interval_ms,
            interval_label,
            depth,
            batch_size,
            verify,
        )
        result["output"] = output_path
        return result
    finally:
        temp_path.unlink(missing_ok=True)


def process_symbol(
    symbol: str,
    start: date,
    end: date,
    output_dir: Path,
    temp_dir: Path,
    interval_ms: int,
    interval_label: str,
    depth: int,
    batch_size: int,
    verify: bool,
    workers: int,
    dry_run: bool,
) -> Dict[str, int]:
    """Handle the requested date span for a single symbol."""
    symbol = symbol.upper()

    stats = {"converted": 0, "skipped": 0, "failed": 0, "missing": 0}
    dates = list(daterange(start, end))

    pending = []
    for date_to_process in dates:
        output_path = build_output_path(
            symbol, date_to_process, interval_label, depth, output_dir
        )
        if output_path.exists():
            stats["skipped"] += 1
            continue
        pending.append(date_to_process)

    print(f"  To process: {len(pending)}, Skipped: {stats['skipped']}")

    if dry_run:
        for date in pending:
            print(
                "    → "
                f"{build_url(symbol, date)} -> "
                f"{build_output_path(symbol, date, interval_label, depth, output_dir).name}"
            )
        return stats

    if not pending:
        return stats

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_day,
                symbol,
                date,
                output_dir,
                temp_dir,
                interval_ms,
                interval_label,
                depth,
                batch_size,
                verify,
            ): date
            for date in pending
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"    ✗ {date.strftime('%Y-%m-%d')} {symbol}: {exc}")
                stats["failed"] += 1
                continue

            status = result.get("status")
            output_path = cast(Path, result.get("output"))
            if status == "success":
                stats["converted"] += 1
                print(
                    f"    ✓ {output_path.name} ({result['records']:,} rows, {result['size_mb']:.1f} MB)"
                )
            elif status == "not_found":
                stats["missing"] += 1
                print(f"    - {output_path.name} (remote file missing)")
            else:
                stats["failed"] += 1
                reason = result.get("reason", "conversion error")
                print(f"    ✗ {output_path.name} ({reason})")

    return stats


def parse_args() -> argparse.Namespace:
    """Configure CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Download Bybit order book archives and emit interval-normalized Parquet files in one pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_and_convert.py BTCUSDT --start-date 2025-05-01 --end-date 2025-05-07 --interval 1m --depth 10
    python download_and_convert.py --symbols BTCUSDT,ETHUSDT --start-date 2025-06-01 --end-date 2025-06-30 --interval 15m --workers 6 --depth 5
        """,
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Single symbol to download (e.g. BTCUSDT). Overrides --symbols when provided.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (BTCUSDT,ETHUSDT,...).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date inclusive (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date inclusive (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="250ms",
        help="Target interval resolution (250ms, 1s, 1m, 15m, 1h ...).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=200,
        help="Number of price levels per side to store after reconstruction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to store Parquet files (defaults to data/parquet).",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=str(DEFAULT_TEMP_DIR),
        help="Directory used for temporary ZIP downloads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent downloads/conversions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per in-memory batch before flushing to Polars DataFrame.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip Parquet verification pass after writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List download URLs without fetching or converting.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()

    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = ["BTCUSDT"]

    try:
        interval_ms, interval_label = parse_interval(args.interval)
    except ValueError as exc:
        raise SystemExit(f"Interval error: {exc}") from exc

    if args.depth <= 0:
        raise SystemExit("Depth must be greater than zero.")

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if end < start:
        raise SystemExit("End date must be after start date.")

    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    total_days = (end - start).days + 1

    print("Bybit Downloader + Converter")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.start_date} → {args.end_date} ({total_days} days)")
    print(f"Interval: {interval_label}")
    print(f"Depth: top {args.depth}")
    print(f"Output: {output_dir}")
    print(f"Temp: {temp_dir}")
    print("=" * 60)

    overall = {"converted": 0, "failed": 0, "skipped": 0, "missing": 0}

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] {symbol}")
        print("-" * 30)

        symbol_stats = process_symbol(
            symbol,
            start,
            end,
            output_dir,
            temp_dir,
            interval_ms,
            interval_label,
            args.depth,
            args.batch_size,
            not args.no_verify,
            args.workers,
            args.dry_run,
        )

        for key in overall:
            overall[key] += symbol_stats.get(key, 0)

    print("\n" + "=" * 60)
    print(
        "TOTAL: "
        f"{overall['converted']} converted, {overall['missing']} missing, "
        f"{overall['failed']} failed, {overall['skipped']} skipped"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
