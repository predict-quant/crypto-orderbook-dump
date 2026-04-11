# --- Class-based client with robust 24/7 operation ---
import argparse
import asyncio
import functools
import json
import os
import time
from datetime import date
from os import PathLike
from pathlib import Path

import aiohttp
import polars as pl
import websockets
from dotenv import load_dotenv
from huggingface_hub import login, upload_file

load_dotenv()


class BinanceSpotOrderBookDumper:
    BASE_URL = "wss://stream.binance.com:9443"
    MAX_CONN_HOURS = 23.5  # reconnect before 24h forced disconnect
    # Spot: server sends ping every 20s, websockets library auto-responds with pong
    MAX_MSGS_PER_SEC = 10

    MAX_UPLOAD_RETRIES = 3
    UPLOAD_RETRY_DELAY = 30  # seconds between retries

    # Snapshot is buffered before we start, but if lastUpdateId < U of first
    # buffered event we must re-fetch. We retry up to this many times.
    MAX_SNAPSHOT_RETRIES = 5

    def __init__(
        self, symbols: list[str], depth: int, output_dir: PathLike, batch_size: int
    ):
        self.symbols = [s.upper() for s in symbols]
        self.depth = depth
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.buffers = {symbol: [] for symbol in self.symbols}
        self.huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        self._hf_logged_in = False
        self._upload_tasks: set[asyncio.Task] = set()
        self._stop = False

    def make_stream_name(self, symbol):
        return f"{symbol.lower()}@depth@100ms"

    def build_url(self):
        streams = "/".join([self.make_stream_name(s) for s in self.symbols])
        url = f"{self.BASE_URL}/stream?streams={streams}"
        return url

    def _hf_login(self):
        if not self._hf_logged_in and self.huggingface_token:
            login(token=self.huggingface_token)
            self._hf_logged_in = True

    async def run(self):
        # Retry uploading any leftover files from previous runs
        await self._retry_leftover_uploads()

        while not self._stop:
            try:
                await self._run_once()
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback

                traceback.print_exc()
                print("Reconnecting in 10 seconds...")
                await asyncio.sleep(10)

    async def _run_once(self):
        url = self.build_url()
        print(f"Connecting to {url}")
        start_time = time.time()

        # Per-symbol state
        snapshots = {}  # symbol -> snapshot record (or None)
        last_records = {}  # symbol -> last applied record
        first_event_U = {}  # symbol -> U of first buffered event after (re-)connect
        event_queues = {symbol: [] for symbol in self.symbols}  # pre-snapshot buffer

        async with websockets.connect(url, ping_interval=None) as ws:
            # ping_interval=None disables client-initiated pings; the library
            # still auto-responds to server ping frames with pong frames.
            print("WebSocket connection established.")

            async for msg in ws:
                # Reconnect before 24h forced disconnect
                if time.time() - start_time > self.MAX_CONN_HOURS * 3600:
                    print("Reconnecting before 24h forced disconnect.")
                    break

                data = json.loads(msg)
                payload = data.get("data", {})
                stream = data.get("stream", "")
                symbol = stream.split("@", 1)[0].upper()
                if symbol not in self.buffers:
                    continue

                # Spot diff-depth payload:
                # e, E (event time), s (symbol), U (first update ID), u (final update ID)
                # b (bids), a (asks)
                # NOTE: no T (trade time) or pu (prev update ID) in spot
                record = {
                    "e": payload.get("e"),
                    "lastUpdateId": None,
                    "E": payload.get("E"),
                    "U": payload.get("U"),
                    "u": payload.get("u"),
                    "bids": json.dumps(payload.get("b", [])),
                    "asks": json.dumps(payload.get("a", [])),
                }

                snapshot = snapshots.get(symbol)

                if snapshot is None:
                    # Buffer events while we (re-)fetch the snapshot.
                    # Track U of the very first buffered event per the docs.
                    if symbol not in first_event_U:
                        first_event_U[symbol] = record["U"]
                    event_queues[symbol].append(record)

                    snapshot = await self._get_snapshot_with_retry(
                        symbol, first_event_U[symbol]
                    )
                    if snapshot is None:
                        print(
                            f"Failed to get a valid snapshot for {symbol} after "
                            f"{self.MAX_SNAPSHOT_RETRIES} retries. Dropping buffered events."
                        )
                        event_queues[symbol].clear()
                        first_event_U.pop(symbol, None)
                        continue

                    snapshots[symbol] = snapshot
                    self.buffers[symbol].append(snapshot)
                    print(
                        f"Got initial snapshot for {symbol} (lastUpdateId={snapshot['lastUpdateId']})"
                    )

                    # Apply all buffered events: discard those with u <= lastUpdateId
                    for buffered in event_queues[symbol]:
                        self._apply_record(symbol, buffered, snapshot, last_records)
                    event_queues[symbol].clear()
                    first_event_U.pop(symbol, None)
                    continue

                # Normal path: apply live event
                self._apply_record(symbol, record, snapshot, last_records)

                # Flush buffer on new day or batch size reached
                _last_record = last_records.get(symbol)
                is_new_day = False
                prev_record = (
                    self.buffers[symbol][-1] if len(self.buffers[symbol]) > 1 else None
                )
                if prev_record is not None and prev_record["e"] != "snapshot":
                    last_day = date.fromtimestamp(prev_record["E"] // 1000)
                    cur_day = date.fromtimestamp(record["E"] // 1000)
                    if cur_day != last_day:
                        is_new_day = True

                if is_new_day or len(self.buffers[symbol]) >= self.batch_size:
                    ts = self.buffers[symbol][0].get("E") or int(time.time() * 1000)
                    out_path = self._get_file_path(symbol, ts)
                    schema = {
                        "e": pl.Utf8,
                        "lastUpdateId": pl.UInt64,
                        "E": pl.UInt64,
                        "U": pl.UInt64,
                        "u": pl.UInt64,
                        "bids": pl.Utf8,
                        "asks": pl.Utf8,
                    }
                    df = pl.DataFrame(self.buffers[symbol], schema=schema)
                    if out_path.exists():
                        df_existing = pl.read_parquet(out_path, schema=schema)
                        df = pl.concat([df_existing, df])
                    df.write_parquet(out_path, compression="zstd", compression_level=19)
                    print(f"Wrote {len(self.buffers[symbol])} records to {out_path}")

                    if is_new_day:
                        last_records[symbol] = None
                        snapshots[symbol] = None
                        task = asyncio.create_task(
                            self.upload_to_huggingface(
                                out_path, delete_after_upload=True
                            )
                        )
                        self._upload_tasks.add(task)
                        task.add_done_callback(self._upload_tasks.discard)
                    self.buffers[symbol].clear()

    def _apply_record(self, symbol, record, snapshot, last_records):
        """Apply a single diff-depth record to the buffer following the spot sync rules."""
        last_updateid = snapshot["lastUpdateId"]

        # Step 5: discard events where u <= lastUpdateId
        if record["u"] <= last_updateid:
            return

        # Step 6: First valid event should have lastUpdateId within [U, u].
        # If U > lastUpdateId + 1, we have a gap — log and flag re-sync.
        last_record = last_records.get(symbol)
        if last_record is None:
            # Validate first event: U must be <= lastUpdateId + 1
            if record["U"] > last_updateid + 1:
                print(
                    f"Gap detected for {symbol}: snapshot lastUpdateId={last_updateid} "
                    f"but first event U={record['U']}. Will re-sync on next iteration."
                )
                # Mark snapshot as invalid so we re-fetch on the next message
                snapshot["lastUpdateId"] = -1
                return
        else:
            # Subsequent events: U of this event must equal u of last event + 1
            if record["U"] != last_record["u"] + 1:
                print(
                    f"Gap detected for {symbol}: expected U={last_record['u'] + 1}, "
                    f"got U={record['U']}. Re-syncing snapshot."
                )
                snapshot["lastUpdateId"] = -1  # trigger re-fetch
                return

        self.buffers[symbol].append(record)
        last_records[symbol] = record

    async def _get_snapshot(self, symbol):
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={self.depth}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    record = {
                        "e": "snapshot",
                        "lastUpdateId": data.get("lastUpdateId"),
                        "E": None,
                        "U": None,
                        "u": None,
                        "bids": json.dumps(data.get("bids", [])),
                        "asks": json.dumps(data.get("asks", [])),
                    }
                    return record
                else:
                    print(f"Failed to get snapshot for {symbol}: {resp.status}")
                    return None

    async def _get_snapshot_with_retry(self, symbol, first_event_U):
        """
        Fetch a snapshot whose lastUpdateId >= first_event_U (i.e. the snapshot
        was taken after we started listening), as required by spot docs step 4.
        """
        for attempt in range(1, self.MAX_SNAPSHOT_RETRIES + 1):
            snapshot = await self._get_snapshot(symbol)
            if snapshot is None:
                await asyncio.sleep(1)
                continue
            if snapshot["lastUpdateId"] >= first_event_U:
                return snapshot
            print(
                f"[{symbol}] Snapshot lastUpdateId={snapshot['lastUpdateId']} < "
                f"first buffered U={first_event_U}. Retrying snapshot "
                f"(attempt {attempt}/{self.MAX_SNAPSHOT_RETRIES})."
            )
            await asyncio.sleep(0.5)
        return None

    async def upload_to_huggingface(self, file_path: Path, delete_after_upload=False):
        if not self.huggingface_token:
            print("Hugging Face token not found. Skipping upload.")
            return False

        loop = asyncio.get_running_loop()

        for attempt in range(1, self.MAX_UPLOAD_RETRIES + 1):
            try:
                await loop.run_in_executor(
                    None,
                    functools.partial(self._upload_file_sync, file_path),
                )
                print(f"Uploaded {file_path} to Hugging Face.")
                if delete_after_upload:
                    file_path.unlink()
                    print(f"Deleted local file {file_path} after upload.")
                return True
            except Exception as e:
                print(
                    f"[Attempt {attempt}/{self.MAX_UPLOAD_RETRIES}] "
                    f"Failed to upload {file_path}: {e}"
                )
                if attempt < self.MAX_UPLOAD_RETRIES:
                    await asyncio.sleep(self.UPLOAD_RETRY_DELAY * attempt)

        print(
            f"Giving up uploading {file_path} after {self.MAX_UPLOAD_RETRIES} attempts."
        )
        return False

    def _upload_file_sync(self, file_path: Path):
        """Synchronous upload, safe to call from run_in_executor."""
        self._hf_login()
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"{file_path.parent.name}/{file_path.name}",
            repo_id="predict-quant/binance-spot-orderbook",
            repo_type="dataset",
        )

    async def _retry_leftover_uploads(self):
        """Upload parquet files left over from previous failed uploads."""
        if not self.huggingface_token:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        today_str = time.strftime("%Y-%m-%d", time.gmtime())
        for symbol_dir in self.output_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            for f in sorted(symbol_dir.glob("*.parquet")):
                # Skip today's file — it may still be written to
                if f.name.startswith(today_str):
                    continue
                print(f"[RETRY] Uploading leftover file {f}")
                await self.upload_to_huggingface(f, delete_after_upload=True)

    def _get_file_path(self, symbol, timestamp) -> Path:
        dt = time.gmtime(timestamp // 1000)
        date_str = f"{dt.tm_year:04d}-{dt.tm_mon:02d}-{dt.tm_mday:02d}"
        out_path: Path = (
            self.output_dir / symbol / f"{date_str}_{symbol}_depth{self.depth}.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    def stop(self):
        self._stop = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binance Spot Order Book WebSocket Dumper"
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols to stream (e.g. BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Order book depth for REST snapshot (e.g. 5, 10, 20, 100, 500, 1000, 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/parquet/binance_spot_ws",
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for Parquet write"
    )
    args = parser.parse_args()
    args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    return args


async def main():
    args = parse_args()
    client = BinanceSpotOrderBookDumper(
        args.symbols, args.depth, args.output, args.batch_size
    )
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
