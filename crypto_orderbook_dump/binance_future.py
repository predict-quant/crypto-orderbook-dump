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


class BinanceOrderBookDumper:
    BASE_URL = "wss://fstream.binance.com"
    MAX_CONN_HOURS = 23.5  # reconnect before 24h forced disconnect
    PING_INTERVAL = 60 * 2  # send pong every 2 minutes
    MAX_MSGS_PER_SEC = 10

    MAX_UPLOAD_RETRIES = 3
    UPLOAD_RETRY_DELAY = 30  # seconds between retries

    def __init__(
        self, symbols: list[str], depth: int, output_dir: PathLike, batch_size: int
    ):
        self.symbols = symbols
        self.depth = depth
        self.batch_size = batch_size
        self.buffers = {symbol: [] for symbol in symbols}
        self.output_dir = Path(output_dir)
        self.depth = depth
        self.huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        self._hf_logged_in = False
        self._upload_tasks: set[asyncio.Task] = set()
        # Output paths will be generated per batch using timestamp
        self._stop = False

    def make_stream_name(self, symbol):
        return f"{symbol.lower()}@depth{self.depth}@100ms"

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

        snapshots = {}
        last_records = {}
        first_event_processed = {symbol: False for symbol in self.symbols}
        async with websockets.connect(url, ping_interval=None) as ws:
            # ping_interval=None disables client-initiated pings; the library
            # still auto-responds to server ping frames with pong frames.
            print("WebSocket connection established.")

            pong_task = asyncio.create_task(self._send_pong(ws))
            try:
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
                    record = {
                        "e": payload.get("e"),
                        "lastUpdateId": None,
                        "E": payload.get("E"),
                        "T": payload.get("T"),
                        "U": payload.get("U"),
                        "u": payload.get("u"),
                        "pu": payload.get("pu"),
                        "bids": json.dumps(payload.get("b", [])),
                        "asks": json.dumps(payload.get("a", [])),
                    }

                    snapshot = snapshots.get(symbol)
                    if snapshot is None:
                        snapshot = await self._get_snapshot(symbol)
                        if snapshot is not None:
                            snapshots[symbol] = snapshot
                            self.buffers[symbol].append(snapshot)
                            print(f"Got initial snapshot for {symbol}")
                        else:
                            print(
                                f"Failed to get initial snapshot for {symbol}, skipping updates until next attempt."
                            )
                            continue

                    # https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly
                    # How to manage a local order book correctly
                    # 4. Drop any event where u is < lastUpdateId in the snapshot.
                    if record["u"] <= snapshot["lastUpdateId"]:
                        continue

                    # 5. The first processed event should have U <= lastUpdateId AND u >= lastUpdateId
                    # U = firstUpdateId (the first update ID) from the WebSocket stream.
                    # u = finalUpdateId (the last update ID) from the WebSocket stream.
                    # lastUpdateId = the update ID you got from the REST depth snapshot.
                    if (
                        record["U"] <= snapshot["lastUpdateId"]
                        and record["u"] >= snapshot["lastUpdateId"]
                    ):
                        first_event_processed[symbol] = True
                    if not first_event_processed[symbol]:
                        continue

                    # 6. While listening to the stream, each new event's pu should be equal to the previous event's u, otherwise initialize the process from step 3.
                    last_record = last_records.get(symbol)
                    if last_record is not None:
                        if record["pu"] != last_record["u"]:
                            print(
                                f"Missing update for {symbol}: expected U={last_record['u'] + 1}, got U={record['U']}. Re-syncing snapshot."
                            )
                            snapshots[symbol] = None
                            first_event_processed[symbol] = False
                            continue
                    self.buffers[symbol].append(record)
                    last_records[symbol] = record

                    # If record is for new day, write existing buffer to Parquet and clear buffer
                    is_new_day = False
                    if last_record is not None:
                        last_record_day = date.fromtimestamp(last_record["E"] // 1000)
                        record_day = date.fromtimestamp(record["E"] // 1000)
                        if record_day != last_record_day:
                            is_new_day = True
                    # Write to Parquet in batches
                    if is_new_day or len(self.buffers[symbol]) >= self.batch_size:
                        # Use timestamp from first record in batch
                        ts = self.buffers[symbol][0].get("E") or int(time.time() * 1000)
                        out_path = self._get_file_path(symbol, ts)
                        schema = {
                            "e": pl.Utf8,
                            "lastUpdateId": pl.UInt64,
                            "E": pl.UInt64,
                            "T": pl.UInt64,
                            "U": pl.UInt64,
                            "u": pl.UInt64,
                            "pu": pl.UInt64,
                            "bids": pl.Utf8,
                            "asks": pl.Utf8,
                        }
                        df = pl.DataFrame(self.buffers[symbol], schema=schema)
                        if out_path.exists():
                            df_existing = pl.read_parquet(out_path, schema=schema)
                            df = pl.concat([df_existing, df])
                        df.write_parquet(
                            out_path, compression="zstd", compression_level=19
                        )
                        print(
                            f"Wrote {len(self.buffers[symbol])} records to {out_path}"
                        )
                        # Reset buffer and snapshot for next batch
                        if is_new_day:
                            last_records[symbol] = None
                            snapshots[symbol] = None
                            # Upload to Hugging Face in background
                            task = asyncio.create_task(
                                self.upload_to_huggingface(
                                    out_path, delete_after_upload=True
                                )
                            )
                            self._upload_tasks.add(task)
                            task.add_done_callback(self._upload_tasks.discard)
                        self.buffers[symbol].clear()
            finally:
                pong_task.cancel()
                await asyncio.gather(pong_task, return_exceptions=True)

    async def _send_pong(self, ws: websockets.ClientConnection):
        try:
            while True:
                await asyncio.sleep(self.PING_INTERVAL)
                try:
                    await ws.pong()
                    print("[PING] Sent pong frame to keep connection alive.")
                except Exception as e:
                    print(f"[PING] Pong failed: {e}")
                    break
        except asyncio.CancelledError:
            pass

    async def _get_snapshot(self, symbol):
        url = (
            f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={self.depth}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    record = {
                        "e": "snapshot",
                        "lastUpdateId": data.get("lastUpdateId"),
                        "E": data.get("E"),
                        "T": data.get("T"),
                        "U": None,
                        "u": None,
                        "pu": None,
                        "bids": json.dumps(data.get("bids", [])),
                        "asks": json.dumps(data.get("asks", [])),
                    }
                    return record
                else:
                    print(f"Failed to get snapshot for {symbol}: {resp.status}")
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
            path_in_repo=str(file_path.relative_to(self.output_dir)),
            repo_id="predict-quant/binance-orderbook",
            repo_type="dataset",
        )

    def _migrate_flat_files(self):
        """Move flat symbol/file.parquet files into symbol/year/month/file.parquet."""
        for symbol_dir in self.output_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            for f in list(symbol_dir.glob("*.parquet")):
                try:
                    date_str = f.name.split("_")[0]
                    year, month, _ = date_str.split("-")
                    new_dir = symbol_dir / year / month
                    new_dir.mkdir(parents=True, exist_ok=True)
                    new_path = new_dir / f.name
                    f.rename(new_path)
                    print(f"[MIGRATE] Moved {f.name} -> {new_path}")
                except Exception as e:
                    print(f"[MIGRATE] Failed to migrate {f}: {e}")

    async def _retry_leftover_uploads(self):
        """Migrate flat files and upload parquet files left over from previous failed uploads."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_flat_files()
        if not self.huggingface_token:
            return
        today_str = time.strftime("%Y-%m-%d", time.gmtime())
        for symbol_dir in self.output_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            for f in sorted(symbol_dir.rglob("*.parquet")):
                # Skip today's file — it may still be written to
                if f.name.startswith(today_str):
                    continue
                print(f"[RETRY] Uploading leftover file {f}")
                await self.upload_to_huggingface(f, delete_after_upload=True)

    def _get_file_path(self, symbol, timestamp) -> Path:
        dt = time.gmtime(timestamp // 1000)
        date_str = f"{dt.tm_year:04d}-{dt.tm_mon:02d}-{dt.tm_mday:02d}"
        out_path: Path = (
            self.output_dir
            / symbol
            / f"{dt.tm_year:04d}"
            / f"{dt.tm_mon:02d}"
            / f"{date_str}_{symbol}_depth{self.depth}.parquet"
        )
        print(f"Generated file path: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    def stop(self):
        self._stop = True


def parse_args():
    parser = argparse.ArgumentParser(description="Binance Order Book WebSocket Dumper")
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols to stream (e.g. BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Order book depth (e.g. 5, 10, 20, 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/parquet/binance_ws",
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
    client = BinanceOrderBookDumper(
        args.symbols, args.depth, args.output, args.batch_size
    )
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
