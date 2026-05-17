# --- Class-based client with robust 24/7 operation ---
import argparse
import asyncio
import functools
import json
import logging
import os
import time
from datetime import date
from os import PathLike
from pathlib import Path

import polars as pl
from binance_common.configuration import ConfigurationWebSocketStreams
from binance_common.websocket import global_stream_connections
from binance_sdk_spot.spot import Spot
from binance_sdk_spot.websocket_streams import SpotWebSocketStreams
from binance_sdk_spot.websocket_streams.models import DiffBookDepthResponse
from dotenv import load_dotenv
from huggingface_hub import login, upload_file

load_dotenv()


class InvalidSpotSymbolError(Exception):
    pass


class BinanceSpotOrderBookDumper:
    MAX_CONN_HOURS = 23.5  # reconnect before 24h forced disconnect
    MAX_MSGS_PER_SEC = 10
    STALE_STREAM_SECONDS = 5  # force reconnect if no message arrives

    MAX_UPLOAD_RETRIES = 3
    UPLOAD_RETRY_DELAY = 30  # seconds between retries

    # Snapshot is buffered before we start, but if lastUpdateId < U of first
    # buffered event we must re-fetch. We retry up to this many times.
    MAX_SNAPSHOT_RETRIES = 5
    MAX_STALE_RECOVERY_ATTEMPTS = 3

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
        self._rest_client = Spot()
        self._invalid_symbols: set[str] = set()

    def _hf_login(self):
        if not self._hf_logged_in and self.huggingface_token:
            login(token=self.huggingface_token)
            self._hf_logged_in = True

    async def run(self):
        await self._validate_symbols()

        # Retry uploading any leftover files from previous runs
        await self._retry_leftover_uploads()

        while not self._stop:
            try:
                await self._run_once()
            except Exception as e:
                logging.error(f"[ERROR] {e}")
                import traceback

                traceback.print_exc()
                logging.info("Reconnecting in 10 seconds...")
                await asyncio.sleep(10)

    async def _run_once(self):
        start_time = time.time()

        # Per-symbol state
        snapshots = {}  # symbol -> snapshot record (or None)
        last_records = {}  # symbol -> last applied record
        first_event_U = {}  # symbol -> U of first buffered event after (re-)connect
        event_queues = {symbol: [] for symbol in self.symbols}  # pre-snapshot buffer
        msg_queues: dict[str, asyncio.Queue[DiffBookDepthResponse]] = {
            symbol: asyncio.Queue() for symbol in self.symbols
        }
        active_day: dict[str, date | None] = {symbol: None for symbol in self.symbols}
        last_msg_at = {symbol: time.time() for symbol in self.symbols}
        stale_recoveries = {symbol: 0 for symbol in self.symbols}

        ws_client = Spot(config_ws_streams=ConfigurationWebSocketStreams())
        logging.info("Connecting to Binance WebSocket Streams...")
        connection: (
            SpotWebSocketStreams | None
        ) = await ws_client.websocket_streams.create_connection()
        if connection is None:
            raise ConnectionError("Failed to establish WebSocket connection.")
        logging.info("WebSocket connection established.")

        for symbol in self.symbols:
            logging.info(f"Subscribing to diff book depth stream for {symbol}...")
            stream = await connection.diff_book_depth(
                symbol.lower(), update_speed="100ms"
            )

            def make_callback(sym: str):
                def callback(data: DiffBookDepthResponse) -> None:
                    last_msg_at[sym] = time.time()
                    msg_queues[sym].put_nowait(data)

                return callback

            stream.on("message", make_callback(symbol))

        logging.info("Starting main processing loop...")

        async def process_symbol(symbol: str) -> None:
            queue = msg_queues[symbol]
            while not self._stop:
                # Reconnect before 24h forced disconnect
                if time.time() - start_time > self.MAX_CONN_HOURS * 3600:
                    logging.info("Reconnecting before 24h forced disconnect.")
                    return

                try:
                    data: DiffBookDepthResponse = await asyncio.wait_for(
                        queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    stale_for = time.time() - last_msg_at[symbol]
                    if stale_for > self.STALE_STREAM_SECONDS:
                        stale_recoveries[symbol] += 1
                        logging.warning(
                            f"[{symbol}] WebSocket stream stale for {stale_for:.1f}s "
                            f"(threshold={self.STALE_STREAM_SECONDS}s, "
                            f"attempt={stale_recoveries[symbol]}/{self.MAX_STALE_RECOVERY_ATTEMPTS})."
                        )
                        if stale_recoveries[symbol] >= self.MAX_STALE_RECOVERY_ATTEMPTS:
                            raise ConnectionError(
                                f"[{symbol}] WebSocket stream stale for {stale_for:.1f}s "
                                f"after {self.MAX_STALE_RECOVERY_ATTEMPTS} recovery attempts."
                            )

                        # Isolate stale symbols so one dead stream does not
                        # immediately force reconnect for all symbols.
                        last_msg_at[symbol] = time.time()
                        snapshots[symbol] = None
                        last_records[symbol] = None
                        event_queues[symbol].clear()
                        first_event_U.pop(symbol, None)
                    continue

                stale_recoveries[symbol] = 0

                # Spot diff-depth payload:
                # e, E (event time), s (symbol), U (first update ID), u (final update ID)
                # b (bids), a (asks)
                # NOTE: no T (trade time) or pu (prev update ID) in spot
                record = {
                    "e": data.e,
                    "lastUpdateId": None,
                    "E": data.E,
                    "U": data.U,
                    "u": data.u,
                    "bids": json.dumps(data.b or []),
                    "asks": json.dumps(data.a or []),
                }

                snapshot = snapshots.get(symbol)

                if snapshot is not None and snapshot["lastUpdateId"] < 0:
                    snapshots[symbol] = None
                    last_records[symbol] = None
                    event_queues[symbol].clear()
                    first_event_U.pop(symbol, None)
                    snapshot = None

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
                        logging.warning(
                            f"Failed to get a valid snapshot for {symbol} after "
                            f"{self.MAX_SNAPSHOT_RETRIES} retries. Dropping buffered events."
                        )
                        event_queues[symbol].clear()
                        first_event_U.pop(symbol, None)
                        continue

                    snapshots[symbol] = snapshot
                    self.buffers[symbol].append(snapshot)
                    logging.info(
                        f"Got initial snapshot for {symbol} (lastUpdateId={snapshot['lastUpdateId']})"
                    )

                    # Apply all buffered events: discard those with u <= lastUpdateId
                    for buffered in event_queues[symbol]:
                        self._apply_record(symbol, buffered, snapshot, last_records)
                    event_queues[symbol].clear()
                    first_event_U.pop(symbol, None)
                    continue

                record_day = date.fromtimestamp(record["E"] // 1000)
                self._handle_day_rollover(symbol, record_day, active_day)

                # Normal path: apply live event
                self._apply_record(symbol, record, snapshot, last_records)

                if len(self.buffers[symbol]) >= self.batch_size:
                    self._flush_symbol_buffer(symbol)

        try:
            await asyncio.gather(*[process_symbol(s) for s in self.symbols])
        except Exception as e:
            logging.error(f"Error in processing loop: {e}")
        finally:
            logging.info("Closing WebSocket connection.")
            # Clear the SDK's global stream registry so streams can be
            # re-subscribed on the next connection. Without this, subscribe()
            # silently skips streams already present in global_stream_connections
            # (even though the underlying WebSocket is now closed), causing all
            # subsequent reconnects to receive no data.
            for symbol in self.symbols:
                global_stream_connections.stream_connections_map.pop(
                    f"{symbol.lower()}@depth@100ms", None
                )
            await connection.close_connection(close_session=True)

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
                logging.warning(
                    f"Gap detected for {symbol}: snapshot lastUpdateId={last_updateid} "
                    f"but first event U={record['U']}. Will re-sync on next iteration."
                )
                # Mark snapshot as invalid so we re-fetch on the next message
                snapshot["lastUpdateId"] = -1
                return
        else:
            # Subsequent events: U of this event must equal u of last event + 1
            if record["U"] != last_record["u"] + 1:
                logging.warning(
                    f"Gap detected for {symbol}: expected U={last_record['u'] + 1}, "
                    f"got U={record['U']}. Re-syncing snapshot."
                )
                snapshot["lastUpdateId"] = -1  # trigger re-fetch
                return

        self.buffers[symbol].append(record)
        last_records[symbol] = record

    async def _get_snapshot(self, symbol):
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                functools.partial(
                    self._rest_client.rest_api.depth, symbol, limit=self.depth
                ),
            )
            data = resp.data()
            record = {
                "e": "snapshot",
                "lastUpdateId": data.last_update_id,
                "E": None,
                "U": None,
                "u": None,
                "bids": json.dumps(data.bids or []),
                "asks": json.dumps(data.asks or []),
            }
            return record
        except Exception as e:
            if "-1121" in str(e) or "Invalid symbol" in str(e):
                self._invalid_symbols.add(symbol)
                raise InvalidSpotSymbolError(
                    f"{symbol} is not a valid Binance spot symbol"
                )
            logging.error(f"Failed to get snapshot for {symbol}: {e}")
            return None

    async def _validate_symbols(self):
        valid_symbols: list[str] = []

        for symbol in self.symbols:
            try:
                snapshot = await self._get_snapshot(symbol)
            except InvalidSpotSymbolError:
                logging.error(
                    f"[{symbol}] Invalid Binance spot symbol. It will be excluded."
                )
                continue

            if snapshot is None:
                logging.warning(
                    f"[{symbol}] Snapshot validation failed at startup. Keeping symbol and retrying in stream loop."
                )
            valid_symbols.append(symbol)

        if not valid_symbols:
            raise ValueError("No valid symbols available for Binance spot streams.")

        removed = sorted(set(self.symbols) - set(valid_symbols))
        if removed:
            logging.warning(f"Excluded invalid symbols: {', '.join(removed)}")
            self.buffers = {
                symbol: self.buffers.get(symbol, []) for symbol in valid_symbols
            }
            self.symbols = valid_symbols

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
            logging.debug(
                f"[{symbol}] Snapshot lastUpdateId={snapshot['lastUpdateId']} < "
                f"first buffered U={first_event_U}. Retrying snapshot "
                f"(attempt {attempt}/{self.MAX_SNAPSHOT_RETRIES})."
            )
            await asyncio.sleep(0.5)
        return None

    def _spot_schema(self):
        return {
            "e": pl.Utf8,
            "lastUpdateId": pl.UInt64,
            "E": pl.UInt64,
            "U": pl.UInt64,
            "u": pl.UInt64,
            "bids": pl.Utf8,
            "asks": pl.Utf8,
        }

    def _handle_day_rollover(
        self,
        symbol: str,
        record_day: date,
        active_day: dict[str, date | None],
    ) -> None:
        prev_day = active_day[symbol]

        if prev_day is None:
            active_day[symbol] = record_day
            return

        if record_day == prev_day:
            return

        # Flush pending rows first so the previous day's parquet is complete.
        self._flush_symbol_buffer(symbol)

        prev_day_path = self._get_file_path_for_day(symbol, prev_day)
        if prev_day_path.exists():
            task = asyncio.create_task(
                self.upload_to_huggingface(prev_day_path, delete_after_upload=True)
            )
            self._upload_tasks.add(task)
            task.add_done_callback(self._upload_tasks.discard)

        active_day[symbol] = record_day

    def _flush_symbol_buffer(self, symbol: str) -> Path | None:
        if not self.buffers[symbol]:
            return None

        ts = self.buffers[symbol][0].get("E") or int(time.time() * 1000)
        out_path = self._get_file_path(symbol, ts)
        schema = self._spot_schema()
        df = pl.DataFrame(self.buffers[symbol], schema=schema)

        if out_path.exists():
            df_existing = pl.read_parquet(out_path, schema=schema)
            df = pl.concat([df_existing, df])

        df.write_parquet(out_path, compression="zstd", compression_level=19)
        logging.debug(f"Wrote {len(self.buffers[symbol])} records to {out_path}")
        self.buffers[symbol].clear()
        return out_path

    async def upload_to_huggingface(self, file_path: Path, delete_after_upload=False):
        if not self.huggingface_token:
            logging.warning("Hugging Face token not found. Skipping upload.")
            return False

        loop = asyncio.get_running_loop()

        for attempt in range(1, self.MAX_UPLOAD_RETRIES + 1):
            try:
                await loop.run_in_executor(
                    None,
                    functools.partial(self._upload_file_sync, file_path),
                )
                logging.info(f"Uploaded {file_path} to Hugging Face.")
                if delete_after_upload:
                    file_path.unlink()
                    logging.debug(f"Deleted local file {file_path} after upload.")
                return True
            except Exception as e:
                logging.warning(
                    f"[Attempt {attempt}/{self.MAX_UPLOAD_RETRIES}] "
                    f"Failed to upload {file_path}: {e}"
                )
                if attempt < self.MAX_UPLOAD_RETRIES:
                    await asyncio.sleep(self.UPLOAD_RETRY_DELAY * attempt)

        logging.error(
            f"Giving up uploading {file_path} after {self.MAX_UPLOAD_RETRIES} attempts."
        )
        return False

    def _upload_file_sync(self, file_path: Path):
        """Synchronous upload, safe to call from run_in_executor."""
        self._hf_login()
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=str(file_path.relative_to(self.output_dir)),
            repo_id="predict-quant/binance-spot-orderbook",
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
                    logging.info(f"[MIGRATE] Moved {f.name} -> {new_path}")
                except Exception as e:
                    logging.error(f"[MIGRATE] Failed to migrate {f}: {e}")

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
                logging.info(f"[RETRY] Uploading leftover file {f}")
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    def _get_file_path_for_day(self, symbol: str, day: date) -> Path:
        out_path: Path = (
            self.output_dir
            / symbol
            / f"{day.year:04d}"
            / f"{day.month:02d}"
            / f"{day.isoformat()}_{symbol}_depth{self.depth}.parquet"
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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    client = BinanceSpotOrderBookDumper(
        args.symbols, args.depth, args.output, args.batch_size
    )
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
