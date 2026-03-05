# --- Class-based client with robust 24/7 operation ---
import argparse
import asyncio
import json
import time
from datetime import date
from os import PathLike
from pathlib import Path

import aiohttp
import polars as pl
import websockets


class BinanceOrderBookDumper:
    BASE_URL = "wss://fstream.binance.com"
    MAX_CONN_HOURS = 23.5  # reconnect before 24h forced disconnect
    PING_INTERVAL = 60 * 2  # send pong every 2 minutes
    MAX_MSGS_PER_SEC = 10

    def __init__(
        self, symbols: list[str], depth: int, output_dir: PathLike, batch_size: int
    ):
        self.symbols = symbols
        self.depth = depth
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.buffers = {symbol: [] for symbol in symbols}
        self.output_dir = Path(output_dir)
        self.depth = depth
        # Output paths will be generated per batch using timestamp
        self._stop = False

    def make_stream_name(self, symbol):
        return f"{symbol.lower()}@depth{self.depth}@100ms"

    def build_url(self):
        streams = "/".join([self.make_stream_name(s) for s in self.symbols])
        url = f"{self.BASE_URL}/stream?streams={streams}"
        return url

    async def run(self):
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
            print("WebSocket connection established.")

            asyncio.create_task(self._send_pong(ws))

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
                    df = pl.DataFrame(self.buffers[symbol])
                    if out_path.exists():
                        df_existing = pl.read_parquet(out_path)
                        df = pl.concat([df_existing, df])
                    df.write_parquet(out_path, compression="zstd", compression_level=19)
                    print(f"Wrote {len(self.buffers[symbol])} records to {out_path}")
                    # Reset buffer and snapshot for next batch
                    last_records[symbol] = None
                    snapshots[symbol] = None
                    self.buffers[symbol].clear()

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
