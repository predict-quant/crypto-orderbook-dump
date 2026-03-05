# --- Class-based client with robust 24/7 operation ---
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import polars as pl
import websockets


class BinanceOrderBookDumper:
    BASE_URL = "wss://fstream.binance.com"
    MAX_CONN_HOURS = 23.5  # reconnect before 24h forced disconnect
    PING_INTERVAL = 60 * 2  # send pong every 2 minutes
    MAX_MSGS_PER_SEC = 10

    def __init__(self, symbols, depth, output_dir, batch_size):
        self.symbols = symbols
        self.depth = depth
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.buffers = {symbol: [] for symbol in symbols}
        self.output_paths = {
            symbol: Path(output_dir) / f"{symbol}_depth{depth}.parquet"
            for symbol in symbols
        }
        for path in self.output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
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
        async with websockets.connect(url, ping_interval=None) as ws:
            print("WebSocket connection established.")

            pong_task = asyncio.create_task(self._send_pong(ws))

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
                    "E": payload.get("E"),
                    "T": payload.get("T"),
                    "u": payload.get("u"),
                    "pu": payload.get("pu"),
                    "bids": json.dumps(payload.get("b", [])),
                    "asks": json.dumps(payload.get("a", [])),
                }
                self.buffers[symbol].append(record)
                if len(self.buffers[symbol]) >= self.batch_size:
                    df = pl.DataFrame(self.buffers[symbol])
                    out_path = self.output_paths[symbol]
                    if out_path.exists():
                        df_existing = pl.read_parquet(out_path)
                        df = pl.concat([df_existing, df])
                    df.write_parquet(out_path, compression="zstd", compression_level=19)
                    print(f"Wrote {len(self.buffers[symbol])} records to {out_path}")
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
                    return data
                else:
                    print(f"Failed to get snapshot for {symbol}: {resp.status}")
                    return None

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
