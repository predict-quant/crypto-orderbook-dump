# crypto-orderbook-dump

Continuously streams Binance order book diffs over WebSocket, reconstructs a consistent local order book following the official Binance sync protocol, and writes the data to daily Parquet files. Completed day files are automatically uploaded to Hugging Face and removed locally.

## Features

- **Spot** (`binance_spot.py`) and **USD-M Futures** (`binance_future.py`) support
- Streams multiple symbols in a single combined WebSocket connection
- Follows the Binance diff-depth sync protocol (snapshot + event buffering + gap detection)
- Rolls over to a new file at UTC midnight and uploads the previous day's file in the background
- On startup, retries any leftover files that were not uploaded in a previous run
- Reconnects automatically on any error or after 23.5 hours (before Binance's forced 24 h disconnect)

## Requirements

- Python 3.12
- Dependencies are listed in `pyproject.toml`:
  - `aiohttp`, `websockets`, `polars`, `huggingface-hub`, `python-dotenv`

Install:

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```
HUGGINGFACE_HUB_TOKEN=hf_...
```

If the token is not set, data is still written locally — Hugging Face uploads are silently skipped.

## Usage

### Spot

```bash
python -m crypto_orderbook_dump.binance_spot \
  --symbols BTCUSDT,ETHUSDT \
  --depth 20 \
  --output data/parquet/binance_spot_ws \
  --batch-size 1000
```

| Argument | Default | Description |
|---|---|---|
| `--symbols` | *(required)* | Comma-separated trading pairs, e.g. `BTCUSDT,ETHUSDT` |
| `--depth` | — | Snapshot depth for the REST endpoint (5, 10, 20, 100, 500, 1000, 5000) |
| `--output` | `data/parquet/binance_spot_ws` | Root directory for output files |
| `--batch-size` | `1000` | Number of records buffered before flushing to Parquet |

### USD-M Futures

```bash
python -m crypto_orderbook_dump.binance_future \
  --symbols BTCUSDT,ETHUSDT \
  --depth 20 \
  --output data/parquet/binance_ws \
  --batch-size 1000
```

| Argument | Default | Description |
|---|---|---|
| `--symbols` | *(required)* | Comma-separated perpetual futures pairs |
| `--depth` | `20` | Stream and snapshot depth (5, 10, 20, 100) |
| `--output` | `data/parquet/binance_ws` | Root directory for output files |
| `--batch-size` | `1000` | Number of records buffered before flushing to Parquet |

## Output format

Files are written under `{output}/{SYMBOL}/{YYYY-MM-DD}_{SYMBOL}_depth{depth}.parquet`, for example:

```
data/parquet/binance_spot_ws/BTCUSDT/2026-04-11_BTCUSDT_depth20.parquet
data/parquet/binance_ws/BTCUSDT/2026-04-11_BTCUSDT_depth20.parquet
```

Files are compressed with Zstandard (level 19).

### Spot schema

| Column | Type | Description |
|---|---|---|
| `e` | `Utf8` | Event type (`depthUpdate` or `snapshot`) |
| `lastUpdateId` | `UInt64` | Snapshot `lastUpdateId`; `null` for diff events |
| `E` | `UInt64` | Event time (Unix ms); `null` for the snapshot row |
| `U` | `UInt64` | First update ID in event; `null` for the snapshot row |
| `u` | `UInt64` | Final update ID in event; `null` for the snapshot row |
| `bids` | `Utf8` | JSON array of `[price, qty]` pairs |
| `asks` | `Utf8` | JSON array of `[price, qty]` pairs |

### USD-M Futures schema

| Column | Type | Description |
|---|---|---|
| `e` | `Utf8` | Event type (`depthUpdate` or `snapshot`) |
| `lastUpdateId` | `UInt64` | Snapshot `lastUpdateId`; `null` for diff events |
| `E` | `UInt64` | Event time (Unix ms) |
| `T` | `UInt64` | Transaction time (Unix ms) |
| `U` | `UInt64` | First update ID in event |
| `u` | `UInt64` | Final update ID in event |
| `pu` | `UInt64` | Previous final update ID (used for gap detection) |
| `bids` | `Utf8` | JSON array of `[price, qty]` pairs |
| `asks` | `Utf8` | JSON array of `[price, qty]` pairs |

Each file begins with a snapshot row (`e = "snapshot"`) followed by consecutive diff-depth events. To reconstruct the order book at any point, start from the snapshot and apply each diff in order.

## Sync protocol

Both dumpers follow the Binance documentation for maintaining a consistent local order book:

**Spot** ([docs](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#how-to-manage-a-local-order-book-correctly)):
1. Events are buffered before the REST snapshot is fetched.
2. A snapshot is fetched and validated: `lastUpdateId` must be ≥ the `U` of the first buffered event. Otherwise the snapshot is retried (up to 5 times).
3. Buffered events with `u ≤ lastUpdateId` are discarded.
4. For subsequent events, `U` of each event must equal `u` of the previous event + 1. A gap triggers an immediate re-sync.

**USD-M Futures** ([docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly)):
1. A REST snapshot is fetched; events with `u ≤ lastUpdateId` are dropped.
2. The first valid event must satisfy `U ≤ lastUpdateId ≤ u`.
3. For subsequent events, `pu` must equal the previous event's `u`. A mismatch triggers a re-sync.
4. Proactive pong frames are sent every 2 minutes to keep the connection alive.

## Hugging Face upload

Completed day files are uploaded to:

- Spot: [`predict-quant/binance-spot-orderbook`](https://huggingface.co/datasets/predict-quant/binance-spot-orderbook)
- Futures: [`predict-quant/binance-orderbook`](https://huggingface.co/datasets/predict-quant/binance-orderbook)

The local file is deleted after a successful upload. Failed uploads are retried up to 3 times with exponential back-off. On the next startup, any files that were not uploaded are retried automatically.
