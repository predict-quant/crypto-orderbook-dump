module.exports = {
  apps: [
    {
      name: "binance-spot-orderbook-dump",
      script:
        "uv run python crypto_orderbook_dump/binance_spot.py --symbols=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,HYPEUSDT,BNBUSDT",
    },
  ],
};
