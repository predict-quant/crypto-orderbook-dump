import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import Mock

from crypto_orderbook_dump.binance_future import BinanceOrderBookDumper
from crypto_orderbook_dump.binance_spot import BinanceSpotOrderBookDumper


class _DummyTask:
    def add_done_callback(self, _callback):
        return None


def _patch_create_task(monkeypatch):
    calls = []

    def fake_create_task(coro):
        calls.append(coro)
        # Avoid un-awaited coroutine warnings in tests.
        coro.close()
        return _DummyTask()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    return calls


def _assert_day_path(path: Path, root: Path, symbol: str, day: date, depth: int):
    assert path == (
        root
        / symbol
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.isoformat()}_{symbol}_depth{depth}.parquet"
    )


def test_spot_day_path(tmp_path):
    dumper = BinanceSpotOrderBookDumper(
        symbols=["BTCUSDT"],
        depth=20,
        output_dir=tmp_path,
        batch_size=1000,
    )

    day = date(2026, 5, 14)
    path = dumper._get_file_path_for_day("BTCUSDT", day)
    _assert_day_path(path, tmp_path, "BTCUSDT", day, 20)


def test_future_day_path(tmp_path):
    dumper = BinanceOrderBookDumper(
        symbols=["BTCUSDT"],
        depth=20,
        output_dir=tmp_path,
        batch_size=1000,
    )

    day = date(2026, 5, 14)
    path = dumper._get_file_path_for_day("BTCUSDT", day)
    _assert_day_path(path, tmp_path, "BTCUSDT", day, 20)


def test_spot_rollover_schedules_upload_even_when_buffer_empty(tmp_path, monkeypatch):
    symbol = "BTCUSDT"
    prev_day = date(2026, 5, 14)
    new_day = date(2026, 5, 15)

    dumper = BinanceSpotOrderBookDumper(
        symbols=[symbol],
        depth=20,
        output_dir=tmp_path,
        batch_size=1000,
    )

    active_day = {symbol: prev_day}
    dumper._flush_symbol_buffer = Mock(return_value=None)

    async def fake_upload(_file_path, delete_after_upload=False):
        return True

    dumper.upload_to_huggingface = fake_upload

    prev_day_path = dumper._get_file_path_for_day(symbol, prev_day)
    prev_day_path.parent.mkdir(parents=True, exist_ok=True)
    prev_day_path.touch()

    task_calls = _patch_create_task(monkeypatch)

    dumper._handle_day_rollover(symbol, new_day, active_day)

    dumper._flush_symbol_buffer.assert_called_once_with(symbol)
    assert len(task_calls) == 1
    assert active_day[symbol] == new_day


def test_future_rollover_does_not_schedule_upload_when_prev_file_missing(
    tmp_path, monkeypatch
):
    symbol = "BTCUSDT"
    prev_day = date(2026, 5, 14)
    new_day = date(2026, 5, 15)

    dumper = BinanceOrderBookDumper(
        symbols=[symbol],
        depth=20,
        output_dir=tmp_path,
        batch_size=1000,
    )

    active_day = {symbol: prev_day}
    dumper._flush_symbol_buffer = Mock(return_value=None)

    async def fake_upload(_file_path, delete_after_upload=False):
        return True

    dumper.upload_to_huggingface = fake_upload

    task_calls = _patch_create_task(monkeypatch)

    dumper._handle_day_rollover(symbol, new_day, active_day)

    dumper._flush_symbol_buffer.assert_called_once_with(symbol)
    assert len(task_calls) == 0
    assert active_day[symbol] == new_day
