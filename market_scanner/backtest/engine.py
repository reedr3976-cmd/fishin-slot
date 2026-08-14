"""Look-ahead-safe historical backtest of existing scanner confidence rules.

Rules are NOT modified to improve results. Each historical signal is scored
using only bars available at that point in time (series sliced to [:i+1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import requests

from config import (
    BACKTEST_DEFAULT_TIMEFRAMES,
    BACKTEST_WARMUP_BARS,
    FORWARD_BARS,
    INSTRUMENTS,
    ROUND_TRIP_COST,
    SMA_SLOW,
)
from models import CandleSeries
from providers.yahoo import DataFetchError, fetch_instrument
from scanner.opportunity import evaluate_opportunity
from backtest.metrics import TradeResult, group_metrics


@dataclass
class BacktestRun:
    trades: list[TradeResult]
    errors: list[str]
    bars_scanned: int
    instruments: list[str]
    timeframes: list[str]
    mode: str


def _slice_series(series: CandleSeries, end_inclusive: int) -> CandleSeries:
    """Return bars [0 .. end_inclusive] only — no future bars."""
    n = end_inclusive + 1
    return CandleSeries(
        instrument=series.instrument,
        symbol=series.symbol,
        asset_class=series.asset_class,
        timeframe=series.timeframe,
        timestamps=series.timestamps[:n],
        open=series.open[:n],
        high=series.high[:n],
        low=series.low[:n],
        close=series.close[:n],
        volume=series.volume[:n],
    )


def backtest_series(series: CandleSeries) -> list[TradeResult]:
    """Walk forward through one series with no look-ahead bias.

    Signal rules match the live scanner via evaluate_opportunity().
    Non-overlapping: after a signal, skip until the forward horizon ends.
    """
    horizon = FORWARD_BARS.get(series.timeframe)
    if horizon is None:
        return []

    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    cost = ROUND_TRIP_COST.get(series.asset_class, 0.001)
    trades: list[TradeResult] = []

    i = warmup
    n = len(series)
    while i < n - horizon:
        hist = _slice_series(series, i)
        opp = evaluate_opportunity(hist, series.instrument)
        if (
            opp.confidence in ("HIGH", "MEDIUM", "LOW")
            and opp.direction in ("bullish", "bearish")
        ):
            entry = float(series.close[i])
            exit_px = float(series.close[i + horizon])
            if opp.direction == "bullish":
                gross = (exit_px - entry) / entry
            else:
                gross = (entry - exit_px) / entry
            net = gross - cost
            trades.append(
                TradeResult(
                    instrument=series.instrument,
                    asset_class=series.asset_class,
                    timeframe=series.timeframe,
                    confidence=opp.confidence,
                    direction=opp.direction,
                    score=opp.score,
                    entry_idx=i,
                    exit_idx=i + horizon,
                    entry_ts=int(series.timestamps[i]),
                    exit_ts=int(series.timestamps[i + horizon]),
                    entry_price=entry,
                    exit_price=exit_px,
                    gross_return=gross,
                    cost=cost,
                    net_return=net,
                    win=net > 0,
                )
            )
            i += horizon  # non-overlapping next opportunity
        else:
            i += 1
    return trades


def run_backtest(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
) -> BacktestRun:
    keys = list(instruments) if instruments else list(INSTRUMENTS.keys())
    tfs = list(timeframes) if timeframes else list(BACKTEST_DEFAULT_TIMEFRAMES)
    session = requests.Session()
    trades: list[TradeResult] = []
    errors: list[str] = []
    bars_scanned = 0

    for key in keys:
        if key not in INSTRUMENTS:
            errors.append(f"Unknown instrument: {key}")
            continue
        for tf in tfs:
            try:
                print(f"  backtesting {key} {tf}...", flush=True)
                series = fetch_instrument(
                    key, tf, demo=demo, session=session, for_backtest=True
                )
                bars_scanned += len(series)
                trades.extend(backtest_series(series))
            except DataFetchError as exc:
                errors.append(f"{key} {tf}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{key} {tf}: unexpected {type(exc).__name__}: {exc}")

    return BacktestRun(
        trades=trades,
        errors=errors,
        bars_scanned=bars_scanned,
        instruments=keys,
        timeframes=tfs,
        mode="demo" if demo else "public_historical",
    )


def run_backtest_with_metrics(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
) -> tuple[BacktestRun, dict]:
    run = run_backtest(instruments, timeframes, demo=demo)
    metrics = group_metrics(run.trades)
    return run, metrics
