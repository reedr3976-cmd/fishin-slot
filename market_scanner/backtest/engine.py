"""Look-ahead-safe historical backtest of scanner confidence rules.

Each historical signal is scored using only bars available at that time
(series sliced to [:i+1]). Rules are injected via ScoringRules.
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
from scanner.scoring import ORIGINAL_RULES, ScoringRules
from backtest.metrics import TradeResult, group_metrics


@dataclass
class BacktestRun:
    trades: list[TradeResult]
    errors: list[str]
    bars_scanned: int
    instruments: list[str]
    timeframes: list[str]
    mode: str
    rules_name: str = "original"


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


def backtest_series(
    series: CandleSeries,
    rules: ScoringRules | None = None,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
) -> list[TradeResult]:
    """Walk forward through one series with no look-ahead bias.

    Optional start_idx / end_idx_exclusive restrict where NEW signals may start
    (for chronological train/test splits). Exit may use bars beyond end for the
    hold period only (realized outcome), which is standard and not feature leakage.
    """
    rules = rules or ORIGINAL_RULES
    horizon = FORWARD_BARS.get(series.timeframe)
    if horizon is None:
        return []

    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    cost = ROUND_TRIP_COST.get(series.asset_class, 0.001)
    trades: list[TradeResult] = []

    i = max(warmup, start_idx or warmup)
    n = len(series)
    last_start = n - horizon if end_idx_exclusive is None else min(n - horizon, end_idx_exclusive)
    while i < last_start:
        hist = _slice_series(series, i)
        opp = evaluate_opportunity(hist, series.instrument, rules=rules)
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
                    feature_flags=dict(opp.feature_flags),
                    rules_name=rules.name,
                )
            )
            i += horizon
        else:
            i += 1
    return trades


def load_series_map(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
) -> tuple[dict[tuple[str, str], CandleSeries], list[str], int]:
    """Fetch all series once for reuse across original/revised comparisons."""
    keys = list(instruments) if instruments else list(INSTRUMENTS.keys())
    tfs = list(timeframes) if timeframes else list(BACKTEST_DEFAULT_TIMEFRAMES)
    session = requests.Session()
    series_map: dict[tuple[str, str], CandleSeries] = {}
    errors: list[str] = []
    bars = 0
    for key in keys:
        if key not in INSTRUMENTS:
            errors.append(f"Unknown instrument: {key}")
            continue
        for tf in tfs:
            try:
                print(f"  loading {key} {tf}...", flush=True)
                series = fetch_instrument(
                    key, tf, demo=demo, session=session, for_backtest=True
                )
                series_map[(key, tf)] = series
                bars += len(series)
            except DataFetchError as exc:
                errors.append(f"{key} {tf}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{key} {tf}: unexpected {type(exc).__name__}: {exc}")
    return series_map, errors, bars


def run_backtest_on_map(
    series_map: dict[tuple[str, str], CandleSeries],
    rules: ScoringRules,
    *,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    mode: str = "public_historical",
    errors: Optional[list[str]] = None,
) -> BacktestRun:
    """Backtest rules on a preloaded map with chronological index fraction window."""
    trades: list[TradeResult] = []
    bars_scanned = 0
    instruments = sorted({k for k, _ in series_map})
    timeframes = sorted({t for _, t in series_map})
    for (key, tf), series in series_map.items():
        bars_scanned += len(series)
        n = len(series)
        start_idx = int(n * start_frac)
        end_idx = int(n * end_frac)
        trades.extend(
            backtest_series(
                series, rules, start_idx=start_idx, end_idx_exclusive=end_idx
            )
        )
    return BacktestRun(
        trades=trades,
        errors=list(errors or []),
        bars_scanned=bars_scanned,
        instruments=instruments,
        timeframes=timeframes,
        mode=mode,
        rules_name=rules.name,
    )


def run_backtest(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
    rules: ScoringRules | None = None,
) -> BacktestRun:
    rules = rules or ORIGINAL_RULES
    series_map, errors, _ = load_series_map(instruments, timeframes, demo=demo)
    return run_backtest_on_map(
        series_map,
        rules,
        mode="demo" if demo else "public_historical",
        errors=errors,
    )


def run_backtest_with_metrics(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
    rules: ScoringRules | None = None,
) -> tuple[BacktestRun, dict]:
    run = run_backtest(instruments, timeframes, demo=demo, rules=rules)
    metrics = group_metrics(run.trades)
    return run, metrics
