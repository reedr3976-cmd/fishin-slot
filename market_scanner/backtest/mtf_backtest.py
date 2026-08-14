"""Multi-timeframe filter backtest helpers (no look-ahead)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import BACKTEST_WARMUP_BARS, FORWARD_BARS, ROUND_TRIP_COST, SMA_SLOW
from models import CandleSeries
from scanner.opportunity import evaluate_opportunity
from scanner.mtf_filter import directions_agree
from scanner.scoring import ORIGINAL_RULES, ScoringRules
from backtest.engine import BacktestRun, _slice_series
from backtest.metrics import TradeResult


def _idx_at_or_before(series: CandleSeries, ts: int) -> Optional[int]:
    if len(series) == 0:
        return None
    idxs = np.where(series.timestamps <= ts)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[-1])


def _direction_at_ts(
    series: CandleSeries,
    ts: int,
    rules: ScoringRules,
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Evaluate direction using only bars with timestamp <= ts."""
    idx = _idx_at_or_before(series, ts)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    if idx is None or idx < warmup:
        return None, idx, None
    hist = _slice_series(series, idx)
    opp = evaluate_opportunity(hist, series.instrument, rules=rules)
    if opp.direction in ("bullish", "bearish"):
        return opp.direction, idx, opp.score
    return None, idx, opp.score


def backtest_series_mtf(
    primary: CandleSeries,
    confirm: CandleSeries,
    rules: ScoringRules | None = None,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
) -> tuple[list[TradeResult], dict[str, int]]:
    """Walk-forward backtest with MTF confirmation for MEDIUM/HIGH only.

    LOW signals are kept without confirmation.
    MEDIUM/HIGH require confirm timeframe direction to agree (no look-ahead).
    """
    rules = rules or ORIGINAL_RULES
    horizon = FORWARD_BARS.get(primary.timeframe)
    if horizon is None:
        return [], {}

    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    cost = ROUND_TRIP_COST.get(primary.asset_class, 0.001)
    trades: list[TradeResult] = []
    stats = {
        "candidates_medium_high": 0,
        "suppressed_disagree": 0,
        "suppressed_missing": 0,
        "kept_agree": 0,
        "low_kept": 0,
    }

    i = max(warmup, start_idx or warmup)
    n = len(primary)
    last_start = n - horizon if end_idx_exclusive is None else min(n - horizon, end_idx_exclusive)

    while i < last_start:
        hist = _slice_series(primary, i)
        opp = evaluate_opportunity(hist, primary.instrument, rules=rules)
        if not (
            opp.confidence in ("HIGH", "MEDIUM", "LOW")
            and opp.direction in ("bullish", "bearish")
        ):
            i += 1
            continue

        mtf_status = "n/a"
        take = True
        if opp.confidence in ("HIGH", "MEDIUM"):
            stats["candidates_medium_high"] += 1
            conf_dir, _, _ = _direction_at_ts(
                confirm, int(primary.timestamps[i]), rules
            )
            if conf_dir is None:
                stats["suppressed_missing"] += 1
                mtf_status = "suppressed_missing"
                take = False
            elif directions_agree(opp.direction, conf_dir):
                stats["kept_agree"] += 1
                mtf_status = "agreed"
                take = True
            else:
                stats["suppressed_disagree"] += 1
                mtf_status = "suppressed_disagree"
                take = False
        else:
            stats["low_kept"] += 1

        if not take:
            i += 1
            continue

        entry = float(primary.close[i])
        exit_px = float(primary.close[i + horizon])
        if opp.direction == "bullish":
            gross = (exit_px - entry) / entry
        else:
            gross = (entry - exit_px) / entry
        net = gross - cost
        trades.append(
            TradeResult(
                instrument=primary.instrument,
                asset_class=primary.asset_class,
                timeframe=primary.timeframe,
                confidence=opp.confidence,
                direction=opp.direction,
                score=opp.score,
                entry_idx=i,
                exit_idx=i + horizon,
                entry_ts=int(primary.timestamps[i]),
                exit_ts=int(primary.timestamps[i + horizon]),
                entry_price=entry,
                exit_price=exit_px,
                gross_return=gross,
                cost=cost,
                net_return=net,
                win=net > 0,
                feature_flags=dict(opp.feature_flags),
                rules_name=f"{rules.name}+mtf",
                mtf_status=mtf_status,
            )
        )
        i += horizon

    return trades, stats


def run_mtf_backtest_on_map(
    series_map: dict[tuple[str, str], CandleSeries],
    rules: ScoringRules,
    *,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    mode: str = "public_historical",
    errors: Optional[list[str]] = None,
) -> tuple[BacktestRun, dict[str, int]]:
    """Backtest with MTF filter across instruments that have both 1d and 1wk."""
    trades: list[TradeResult] = []
    totals = {
        "candidates_medium_high": 0,
        "suppressed_disagree": 0,
        "suppressed_missing": 0,
        "kept_agree": 0,
        "low_kept": 0,
    }
    bars_scanned = 0
    instruments = sorted({k for k, _ in series_map})

    for key in instruments:
        d1 = series_map.get((key, "1d"))
        wk = series_map.get((key, "1wk"))
        if d1 is None or wk is None:
            continue
        for primary, confirm in ((d1, wk), (wk, d1)):
            bars_scanned += len(primary)
            n = len(primary)
            start_idx = int(n * start_frac)
            end_idx = int(n * end_frac)
            tlist, stats = backtest_series_mtf(
                primary,
                confirm,
                rules,
                start_idx=start_idx,
                end_idx_exclusive=end_idx,
            )
            trades.extend(tlist)
            for k, v in stats.items():
                totals[k] += v

    run = BacktestRun(
        trades=trades,
        errors=list(errors or []),
        bars_scanned=bars_scanned,
        instruments=instruments,
        timeframes=["1d", "1wk"],
        mode=mode,
        rules_name=f"{rules.name}+mtf",
    )
    return run, totals
