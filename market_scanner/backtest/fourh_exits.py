"""Exit-policy simulations for analysis-only studies (SL / TP / fixed hold).

Collect ORIGINAL-rules entries once, then realize different exits on the
same entry set so stop/target experiments stay comparable and fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import BACKTEST_WARMUP_BARS, FORWARD_BARS, ROUND_TRIP_COST, SMA_SLOW
from models import CandleSeries
from scanner.opportunity import evaluate_opportunity
from scanner.scoring import ORIGINAL_RULES, ScoringRules
from backtest.engine import _slice_series
from backtest.metrics import TradeResult


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    mode: str  # fixed | stop | target
    atr_mult: float = 0.0
    horizon_bars: Optional[int] = None


@dataclass
class EntrySignal:
    instrument: str
    asset_class: str
    timeframe: str
    confidence: str
    direction: str
    score: int
    entry_idx: int
    entry_ts: int
    entry_price: float
    atr_at_entry: Optional[float]
    feature_flags: dict[str, int]
    rules_name: str


FIXED_HOLD = ExitPolicy(name="fixed_hold", mode="fixed")
STOP_1_0_ATR = ExitPolicy(name="stop_1.0atr", mode="stop", atr_mult=1.0)
STOP_1_5_ATR = ExitPolicy(name="stop_1.5atr", mode="stop", atr_mult=1.5)
STOP_2_0_ATR = ExitPolicy(name="stop_2.0atr", mode="stop", atr_mult=2.0)
TP_1_5_ATR = ExitPolicy(name="tp_1.5atr", mode="target", atr_mult=1.5)
TP_2_0_ATR = ExitPolicy(name="tp_2.0atr", mode="target", atr_mult=2.0)
TP_3_0_ATR = ExitPolicy(name="tp_3.0atr", mode="target", atr_mult=3.0)

ENTRY_STOP_POLICIES = (FIXED_HOLD, STOP_1_0_ATR, STOP_1_5_ATR, STOP_2_0_ATR)
ENTRY_TP_POLICIES = (FIXED_HOLD, TP_1_5_ATR, TP_2_0_ATR, TP_3_0_ATR)


def _signed_move(direction: str, entry: float, exit_px: float) -> float:
    if direction == "bullish":
        return (exit_px - entry) / entry
    return (entry - exit_px) / entry


def simulate_exit(
    series: CandleSeries,
    entry_idx: int,
    direction: str,
    atr: Optional[float],
    policy: ExitPolicy,
) -> tuple[int, float, str, Optional[float]]:
    """Return (exit_idx, exit_price, reason, r_multiple)."""
    horizon = policy.horizon_bars or FORWARD_BARS.get(series.timeframe, 4)
    last = min(entry_idx + horizon, len(series) - 1)
    entry = float(series.close[entry_idx])
    risk = (policy.atr_mult * atr) if (atr is not None and policy.atr_mult > 0) else None

    if policy.mode == "fixed" or risk is None or risk <= 0:
        exit_idx = last
        exit_px = float(series.close[exit_idx])
        return exit_idx, exit_px, "fixed_hold", None

    if direction == "bullish":
        stop = entry - risk
        target = entry + risk
    else:
        stop = entry + risk
        target = entry - risk

    for j in range(entry_idx + 1, last + 1):
        hi = float(series.high[j])
        lo = float(series.low[j])
        if policy.mode == "stop":
            if direction == "bullish" and lo <= stop:
                r_mult = _signed_move(direction, entry, stop) / (risk / entry)
                return j, stop, "stop", float(r_mult)
            if direction == "bearish" and hi >= stop:
                r_mult = _signed_move(direction, entry, stop) / (risk / entry)
                return j, stop, "stop", float(r_mult)
        elif policy.mode == "target":
            if direction == "bullish" and hi >= target:
                r_mult = _signed_move(direction, entry, target) / (risk / entry)
                return j, target, "target", float(r_mult)
            if direction == "bearish" and lo <= target:
                r_mult = _signed_move(direction, entry, target) / (risk / entry)
                return j, target, "target", float(r_mult)

    exit_px = float(series.close[last])
    r_mult = _signed_move(direction, entry, exit_px) / (risk / entry)
    return last, exit_px, "timeout", float(r_mult)


def collect_entries(
    series: CandleSeries,
    rules: ScoringRules | None = None,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    require_trending: bool = False,
) -> list[EntrySignal]:
    """Walk-forward ORIGINAL entries with fixed-horizon non-overlapping schedule."""
    rules = rules or ORIGINAL_RULES
    horizon = FORWARD_BARS.get(series.timeframe)
    if horizon is None:
        return []

    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    entries: list[EntrySignal] = []
    i = max(warmup, start_idx or warmup)
    n = len(series)
    last_start = (
        n - horizon if end_idx_exclusive is None else min(n - horizon, end_idx_exclusive)
    )

    while i < last_start:
        hist = _slice_series(series, i)
        opp = evaluate_opportunity(hist, series.instrument, rules=rules)
        if not (
            opp.confidence in ("HIGH", "MEDIUM", "LOW")
            and opp.direction in ("bullish", "bearish")
        ):
            i += 1
            continue
        if require_trending and int((opp.feature_flags or {}).get("sma_stack", 0)) != 1:
            i += 1
            continue

        entries.append(
            EntrySignal(
                instrument=series.instrument,
                asset_class=series.asset_class,
                timeframe=series.timeframe,
                confidence=opp.confidence,
                direction=opp.direction,
                score=opp.score,
                entry_idx=i,
                entry_ts=int(series.timestamps[i]),
                entry_price=float(series.close[i]),
                atr_at_entry=float(opp.atr) if opp.atr is not None else None,
                feature_flags=dict(opp.feature_flags),
                rules_name=rules.name,
            )
        )
        i += horizon
    return entries


def realize_entries(
    series: CandleSeries,
    entries: list[EntrySignal],
    exit_policy: ExitPolicy = FIXED_HOLD,
) -> list[TradeResult]:
    """Apply an exit policy to a precomputed entry list (same entries)."""
    cost = ROUND_TRIP_COST.get(series.asset_class, 0.001)
    trades: list[TradeResult] = []
    for e in entries:
        exit_idx, exit_px, reason, r_mult = simulate_exit(
            series, e.entry_idx, e.direction, e.atr_at_entry, exit_policy
        )
        gross = _signed_move(e.direction, e.entry_price, exit_px)
        net = gross - cost
        trades.append(
            TradeResult(
                instrument=e.instrument,
                asset_class=e.asset_class,
                timeframe=e.timeframe,
                confidence=e.confidence,
                direction=e.direction,
                score=e.score,
                entry_idx=e.entry_idx,
                exit_idx=exit_idx,
                entry_ts=e.entry_ts,
                exit_ts=int(series.timestamps[exit_idx]),
                entry_price=e.entry_price,
                exit_price=exit_px,
                gross_return=gross,
                cost=cost,
                net_return=net,
                win=net > 0,
                feature_flags=dict(e.feature_flags),
                rules_name=e.rules_name,
                atr_at_entry=e.atr_at_entry,
                exit_reason=reason,
                r_multiple=r_mult,
            )
        )
    return trades


def backtest_series_exits(
    series: CandleSeries,
    rules: ScoringRules | None = None,
    *,
    exit_policy: ExitPolicy = FIXED_HOLD,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    require_trending: bool = False,
) -> list[TradeResult]:
    entries = collect_entries(
        series,
        rules,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_exclusive,
        require_trending=require_trending,
    )
    return realize_entries(series, entries, exit_policy)
