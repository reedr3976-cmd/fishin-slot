"""Scanner V2 — RESEARCH / BACKTEST ONLY.

Two-stage trend architecture (regime → trigger) with adaptive exits and
normalized risk. Does not modify or replace the live ORIGINAL scanner.

Stages (global params; no per-symbol optimization):
  ORIGINAL          : live score rules, MH actionable, fixed 4-bar hold
  V2_S1_REGIME_HOLD : structure+MA+ADX regime + pullback, fixed hold
  V2_S2_ADAPTIVE    : same entries + ATR stop / trail / structure-break exit
  V2_S3_DUAL_TRIG   : S2 + breakout-continuation trigger

All stages use 1% equity risk sizing (ATR stop distance as risk unit) so
Forex / stocks / commodities are comparable.

Performance: indicators and ORIGINAL scores are computed once per series
(causal precompute — no look-ahead), then walk-forward windows filter by
entry index and rescale costs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    FORWARD_BARS,
    ROUND_TRIP_COST,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SMA_SLOW,
    V2_ADX_MIN,
    V2_ATR_STOP_MULT,
    V2_MAX_HOLD_BARS,
    V2_RISK_FRACTION,
)
from indicators import compute_all, last_confirmed_swing, swing_structure_dir
from models import CandleSeries
from scanner.scoring import ORIGINAL_RULES, ScoringRules
from scanner.setups import _crossed_down, _crossed_up
from backtest.metrics import TradeResult


@dataclass(frozen=True)
class V2Stage:
    name: str
    use_original_score: bool = False
    allow_pullback: bool = True
    allow_breakout: bool = False
    exit_mode: str = "fixed"  # fixed | adaptive
    notes: str = ""


STAGE_ORIGINAL = V2Stage(
    name="ORIGINAL",
    use_original_score=True,
    exit_mode="fixed",
    notes="Control: ORIGINAL MH score signals + fixed FORWARD_BARS hold.",
)
STAGE_S1 = V2Stage(
    name="V2_S1_REGIME_HOLD",
    allow_pullback=True,
    allow_breakout=False,
    exit_mode="fixed",
    notes="Regime (structure+MA slope+ADX+vol) + pullback; fixed hold (entry isolate).",
)
STAGE_S2 = V2Stage(
    name="V2_S2_ADAPTIVE",
    allow_pullback=True,
    allow_breakout=False,
    exit_mode="adaptive",
    notes="S1 entries + ATR initial stop, ATR trail, structure-break exit.",
)
STAGE_S3 = V2Stage(
    name="V2_S3_DUAL_TRIG",
    allow_pullback=True,
    allow_breakout=True,
    exit_mode="adaptive",
    notes="S2 + breakout-continuation trigger (still global params).",
)

ALL_STAGES: tuple[V2Stage, ...] = (STAGE_ORIGINAL, STAGE_S1, STAGE_S2, STAGE_S3)


@dataclass
class V2Trade(TradeResult):
    stage: str = ""
    trigger: str = ""
    regime: str = ""  # trending | ranging
    risk_frac: float = V2_RISK_FRACTION
    stop_dist_pct: float = 0.0
    gross_r: float = 0.0
    cost_r: float = 0.0
    net_r: float = 0.0
    cost_mult: float = 1.0


def _f(arr: np.ndarray, i: int) -> Optional[float]:
    if i < 0 or i >= len(arr) or np.isnan(arr[i]):
        return None
    return float(arr[i])


def _price_move(direction: str, entry: float, exit_px: float) -> float:
    if direction == "bullish":
        return (exit_px - entry) / entry
    return (entry - exit_px) / entry


def _precompute(series: CandleSeries) -> dict[str, np.ndarray]:
    ind = compute_all(series.close, series.high, series.low)
    n = len(series)
    structure = np.zeros(n, dtype=np.int8)
    for i in range(n):
        structure[i] = swing_structure_dir(series.high, series.low, i)
    atr_arr = ind["atr"]
    atr_pct = np.divide(
        atr_arr, series.close, out=np.full(n, np.nan), where=series.close > 0
    )
    med50 = np.full(n, np.nan)
    for i in range(50, n):
        window = atr_pct[i - 50 : i]
        valid = window[~np.isnan(window)]
        if len(valid):
            med50[i] = float(np.median(valid))
    sma_fast = ind["sma_fast"]
    slope = np.full(n, np.nan)
    slope[3:] = sma_fast[3:] - sma_fast[:-3]
    return {
        **ind,
        "structure": structure,
        "atr_pct": atr_pct,
        "atr_pct_med50": med50,
        "sma_slope": slope,
    }


def _original_signal_at(
    series: CandleSeries,
    feat: dict[str, np.ndarray],
    i: int,
    rules: ScoringRules = ORIGINAL_RULES,
) -> tuple[Optional[str], str, int, dict[str, int]]:
    """Causal ORIGINAL score at bar i using precomputed indicators (no recompute)."""
    if i < max(SMA_SLOW, 35):
        return None, "NO STRONG SETUP", 0, {}

    price = float(series.close[i])
    rsi_now = _f(feat["rsi"], i)
    rsi_prev = _f(feat["rsi"], i - 1)
    sma20 = _f(feat["sma_fast"], i)
    sma50 = _f(feat["sma_slow"], i)
    sma20_prev = _f(feat["sma_fast"], i - 1)
    sma50_prev = _f(feat["sma_slow"], i - 1)
    macd_now = _f(feat["macd"], i)
    macd_sig = _f(feat["macd_signal"], i)
    macd_prev = _f(feat["macd"], i - 1)
    macd_sig_prev = _f(feat["macd_signal"], i - 1)
    bb_u = _f(feat["bb_upper"], i)
    bb_l = _f(feat["bb_lower"], i)

    bull_score = 0
    bear_score = 0
    flags: dict[str, int] = {
        "sma_cross": 0,
        "sma_stack": 0,
        "rsi_extreme": 0,
        "rsi_exit": 0,
        "rsi_mild": 0,
        "macd_cross": 0,
        "macd_strong": 0,
        "macd_mild": 0,
        "bb_touch": 0,
    }
    bull_n = 0
    bear_n = 0

    if _crossed_up(sma20_prev, sma20, sma50_prev, sma50):
        bull_score += rules.sma_cross
        flags["sma_cross"] = 1
        bull_n += 1
    elif _crossed_down(sma20_prev, sma20, sma50_prev, sma50):
        bear_score += rules.sma_cross
        flags["sma_cross"] = 1
        bear_n += 1
    elif sma20 is not None and sma50 is not None:
        if sma20 > sma50 and price > sma20:
            bull_score += rules.sma_stack
            flags["sma_stack"] = 1
            bull_n += 1
        elif sma20 < sma50 and price < sma20:
            bear_score += rules.sma_stack
            flags["sma_stack"] = 1
            bear_n += 1

    if rsi_now is not None:
        if rsi_now <= RSI_OVERSOLD:
            bull_score += rules.rsi_extreme_strong if rsi_now <= 20 else rules.rsi_extreme
            flags["rsi_extreme"] = 1
            bull_n += 1
        elif rsi_now >= RSI_OVERBOUGHT:
            bear_score += rules.rsi_extreme_strong if rsi_now >= 80 else rules.rsi_extreme
            flags["rsi_extreme"] = 1
            bear_n += 1
        elif rsi_prev is not None and rsi_prev <= RSI_OVERSOLD < rsi_now:
            bull_score += rules.rsi_exit
            flags["rsi_exit"] = 1
            bull_n += 1
        elif rsi_prev is not None and rsi_prev >= RSI_OVERBOUGHT > rsi_now:
            bear_score += rules.rsi_exit
            flags["rsi_exit"] = 1
            bear_n += 1
        elif rsi_now >= 55:
            bull_score += rules.rsi_mild
            flags["rsi_mild"] = 1
            bull_n += 1
        elif rsi_now <= 45:
            bear_score += rules.rsi_mild
            flags["rsi_mild"] = 1
            bear_n += 1

    # MACD (mirrors opportunity._macd_condition point assignment)
    if None not in (macd_now, macd_sig):
        if _crossed_up(macd_prev, macd_now, macd_sig_prev, macd_sig):
            bull_score += rules.macd_cross
            flags["macd_cross"] = 1
            bull_n += 1
        elif _crossed_down(macd_prev, macd_now, macd_sig_prev, macd_sig):
            bear_score += rules.macd_cross
            flags["macd_cross"] = 1
            bear_n += 1
        elif macd_now > macd_sig and macd_now > 0:
            bull_score += rules.macd_strong
            flags["macd_strong"] = 1
            bull_n += 1
        elif macd_now < macd_sig and macd_now < 0:
            bear_score += rules.macd_strong
            flags["macd_strong"] = 1
            bear_n += 1
        elif macd_now > macd_sig:
            bull_score += rules.macd_mild
            flags["macd_mild"] = 1
            bull_n += 1
        elif macd_now < macd_sig:
            bear_score += rules.macd_mild
            flags["macd_mild"] = 1
            bear_n += 1

    if bb_l is not None and price <= bb_l:
        bull_score += rules.bb_touch
        flags["bb_touch"] = 1
        bull_n += 1
    elif bb_u is not None and price >= bb_u:
        bear_score += rules.bb_touch
        flags["bb_touch"] = 1
        bear_n += 1

    net = bull_score - bear_score
    if bull_score >= bear_score and bull_score > 0 and net >= rules.min_net_for_direction:
        direction: Optional[str] = "bullish"
        raw = bull_score
        factors = bull_n
        if bear_score >= rules.opposing_penalty_trigger:
            raw = max(0, raw - min(bear_score, rules.opposing_penalty_cap))
    elif bear_score > bull_score and bear_score > 0 and -net >= rules.min_net_for_direction:
        direction = "bearish"
        raw = bear_score
        factors = bear_n
        if bull_score >= rules.opposing_penalty_trigger:
            raw = max(0, raw - min(bull_score, rules.opposing_penalty_cap))
    else:
        return None, "NO STRONG SETUP", 0, flags

    if factors >= rules.confluence_min_factors:
        raw += rules.confluence_bonus
    score = int(max(0, min(100, raw)))
    confidence = rules.confidence_label(score, factor_count=factors)
    if confidence == "NO STRONG SETUP":
        return None, confidence, score, flags
    return direction, confidence, score, flags


def _regime_at(feat: dict, i: int) -> tuple[Optional[str], str]:
    adx_v = feat["adx"][i]
    pdi = feat["plus_di"][i]
    mdi = feat["minus_di"][i]
    sma_f = feat["sma_fast"][i]
    sma_s = feat["sma_slow"][i]
    slope = feat["sma_slope"][i]
    struct = int(feat["structure"][i])
    atr_p = feat["atr_pct"][i]
    med = feat["atr_pct_med50"][i]

    if any(np.isnan(x) for x in (adx_v, pdi, mdi, sma_f, sma_s, slope, atr_p, med)):
        return None, "ranging"
    if med <= 0 or atr_p < 0.6 * med:
        return None, "ranging"
    if adx_v < V2_ADX_MIN:
        return None, "ranging"

    bull = struct == 1 and sma_f > sma_s and slope > 0 and pdi > mdi
    bear = struct == -1 and sma_f < sma_s and slope < 0 and mdi > pdi
    if bull:
        return "bullish", "trending"
    if bear:
        return "bearish", "trending"
    return None, "trending" if adx_v >= V2_ADX_MIN else "ranging"


def _pullback_trigger(series: CandleSeries, feat: dict, i: int, direction: str) -> bool:
    sma20 = feat["sma_fast"][i]
    if np.isnan(sma20) or i < 2:
        return False
    if direction == "bullish":
        touched = float(series.low[i]) <= sma20 * 1.001 or float(series.low[i - 1]) <= float(
            feat["sma_fast"][i - 1]
        ) * 1.001
        resume = float(series.close[i]) > sma20 and float(series.close[i]) >= float(
            series.open[i]
        )
        return touched and resume
    touched = float(series.high[i]) >= sma20 * 0.999 or float(series.high[i - 1]) >= float(
        feat["sma_fast"][i - 1]
    ) * 0.999
    resume = float(series.close[i]) < sma20 and float(series.close[i]) <= float(series.open[i])
    return touched and resume


def _breakout_trigger(series: CandleSeries, feat: dict, i: int, direction: str) -> bool:
    look = 20
    if i < look + 1:
        return False
    if direction == "bullish":
        prior = float(np.max(series.high[i - look : i]))
        return float(series.close[i]) > prior and float(series.close[i - 1]) <= prior
    prior = float(np.min(series.low[i - look : i]))
    return float(series.close[i]) < prior and float(series.close[i - 1]) >= prior


def _adaptive_exit(
    series: CandleSeries,
    feat: dict,
    entry_idx: int,
    direction: str,
    entry: float,
    atr0: float,
) -> tuple[int, float, str, float]:
    stop_dist = V2_ATR_STOP_MULT * atr0
    if stop_dist <= 0:
        last = min(entry_idx + FORWARD_BARS.get(series.timeframe, 4), len(series) - 1)
        return last, float(series.close[last]), "fixed_hold_fallback", atr0

    if direction == "bullish":
        stop = entry - stop_dist
        extreme = entry
    else:
        stop = entry + stop_dist
        extreme = entry

    last = min(entry_idx + V2_MAX_HOLD_BARS, len(series) - 1)
    for j in range(entry_idx + 1, last + 1):
        hi = float(series.high[j])
        lo = float(series.low[j])
        atr_j = feat["atr"][j]
        if np.isnan(atr_j) or atr_j <= 0:
            atr_j = atr0

        struct = int(feat["structure"][j])
        if direction == "bullish" and struct == -1:
            return j, float(series.close[j]), "structure_break", stop_dist
        if direction == "bearish" and struct == 1:
            return j, float(series.close[j]), "structure_break", stop_dist

        if direction == "bullish":
            if lo <= stop:
                return j, stop, "atr_stop", stop_dist
            extreme = max(extreme, hi)
            if extreme >= entry + stop_dist:
                trail = extreme - V2_ATR_STOP_MULT * atr_j
                _, swing_lo = last_confirmed_swing(series.high, series.low, j)
                if swing_lo is not None:
                    trail = max(trail, swing_lo)
                stop = max(stop, trail)
        else:
            if hi >= stop:
                return j, stop, "atr_stop", stop_dist
            extreme = min(extreme, lo)
            if extreme <= entry - stop_dist:
                trail = extreme + V2_ATR_STOP_MULT * atr_j
                swing_hi, _ = last_confirmed_swing(series.high, series.low, j)
                if swing_hi is not None:
                    trail = min(trail, swing_hi)
                stop = min(stop, trail)

    return last, float(series.close[last]), "max_hold", stop_dist


def _fixed_exit(series: CandleSeries, entry_idx: int) -> tuple[int, float, str]:
    horizon = FORWARD_BARS.get(series.timeframe, 4)
    last = min(entry_idx + horizon, len(series) - 1)
    return last, float(series.close[last]), "fixed_hold"


def _make_trade(
    *,
    series: CandleSeries,
    stage: str,
    direction: str,
    confidence: str,
    score: int,
    entry_idx: int,
    exit_idx: int,
    entry: float,
    exit_px: float,
    stop_dist: float,
    cost_mult: float,
    trigger: str,
    regime: str,
    exit_reason: str,
    feature_flags: dict[str, int] | None = None,
    atr_at_entry: float | None = None,
) -> V2Trade:
    asset = series.asset_class
    base_cost = ROUND_TRIP_COST.get(asset, 0.001) * cost_mult
    gross_px = _price_move(direction, entry, exit_px)
    stop_pct = stop_dist / entry if entry > 0 else 0.0
    if stop_pct <= 0:
        stop_pct = abs(gross_px) if gross_px != 0 else 1e-6
    gross_r = gross_px / stop_pct
    cost_r = base_cost / stop_pct
    net_r = gross_r - cost_r
    equity_net = net_r * V2_RISK_FRACTION
    equity_gross = gross_r * V2_RISK_FRACTION
    equity_cost = cost_r * V2_RISK_FRACTION
    return V2Trade(
        instrument=series.instrument,
        asset_class=asset,
        timeframe=series.timeframe,
        confidence=confidence,
        direction=direction,
        score=score,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_ts=int(series.timestamps[entry_idx]),
        exit_ts=int(series.timestamps[exit_idx]),
        entry_price=entry,
        exit_price=exit_px,
        gross_return=equity_gross,
        cost=equity_cost,
        net_return=equity_net,
        win=equity_net > 0,
        feature_flags=feature_flags or {},
        rules_name=stage,
        atr_at_entry=atr_at_entry,
        exit_reason=exit_reason,
        r_multiple=net_r,
        stage=stage,
        trigger=trigger,
        regime=regime,
        risk_frac=V2_RISK_FRACTION,
        stop_dist_pct=stop_pct,
        gross_r=gross_r,
        cost_r=cost_r,
        net_r=net_r,
        cost_mult=cost_mult,
    )


def rescale_cost(trade: V2Trade, cost_mult: float) -> V2Trade:
    """Rebuild equity PnL for a different cost multiple without resimulating."""
    if trade.cost_mult == cost_mult:
        return trade
    # cost_r scales linearly with cost_mult
    base_cost_r = trade.cost_r / trade.cost_mult if trade.cost_mult else trade.cost_r
    cost_r = base_cost_r * cost_mult
    net_r = trade.gross_r - cost_r
    equity_net = net_r * V2_RISK_FRACTION
    equity_cost = cost_r * V2_RISK_FRACTION
    return replace(
        trade,
        cost=equity_cost,
        net_return=equity_net,
        win=equity_net > 0,
        r_multiple=net_r,
        cost_r=cost_r,
        net_r=net_r,
        cost_mult=cost_mult,
    )


def filter_by_entry_frac(
    trades: list[V2Trade], series_lens: dict[str, int], start_frac: float, end_frac: float
) -> list[V2Trade]:
    """Keep trades whose entry_idx falls in [start_frac, end_frac) of that series."""
    out: list[V2Trade] = []
    for t in trades:
        n = series_lens.get(t.instrument)
        if not n:
            continue
        lo = int(n * start_frac)
        hi = int(n * end_frac)
        if lo <= t.entry_idx < hi:
            out.append(t)
    return out


def backtest_v2_stage(
    series: CandleSeries,
    stage: V2Stage,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    feat: dict[str, np.ndarray] | None = None,
) -> list[V2Trade]:
    feat = feat or _precompute(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, 40)
    horizon = FORWARD_BARS.get(series.timeframe, 4)
    max_need = V2_MAX_HOLD_BARS if stage.exit_mode == "adaptive" else horizon
    trades: list[V2Trade] = []
    i = max(warmup, start_idx or warmup)
    n = len(series)
    last_start = (
        n - max_need if end_idx_exclusive is None else min(n - max_need, end_idx_exclusive)
    )

    while i < last_start:
        if stage.use_original_score:
            direction, confidence, score, flags = _original_signal_at(series, feat, i)
            if direction is None or confidence not in ("HIGH", "MEDIUM"):
                i += 1
                continue
            trigger = "original_score"
            atr0 = feat["atr"][i]
            if np.isnan(atr0) or atr0 <= 0:
                i += 1
                continue
            adx_v = feat["adx"][i]
            regime_lbl = (
                "trending"
                if (not np.isnan(adx_v) and adx_v >= V2_ADX_MIN)
                else "ranging"
            )
            entry = float(series.close[i])
            stop_dist = V2_ATR_STOP_MULT * float(atr0)
            exit_idx, exit_px, reason = _fixed_exit(series, i)
            trades.append(
                _make_trade(
                    series=series,
                    stage=stage.name,
                    direction=direction,
                    confidence=confidence,
                    score=score,
                    entry_idx=i,
                    exit_idx=exit_idx,
                    entry=entry,
                    exit_px=exit_px,
                    stop_dist=stop_dist,
                    cost_mult=cost_mult,
                    trigger=trigger,
                    regime=regime_lbl,
                    exit_reason=reason,
                    feature_flags=flags,
                    atr_at_entry=float(atr0),
                )
            )
            i = exit_idx + 1
            continue

        direction, regime_lbl = _regime_at(feat, i)
        if direction is None:
            i += 1
            continue

        trigger = ""
        if stage.allow_pullback and _pullback_trigger(series, feat, i, direction):
            trigger = "pullback"
        elif stage.allow_breakout and _breakout_trigger(series, feat, i, direction):
            trigger = "breakout"
        else:
            i += 1
            continue

        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        entry = float(series.close[i])
        stop_dist = V2_ATR_STOP_MULT * float(atr0)

        if stage.exit_mode == "adaptive":
            exit_idx, exit_px, reason, stop_dist = _adaptive_exit(
                series, feat, i, direction, entry, float(atr0)
            )
        else:
            exit_idx, exit_px, reason = _fixed_exit(series, i)

        trades.append(
            _make_trade(
                series=series,
                stage=stage.name,
                direction=direction,
                confidence="V2",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=trigger,
                regime=regime_lbl,
                exit_reason=reason,
                feature_flags={
                    "pullback": int(trigger == "pullback"),
                    "breakout": int(trigger == "breakout"),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def collect_stage_trades(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V2Stage,
    *,
    cost_mult: float = 1.0,
) -> tuple[list[V2Trade], dict[str, int]]:
    """One full-sample pass per series. Returns trades + per-instrument lengths."""
    trades: list[V2Trade] = []
    lengths: dict[str, int] = {}
    for (_, _), series in series_map.items():
        lengths[series.instrument] = len(series)
        feat = _precompute(series)
        trades.extend(backtest_v2_stage(series, stage, cost_mult=cost_mult, feat=feat))
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades, lengths


def run_stage_on_map(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V2Stage,
    *,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[V2Trade]:
    if cached is None:
        all_trades, lengths = collect_stage_trades(series_map, stage, cost_mult=1.0)
    else:
        all_trades, lengths = cached
    windowed = filter_by_entry_frac(all_trades, lengths, start_frac, end_frac)
    if cost_mult != 1.0:
        windowed = [rescale_cost(t, cost_mult) for t in windowed]
    return windowed


def chronological_folds(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V2Stage,
    n_folds: int = 4,
    *,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[dict]:
    if cached is None:
        cached = collect_stage_trades(series_map, stage, cost_mult=1.0)
    results = []
    for k in range(n_folds):
        start = k / n_folds
        end = (k + 1) / n_folds
        trades = run_stage_on_map(
            series_map,
            stage,
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
            cached=cached,
        )
        results.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return results
