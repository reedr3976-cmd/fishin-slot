"""Scanner V3 — RESEARCH / BACKTEST ONLY.

Redesign stages driven by V2 findings (breakout-first, simplified regime,
ORIGINAL exit-only ablation, optional asset-class rules).

Does NOT modify or replace the live ORIGINAL scanner. Does NOT merge V2 live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    FORWARD_BARS,
    SMA_SLOW,
    V2_ADX_MIN,
    V3_ATR_STOP_MULT,
    V3_BREAKOUT_LOOKBACK,
    V3_MAX_HOLD_BARS,
    V3_RR_TARGET,
    V3_STRUCT_PIVOT,
)
from indicators import last_confirmed_swing
from models import CandleSeries
from backtest.scanner_v2 import (
    STAGE_ORIGINAL,
    STAGE_S3,
    V2Trade,
    _adaptive_exit,
    _fixed_exit,
    _make_trade,
    _original_signal_at,
    _precompute,
    _pullback_trigger,
    backtest_v2_stage,
    filter_by_entry_frac,
    rescale_cost,
)


# ---------------------------------------------------------------------------
# V3 stage definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3Stage:
    name: str
    kind: str  # entry | exit_only | asset_class
    notes: str
    # entry knobs
    require_structure: bool = False
    require_ma: bool = False
    require_adx: bool = False
    allow_breakout: bool = True
    allow_pullback: bool = False
    exit_mode: str = "adaptive"  # fixed | adaptive | named exit policy
    exit_policy: str = "adaptive_v2"  # for exit_only / named exits
    asset_class_filter: Optional[str] = None  # forex|stock|commodity|None
    # asset-class overrides (Stage 4): lookback / atr mult by class
    class_params: Optional[dict[str, dict]] = None


# Benchmarks (reused from V2 module semantics)
BENCH_ORIGINAL = V3Stage(
    name="ORIGINAL",
    kind="entry",
    notes="Control: ORIGINAL MH score + fixed 4-bar hold (1% risk-normalized).",
    allow_breakout=False,
    exit_mode="fixed",
    exit_policy="fixed_hold",
)
BENCH_V2_S3 = V3Stage(
    name="V2_S3_DUAL_TRIG",
    kind="entry",
    notes="Prior V2 benchmark: heavy regime + pullback/breakout + adaptive exits.",
    require_structure=True,
    require_ma=True,
    require_adx=True,
    allow_breakout=True,
    allow_pullback=True,
    exit_mode="adaptive",
    exit_policy="adaptive_v2",
)

# Stage 1 — breakout-first, minimal filtering
V3_S1_BREAKOUT_HOLD = V3Stage(
    name="V3_S1_BREAKOUT_HOLD",
    kind="entry",
    notes="Breakout of 20-bar high/low only; no regime gate; fixed hold (entry isolate).",
    exit_mode="fixed",
    exit_policy="fixed_hold",
)
V3_S1_BREAKOUT_ADAPT = V3Stage(
    name="V3_S1_BREAKOUT_ADAPT",
    kind="entry",
    notes="Breakout-first; no regime gate; ATR/structure adaptive exit.",
    exit_mode="adaptive",
    exit_policy="adaptive_v2",
)

# Stage 2 — simplified regime (structure + MA); optional ADX probe
V3_S2_STRUCT_MA = V3Stage(
    name="V3_S2_STRUCT_MA",
    kind="entry",
    notes="HH/HL or LH/LL + SMA stack/slope; breakout entry; adaptive exit. No ADX/vol gate.",
    require_structure=True,
    require_ma=True,
    require_adx=False,
    exit_mode="adaptive",
    exit_policy="adaptive_v2",
)
V3_S2_STRUCT_MA_ADX = V3Stage(
    name="V3_S2_STRUCT_MA_ADX",
    kind="entry",
    notes="S2 + ADX≥20/DI align only (probe whether ADX helps OOS).",
    require_structure=True,
    require_ma=True,
    require_adx=True,
    exit_mode="adaptive",
    exit_policy="adaptive_v2",
)

# Stage 3 — exit-only on ORIGINAL entries
V3_X_FIXED = V3Stage(
    name="V3_X_FIXED",
    kind="exit_only",
    notes="ORIGINAL entries + fixed 4-bar hold.",
    exit_policy="fixed_hold",
)
V3_X_ATR_STOP = V3Stage(
    name="V3_X_ATR_STOP",
    kind="exit_only",
    notes="ORIGINAL entries + 1.5 ATR stop (no trail), max hold cap.",
    exit_policy="atr_stop",
)
V3_X_STRUCT_STOP = V3Stage(
    name="V3_X_STRUCT_STOP",
    kind="exit_only",
    notes="ORIGINAL entries + stop beyond last swing; max hold.",
    exit_policy="structure_stop",
)
V3_X_ATR_TRAIL = V3Stage(
    name="V3_X_ATR_TRAIL",
    kind="exit_only",
    notes="ORIGINAL entries + ATR chandelier trail after entry.",
    exit_policy="atr_trail",
)
V3_X_STRUCT_TRAIL = V3Stage(
    name="V3_X_STRUCT_TRAIL",
    kind="exit_only",
    notes="ORIGINAL entries + trail at confirmed swing structure.",
    exit_policy="structure_trail",
)
V3_X_RR2 = V3Stage(
    name="V3_X_RR2",
    kind="exit_only",
    notes="ORIGINAL entries + 1.5 ATR stop and 2R target.",
    exit_policy="rr_target",
)
V3_X_ADAPTIVE = V3Stage(
    name="V3_X_ADAPTIVE",
    kind="exit_only",
    notes="ORIGINAL entries + full V2 adaptive exit (ATR stop/trail + structure break).",
    exit_policy="adaptive_v2",
)

ENTRY_STAGES: tuple[V3Stage, ...] = (
    BENCH_ORIGINAL,
    BENCH_V2_S3,
    V3_S1_BREAKOUT_HOLD,
    V3_S1_BREAKOUT_ADAPT,
    V3_S2_STRUCT_MA,
    V3_S2_STRUCT_MA_ADX,
)

EXIT_STAGES: tuple[V3Stage, ...] = (
    V3_X_FIXED,
    V3_X_ATR_STOP,
    V3_X_STRUCT_STOP,
    V3_X_ATR_TRAIL,
    V3_X_STRUCT_TRAIL,
    V3_X_RR2,
    V3_X_ADAPTIVE,
)


def _simple_regime(
    feat: dict,
    i: int,
    *,
    require_structure: bool,
    require_ma: bool,
    require_adx: bool,
) -> Optional[str]:
    """Return bullish|bearish|None using light gates only."""
    sma_f = feat["sma_fast"][i]
    sma_s = feat["sma_slow"][i]
    slope = feat["sma_slope"][i]
    struct = int(feat["structure"][i])
    if any(np.isnan(x) for x in (sma_f, sma_s, slope)):
        return None

    bull_ma = sma_f > sma_s and slope > 0
    bear_ma = sma_f < sma_s and slope < 0
    bull_st = struct == 1
    bear_st = struct == -1

    if require_structure and require_ma:
        bull = bull_st and bull_ma
        bear = bear_st and bear_ma
    elif require_structure:
        bull, bear = bull_st, bear_st
    elif require_ma:
        bull, bear = bull_ma, bear_ma
    else:
        # No regime: direction comes from breakout side only
        return "any"

    if require_adx:
        adx_v = feat["adx"][i]
        pdi = feat["plus_di"][i]
        mdi = feat["minus_di"][i]
        if any(np.isnan(x) for x in (adx_v, pdi, mdi)) or adx_v < V2_ADX_MIN:
            return None
        bull = bull and pdi > mdi
        bear = bear and mdi > pdi

    if bull:
        return "bullish"
    if bear:
        return "bearish"
    return None


def _breakout_dir(series: CandleSeries, i: int, lookback: int = V3_BREAKOUT_LOOKBACK) -> Optional[str]:
    if i < lookback + 1:
        return None
    prior_hi = float(np.max(series.high[i - lookback : i]))
    prior_lo = float(np.min(series.low[i - lookback : i]))
    c = float(series.close[i])
    c_prev = float(series.close[i - 1])
    bull = c > prior_hi and c_prev <= prior_hi
    bear = c < prior_lo and c_prev >= prior_lo
    if bull and not bear:
        return "bullish"
    if bear and not bull:
        return "bearish"
    return None


def _exit_policy(
    series: CandleSeries,
    feat: dict,
    entry_idx: int,
    direction: str,
    entry: float,
    atr0: float,
    policy: str,
) -> tuple[int, float, str, float]:
    """Return (exit_idx, exit_px, reason, stop_dist)."""
    stop_dist = V3_ATR_STOP_MULT * atr0
    if stop_dist <= 0:
        j, px, reason = _fixed_exit(series, entry_idx)
        return j, px, reason, max(atr0, 1e-9)

    horizon = FORWARD_BARS.get(series.timeframe, 4)
    last = min(entry_idx + V3_MAX_HOLD_BARS, len(series) - 1)

    if policy == "fixed_hold":
        j, px, reason = _fixed_exit(series, entry_idx)
        return j, px, reason, stop_dist

    if policy == "adaptive_v2":
        return _adaptive_exit(series, feat, entry_idx, direction, entry, atr0)

    # Initial structure stop reference
    swing_hi, swing_lo = last_confirmed_swing(
        series.high, series.low, entry_idx, pivot=V3_STRUCT_PIVOT
    )
    if direction == "bullish":
        atr_stop = entry - stop_dist
        struct_stop = (swing_lo - 1e-12) if swing_lo is not None else atr_stop
        stop = atr_stop if policy.startswith("atr") or policy == "rr_target" else struct_stop
        if policy == "structure_stop":
            stop = struct_stop
        target = entry + V3_RR_TARGET * (entry - stop) if policy == "rr_target" else None
        extreme = entry
    else:
        atr_stop = entry + stop_dist
        struct_stop = (swing_hi + 1e-12) if swing_hi is not None else atr_stop
        stop = atr_stop if policy.startswith("atr") or policy == "rr_target" else struct_stop
        if policy == "structure_stop":
            stop = struct_stop
        target = entry - V3_RR_TARGET * (stop - entry) if policy == "rr_target" else None
        extreme = entry

    # Ensure stop distance for risk unit reflects initial stop
    init_stop_dist = abs(entry - stop) if stop != entry else stop_dist

    for j in range(entry_idx + 1, last + 1):
        hi = float(series.high[j])
        lo = float(series.low[j])
        atr_j = feat["atr"][j]
        if np.isnan(atr_j) or atr_j <= 0:
            atr_j = atr0

        if policy == "atr_stop":
            if direction == "bullish" and lo <= stop:
                return j, stop, "atr_stop", init_stop_dist
            if direction == "bearish" and hi >= stop:
                return j, stop, "atr_stop", init_stop_dist
            # time stop at horizon if sooner than max hold for classic stop study
            if j >= entry_idx + horizon:
                return j, float(series.close[j]), "time_stop", init_stop_dist

        elif policy == "structure_stop":
            if direction == "bullish" and lo <= stop:
                return j, stop, "structure_stop", init_stop_dist
            if direction == "bearish" and hi >= stop:
                return j, stop, "structure_stop", init_stop_dist
            if j >= entry_idx + horizon:
                return j, float(series.close[j]), "time_stop", init_stop_dist

        elif policy == "atr_trail":
            if direction == "bullish":
                if lo <= stop:
                    return j, stop, "atr_trail_stop", init_stop_dist
                extreme = max(extreme, hi)
                stop = max(stop, extreme - V3_ATR_STOP_MULT * atr_j)
            else:
                if hi >= stop:
                    return j, stop, "atr_trail_stop", init_stop_dist
                extreme = min(extreme, lo)
                stop = min(stop, extreme + V3_ATR_STOP_MULT * atr_j)

        elif policy == "structure_trail":
            sh, sl = last_confirmed_swing(series.high, series.low, j, pivot=V3_STRUCT_PIVOT)
            if direction == "bullish":
                if lo <= stop:
                    return j, stop, "structure_trail_stop", init_stop_dist
                if sl is not None:
                    stop = max(stop, sl)
            else:
                if hi >= stop:
                    return j, stop, "structure_trail_stop", init_stop_dist
                if sh is not None:
                    stop = min(stop, sh)

        elif policy == "rr_target":
            assert target is not None
            if direction == "bullish":
                if lo <= stop:
                    return j, stop, "atr_stop", init_stop_dist
                if hi >= target:
                    return j, target, "rr_target", init_stop_dist
            else:
                if hi >= stop:
                    return j, stop, "atr_stop", init_stop_dist
                if lo <= target:
                    return j, target, "rr_target", init_stop_dist

    return last, float(series.close[last]), "max_hold", init_stop_dist


def backtest_v3_entry(
    series: CandleSeries,
    stage: V3Stage,
    *,
    cost_mult: float = 1.0,
    feat: dict | None = None,
) -> list[V2Trade]:
    """Breakout-first / simplified-regime entry stages."""
    if stage.name == "ORIGINAL":
        return backtest_v2_stage(series, STAGE_ORIGINAL, cost_mult=cost_mult, feat=feat)
    if stage.name == "V2_S3_DUAL_TRIG":
        return backtest_v2_stage(series, STAGE_S3, cost_mult=cost_mult, feat=feat)

    feat = feat or _precompute(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, V3_BREAKOUT_LOOKBACK + 5)
    max_need = V3_MAX_HOLD_BARS if stage.exit_mode != "fixed" else FORWARD_BARS.get(
        series.timeframe, 4
    )
    n = len(series)
    last_start = n - max_need
    trades: list[V2Trade] = []
    i = warmup

    lookback = V3_BREAKOUT_LOOKBACK
    if stage.class_params and series.asset_class in stage.class_params:
        lookback = int(stage.class_params[series.asset_class].get("lookback", lookback))

    while i < last_start:
        if stage.asset_class_filter and series.asset_class != stage.asset_class_filter:
            break

        regime = _simple_regime(
            feat,
            i,
            require_structure=stage.require_structure,
            require_ma=stage.require_ma,
            require_adx=stage.require_adx,
        )
        br_dir = _breakout_dir(series, i, lookback=lookback) if stage.allow_breakout else None

        direction: Optional[str] = None
        trigger = ""
        if regime == "any":
            if br_dir:
                direction = br_dir
                trigger = "breakout"
        elif regime in ("bullish", "bearish"):
            if stage.allow_breakout and br_dir == regime:
                direction = regime
                trigger = "breakout"
            elif stage.allow_pullback and _pullback_trigger(series, feat, i, regime):
                direction = regime
                trigger = "pullback"

        if direction is None:
            i += 1
            continue

        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        entry = float(series.close[i])
        atr_mult = V3_ATR_STOP_MULT
        if stage.class_params and series.asset_class in stage.class_params:
            atr_mult = float(
                stage.class_params[series.asset_class].get("atr_mult", atr_mult)
            )
        # temporarily patch via stop_dist scaling inside exit by adjusting atr0
        atr_for_exit = float(atr0) * (atr_mult / V3_ATR_STOP_MULT)

        exit_idx, exit_px, reason, stop_dist = _exit_policy(
            series,
            feat,
            i,
            direction,
            entry,
            atr_for_exit,
            stage.exit_policy,
        )
        trades.append(
            _make_trade(
                series=series,
                stage=stage.name,
                direction=direction,
                confidence="V3",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=trigger,
                regime="trending",
                exit_reason=reason,
                feature_flags={"breakout": int(trigger == "breakout")},
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def collect_original_entries(
    series: CandleSeries, feat: dict | None = None
) -> list[dict]:
    """Collect ORIGINAL MH entries (entry index/direction/score) without exiting."""
    feat = feat or _precompute(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    horizon = FORWARD_BARS.get(series.timeframe, 4)
    n = len(series)
    last_start = n - max(horizon, V3_MAX_HOLD_BARS)
    entries: list[dict] = []
    i = warmup
    while i < last_start:
        direction, confidence, score, flags = _original_signal_at(series, feat, i)
        if direction is None or confidence not in ("HIGH", "MEDIUM"):
            i += 1
            continue
        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        entries.append(
            {
                "idx": i,
                "direction": direction,
                "confidence": confidence,
                "score": score,
                "flags": flags,
                "atr": float(atr0),
                "entry": float(series.close[i]),
            }
        )
        # Non-overlap using fixed hold spacing (same as ORIGINAL control)
        i = i + horizon + 1  # skip hold window so entry set matches fixed-hold ORIGINAL cadence
    return entries


def backtest_v3_exit_only(
    series: CandleSeries,
    stage: V3Stage,
    *,
    cost_mult: float = 1.0,
    feat: dict | None = None,
    entries: list[dict] | None = None,
) -> list[V2Trade]:
    """Apply an exit policy to frozen ORIGINAL entries (no look-ahead on entries)."""
    feat = feat or _precompute(series)
    if entries is None:
        entries = collect_original_entries(series, feat)
    trades: list[V2Trade] = []
    n = len(series)
    next_free = 0
    for e in entries:
        i = e["idx"]
        if i < next_free or i + 1 >= n:
            continue
        direction = e["direction"]
        entry = e["entry"]
        atr0 = e["atr"]
        exit_idx, exit_px, reason, stop_dist = _exit_policy(
            series, feat, i, direction, entry, atr0, stage.exit_policy
        )
        if exit_idx >= n:
            continue
        adx_v = feat["adx"][i]
        regime = (
            "trending"
            if (not np.isnan(adx_v) and adx_v >= V2_ADX_MIN)
            else "ranging"
        )
        trades.append(
            _make_trade(
                series=series,
                stage=stage.name,
                direction=direction,
                confidence=e["confidence"],
                score=e["score"],
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger="original_score",
                regime=regime,
                exit_reason=reason,
                feature_flags=dict(e["flags"]),
                atr_at_entry=atr0,
            )
        )
        next_free = exit_idx + 1
    return trades


def collect_v3_stage(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V3Stage,
    *,
    cost_mult: float = 1.0,
) -> tuple[list[V2Trade], dict[str, int]]:
    trades: list[V2Trade] = []
    lengths: dict[str, int] = {}
    # Cache ORIGINAL entries per instrument for exit-only stages
    entry_cache: dict[str, list[dict]] = {}
    feat_cache: dict[str, dict] = {}

    for (_, _), series in series_map.items():
        lengths[series.instrument] = len(series)
        feat = _precompute(series)
        feat_cache[series.instrument] = feat
        if stage.kind == "exit_only":
            if series.instrument not in entry_cache:
                entry_cache[series.instrument] = collect_original_entries(series, feat)
            trades.extend(
                backtest_v3_exit_only(
                    series,
                    stage,
                    cost_mult=cost_mult,
                    feat=feat,
                    entries=entry_cache[series.instrument],
                )
            )
        else:
            trades.extend(backtest_v3_entry(series, stage, cost_mult=cost_mult, feat=feat))
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades, lengths


def run_v3_window(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V3Stage,
    *,
    start_frac: float,
    end_frac: float,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[V2Trade]:
    if cached is None:
        all_trades, lengths = collect_v3_stage(series_map, stage, cost_mult=1.0)
    else:
        all_trades, lengths = cached
    windowed = filter_by_entry_frac(all_trades, lengths, start_frac, end_frac)
    if cost_mult != 1.0:
        windowed = [rescale_cost(t, cost_mult) for t in windowed]
    return windowed


def chronological_folds_v3(
    series_map: dict[tuple[str, str], CandleSeries],
    stage: V3Stage,
    n_folds: int = 4,
    *,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[dict]:
    if cached is None:
        cached = collect_v3_stage(series_map, stage, cost_mult=1.0)
    out = []
    for k in range(n_folds):
        start = k / n_folds
        end = (k + 1) / n_folds
        trades = run_v3_window(
            series_map,
            stage,
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
            cached=cached,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out


def build_asset_class_stages(base: V3Stage) -> list[V3Stage]:
    """Stage 4: class-scoped copies of a justified base rule (no per-symbol opts)."""
    # Mild class-level lookback differences only (pre-specified, not optimized per symbol)
    params = {
        "forex": {"lookback": 20, "atr_mult": 1.5},
        "stock": {"lookback": 15, "atr_mult": 1.5},
        "commodity": {"lookback": 25, "atr_mult": 2.0},
    }
    stages = []
    for cls in ("forex", "stock", "commodity"):
        stages.append(
            V3Stage(
                name=f"V3_S4_{cls.upper()}",
                kind="asset_class",
                notes=f"Class-scoped {base.name} for {cls} only (lookback/atr pre-specified).",
                require_structure=base.require_structure,
                require_ma=base.require_ma,
                require_adx=base.require_adx,
                allow_breakout=base.allow_breakout,
                allow_pullback=base.allow_pullback,
                exit_mode=base.exit_mode,
                exit_policy=base.exit_policy,
                asset_class_filter=cls,
                class_params={cls: params[cls]},
            )
        )
    # Combined universe running each class with its own params in one pass
    stages.append(
        V3Stage(
            name="V3_S4_COMBINED",
            kind="asset_class",
            notes=f"All classes with class-specific lookback/ATR from {base.name}.",
            require_structure=base.require_structure,
            require_ma=base.require_ma,
            require_adx=base.require_adx,
            allow_breakout=base.allow_breakout,
            allow_pullback=base.allow_pullback,
            exit_mode=base.exit_mode,
            exit_policy=base.exit_policy,
            class_params=params,
        )
    )
    return stages
