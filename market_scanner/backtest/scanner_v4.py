"""Scanner V4 — RESEARCH / BACKTEST ONLY.

Starts from V3_S2 STRUCTURE+MA hypothesis. Diagnoses stocks/commodities vs FX,
then tests selective structural swing-break entries and simple false-break
filters with TRAIN-only selection frozen before OOS.

Does NOT modify live ORIGINAL. Does NOT merge V3. No live/paper trading enablement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V3_BREAKOUT_LOOKBACK,
    V4_ATR_STOP_MULT,
    V4_MAX_HOLD_BARS,
    V4_MIN_BREAK_ATR,
    V4_PERSIST_BARS,
    V4_STRUCT_PIVOT,
)
from indicators import last_confirmed_swing
from models import CandleSeries
from backtest.scanner_v2 import (
    V2Trade,
    _adaptive_exit,
    _make_trade,
    _precompute,
    filter_by_entry_frac,
    rescale_cost,
)
from backtest.scanner_v3 import (
    V3Stage,
    _breakout_dir,
    _simple_regime,
    backtest_v3_entry,
)


@dataclass(frozen=True)
class V4Candidate:
    name: str
    notes: str
    # Entry family
    break_mode: str = "donchian20"  # donchian20 | swing_close | swing_confirm
    # False-break filters (individually; "none" = off)
    filter_name: str = "none"  # none | min_break_atr | trend_persist
    # Universe
    asset_classes: tuple[str, ...] = ("stock", "commodity", "forex")
    require_structure: bool = True
    require_ma: bool = True
    exit_policy: str = "adaptive_v2"


# Stage 1 baseline = exact V3_S2 logic
V4_S1_BASELINE = V4Candidate(
    name="V4_S1_V3S2_BASELINE",
    notes="V3_S2 STRUCTURE+MA + 20-bar breakout unchanged (diagnostic attribution).",
    break_mode="donchian20",
    filter_name="none",
    asset_classes=("stock", "commodity", "forex"),
)

# Stage 2 structural-break variants (stock+commodity universe when justified)
V4_S2_DONCHIAN = V4Candidate(
    name="V4_S2_DONCHIAN20",
    notes="Stock+commodity: V3_S2 unchanged entry (20-bar) as Stage-2 control.",
    break_mode="donchian20",
    asset_classes=("stock", "commodity"),
)
V4_S2_SWING_CLOSE = V4Candidate(
    name="V4_S2_SWING_CLOSE",
    notes="Stock+commodity: close beyond prior confirmed swing high/low.",
    break_mode="swing_close",
    asset_classes=("stock", "commodity"),
)
V4_S2_SWING_CONFIRM = V4Candidate(
    name="V4_S2_SWING_CONFIRM",
    notes="Stock+commodity: swing break then next 4H bar confirms beyond swing.",
    break_mode="swing_confirm",
    asset_classes=("stock", "commodity"),
)

STAGE2_VARIANTS: tuple[V4Candidate, ...] = (
    V4_S2_DONCHIAN,
    V4_S2_SWING_CLOSE,
    V4_S2_SWING_CONFIRM,
)

FILTER_NAMES: tuple[str, ...] = ("none", "min_break_atr", "trend_persist")


def _swing_level_arrays(
    series: CandleSeries, *, pivot: int = V4_STRUCT_PIVOT
) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(n):
        h, l = last_confirmed_swing(series.high, series.low, i, pivot=pivot)
        if h is not None:
            sh[i] = h
        if l is not None:
            sl[i] = l
    return sh, sl


def _structure_persist(feat: dict, i: int, direction: str, bars: int = V4_PERSIST_BARS) -> bool:
    if i < bars:
        return False
    want = 1 if direction == "bullish" else -1
    for j in range(i - bars + 1, i + 1):
        if int(feat["structure"][j]) != want:
            return False
    return True


def _break_signal(
    series: CandleSeries,
    feat: dict,
    i: int,
    direction: str,
    *,
    break_mode: str,
    swing_hi: np.ndarray,
    swing_lo: np.ndarray,
) -> tuple[bool, float]:
    """Return (triggered, breakout_distance_price). Distance = |close - swing|."""
    c = float(series.close[i])
    if break_mode == "donchian20":
        d = _breakout_dir(series, i, lookback=V3_BREAKOUT_LOOKBACK)
        if d != direction:
            return False, 0.0
        # distance vs prior 20-bar extreme
        if direction == "bullish":
            prior = float(np.max(series.high[i - V3_BREAKOUT_LOOKBACK : i]))
            return True, max(0.0, c - prior)
        prior = float(np.min(series.low[i - V3_BREAKOUT_LOOKBACK : i]))
        return True, max(0.0, prior - c)

    if i < 2:
        return False, 0.0

    if break_mode == "swing_close":
        if direction == "bullish":
            lvl = swing_hi[i - 1]
            if np.isnan(lvl):
                return False, 0.0
            if c > float(lvl) and float(series.close[i - 1]) <= float(lvl):
                return True, c - float(lvl)
        else:
            lvl = swing_lo[i - 1]
            if np.isnan(lvl):
                return False, 0.0
            if c < float(lvl) and float(series.close[i - 1]) >= float(lvl):
                return True, float(lvl) - c
        return False, 0.0

    if break_mode == "swing_confirm":
        # Prior bar broke swing; current bar holds beyond and closes in direction
        if direction == "bullish":
            lvl = swing_hi[i - 2] if i >= 2 else np.nan
            if np.isnan(lvl):
                return False, 0.0
            broke = float(series.close[i - 1]) > float(lvl)
            holds = c > float(lvl) and c >= float(series.open[i])
            if broke and holds:
                return True, c - float(lvl)
        else:
            lvl = swing_lo[i - 2] if i >= 2 else np.nan
            if np.isnan(lvl):
                return False, 0.0
            broke = float(series.close[i - 1]) < float(lvl)
            holds = c < float(lvl) and c <= float(series.open[i])
            if broke and holds:
                return True, float(lvl) - c
        return False, 0.0

    return False, 0.0


def _filter_ok(
    feat: dict,
    i: int,
    direction: str,
    *,
    filter_name: str,
    break_dist: float,
    atr0: float,
) -> bool:
    if filter_name == "none":
        return True
    if filter_name == "min_break_atr":
        if atr0 <= 0:
            return False
        return (break_dist / atr0) >= V4_MIN_BREAK_ATR
    if filter_name == "trend_persist":
        return _structure_persist(feat, i, direction, V4_PERSIST_BARS)
    return True


def backtest_v4(
    series: CandleSeries,
    cand: V4Candidate,
    *,
    cost_mult: float = 1.0,
    feat: dict | None = None,
) -> list[V2Trade]:
    if series.asset_class not in cand.asset_classes:
        return []

    # Exact V3_S2 path for donchian+none (bit-compatible baseline when full universe)
    if (
        cand.break_mode == "donchian20"
        and cand.filter_name == "none"
        and cand.require_structure
        and cand.require_ma
    ):
        # Reuse V3 entry with class filter via post-check (already gated above)
        stage = V3Stage(
            name=cand.name,
            kind="entry",
            notes=cand.notes,
            require_structure=True,
            require_ma=True,
            require_adx=False,
            allow_breakout=True,
            allow_pullback=False,
            exit_mode="adaptive",
            exit_policy="adaptive_v2",
        )
        return backtest_v3_entry(series, stage, cost_mult=cost_mult, feat=feat)

    feat = feat or _precompute(series)
    swing_hi, swing_lo = _swing_level_arrays(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, V3_BREAKOUT_LOOKBACK + 5, 10)
    n = len(series)
    last_start = n - V4_MAX_HOLD_BARS
    trades: list[V2Trade] = []
    i = warmup

    while i < last_start:
        regime = _simple_regime(
            feat,
            i,
            require_structure=cand.require_structure,
            require_ma=cand.require_ma,
            require_adx=False,
        )
        if regime not in ("bullish", "bearish"):
            i += 1
            continue

        ok, dist = _break_signal(
            series,
            feat,
            i,
            regime,
            break_mode=cand.break_mode,
            swing_hi=swing_hi,
            swing_lo=swing_lo,
        )
        if not ok:
            i += 1
            continue

        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        if not _filter_ok(
            feat,
            i,
            regime,
            filter_name=cand.filter_name,
            break_dist=dist,
            atr0=float(atr0),
        ):
            i += 1
            continue

        entry = float(series.close[i])
        exit_idx, exit_px, reason, stop_dist = _adaptive_exit(
            series, feat, i, regime, entry, float(atr0)
        )
        # Align stop mult naming with V4 constant (same 1.5 as V2/V3 adaptive)
        _ = V4_ATR_STOP_MULT
        trades.append(
            _make_trade(
                series=series,
                stage=cand.name,
                direction=regime,
                confidence="V4",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=cand.break_mode,
                regime="trending",
                exit_reason=reason,
                feature_flags={
                    "swing": int(cand.break_mode.startswith("swing")),
                    "filter_atr": int(cand.filter_name == "min_break_atr"),
                    "filter_persist": int(cand.filter_name == "trend_persist"),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def collect_v4(
    series_map: dict[tuple[str, str], CandleSeries],
    cand: V4Candidate,
    *,
    cost_mult: float = 1.0,
) -> tuple[list[V2Trade], dict[str, int]]:
    trades: list[V2Trade] = []
    lengths: dict[str, int] = {}
    for (_, _), series in series_map.items():
        lengths[series.instrument] = len(series)
        if series.asset_class not in cand.asset_classes:
            continue
        feat = _precompute(series)
        trades.extend(backtest_v4(series, cand, cost_mult=cost_mult, feat=feat))
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades, lengths


def run_v4_window(
    series_map: dict[tuple[str, str], CandleSeries],
    cand: V4Candidate,
    *,
    start_frac: float,
    end_frac: float,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[V2Trade]:
    if cached is None:
        all_trades, lengths = collect_v4(series_map, cand, cost_mult=1.0)
    else:
        all_trades, lengths = cached
    windowed = filter_by_entry_frac(all_trades, lengths, start_frac, end_frac)
    if cost_mult != 1.0:
        windowed = [rescale_cost(t, cost_mult) for t in windowed]
    return windowed


def chronological_folds_v4(
    series_map: dict[tuple[str, str], CandleSeries],
    cand: V4Candidate,
    n_folds: int = 4,
    *,
    cost_mult: float = 1.0,
    cached: tuple[list[V2Trade], dict[str, int]] | None = None,
) -> list[dict]:
    if cached is None:
        cached = collect_v4(series_map, cand, cost_mult=1.0)
    out = []
    for k in range(n_folds):
        start = k / n_folds
        end = (k + 1) / n_folds
        trades = run_v4_window(
            series_map,
            cand,
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
            cached=cached,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out


def with_filter(base: V4Candidate, filter_name: str) -> V4Candidate:
    return V4Candidate(
        name=f"{base.name}__{filter_name}" if filter_name != "none" else base.name,
        notes=f"{base.notes} | filter={filter_name}",
        break_mode=base.break_mode,
        filter_name=filter_name,
        asset_classes=base.asset_classes,
        require_structure=base.require_structure,
        require_ma=base.require_ma,
        exit_policy=base.exit_policy,
    )


def fx_clone(base: V4Candidate) -> V4Candidate:
    return V4Candidate(
        name=f"{base.name}__FX",
        notes=f"FX control: same frozen params as {base.name} (no FX-specific retune).",
        break_mode=base.break_mode,
        filter_name=base.filter_name,
        asset_classes=("forex",),
        require_structure=base.require_structure,
        require_ma=base.require_ma,
        exit_policy=base.exit_policy,
    )


def stock_commodity_clone(base: V4Candidate, name: str | None = None) -> V4Candidate:
    return V4Candidate(
        name=name or base.name,
        notes=base.notes,
        break_mode=base.break_mode,
        filter_name=base.filter_name,
        asset_classes=("stock", "commodity"),
        require_structure=base.require_structure,
        require_ma=base.require_ma,
        exit_policy=base.exit_policy,
    )
