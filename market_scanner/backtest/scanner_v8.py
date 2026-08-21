"""Scanner V8 — GENERALISATION RESEARCH after V7 FAIL (research only).

Nested TRAIN → VAL → FINAL_TIME; instrument rotations; final instrument holdout.
Does NOT modify live ORIGINAL. No paper/live enablement. No auto-promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V8_ATR_STOP_MULT,
    V8_COMPRESS_MULT,
    V8_LOOKBACK,
    V8_MAX_HOLD,
    V8_RS_LOOKBACK,
    V8_VOL_ATR_MULT,
)
from models import CandleSeries
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade, _precompute
from backtest.scanner_v6 import _daily_trend, _donchian_confirm, _ma_bear, _ma_bull, _pullback
from indicators import swing_structure_dir


@dataclass(frozen=True)
class V8Family:
    key: str
    name: str
    market_class: str  # stocks | commodities | forex | baseline
    rationale: str
    notes: str
    is_baseline: bool = False
    # Optional universe override (e.g. metals-only)
    universe_tag: str = "default"  # default | metals | energy


# --- Behaviour hypotheses (not indicator stacks) ---
FAMILIES: tuple[V8Family, ...] = (
    # Stocks
    V8Family(
        "S1",
        "V8_S_TREND_PULLBACK",
        "stocks",
        "Persistent trend continuation: price mean-reverts to the trend MA then resumes.",
        "SMA20/50+slope trend + pullback-to-SMA20 continuation.",
    ),
    V8Family(
        "S2",
        "V8_S_RS_PULLBACK",
        "stocks",
        "Cross-sectional relative strength: only trade names outperforming SPY, then pullback.",
        "20-bar return > SPY + MA trend + SMA20 pullback.",
    ),
    V8Family(
        "S3",
        "V8_S_COMPRESS_EXPAND",
        "stocks",
        "Volatility expansion after compression often initiates directional moves.",
        "ATR% compressed then expands ≥1.2× median + MA-aligned Donchian confirm.",
    ),
    V8Family(
        "S4",
        "V8_S_BREAKOUT_CONT",
        "stocks",
        "Breakout continuation when daily regime agrees reduces false 4H breaks.",
        "Daily SMA20/50 + 4H Donchian next-bar confirm + MA.",
    ),
    # Commodities — separate claims where justified
    V8Family(
        "C1",
        "V8_C_UNIVERSAL_TREND",
        "commodities",
        "Test whether trend-pullback is a universal commodity edge (V6/V7 cast doubt).",
        "MA trend + pullback on multi-sector commodity DEV panel.",
        universe_tag="default",
    ),
    V8Family(
        "C2",
        "V8_C_METALS_TREND",
        "commodities",
        "Asset-specific: precious/industrial metals may share trend behaviour unlike softs/gas.",
        "MA trend + pullback on metals only (honest non-universal claim).",
        universe_tag="metals",
    ),
    V8Family(
        "C3",
        "V8_C_ENERGY_BREAKOUT",
        "commodities",
        "Asset-specific: energy futures often trend via volatility expansion breakouts.",
        "Vol expansion + MA-aligned Donchian confirm on energy only.",
        universe_tag="energy",
    ),
    # Forex
    V8Family(
        "F1",
        "V8_F_TREND_PULLBACK",
        "forex",
        "FX trends persist at 4H when daily regime agrees; pullbacks are continuation entries.",
        "Daily trend + 4H MA pullback.",
    ),
    V8Family(
        "F2",
        "V8_F_BREAKOUT_FAIL",
        "forex",
        "FX often mean-reverts: failed Donchian breaks that snap back may be fadeable.",
        "Donchian break then close back inside range within 2 bars → fade with MA filter.",
    ),
    V8Family(
        "F3",
        "V8_F_MOM_PERSIST",
        "forex",
        "Short-horizon momentum persistence when ATR is not compressed.",
        "ATR not compressed + 10-bar momentum + RSI side + MA stack.",
    ),
    # Baselines (comparison only; still evaluated under same gates for honesty)
    V8Family(
        "B1",
        "V8_BASE_DONCHIAN",
        "baseline",
        "Simple 20-bar Donchian breakout continuation (no extra filters).",
        "Close beyond prior 20-bar high/low.",
        is_baseline=True,
    ),
    V8Family(
        "B2",
        "V8_BASE_SMA_CROSS",
        "baseline",
        "Simple SMA20/50 stack with slope (minimal trend baseline).",
        "Enter when SMA20>SMA50 and slope>0 (or bearish mirror) on close.",
        is_baseline=True,
    ),
)


_FEAT_CACHE: dict[int, dict] = {}


def _feat(series: CandleSeries) -> dict:
    feat = dict(_precompute(series))
    n = len(series)
    structure = np.zeros(n, dtype=np.int8)
    for i in range(n):
        structure[i] = swing_structure_dir(series.high, series.low, i)
    feat["structure"] = structure
    return feat


def _feat_cached(series: CandleSeries) -> dict:
    key = id(series)
    if key not in _FEAT_CACHE:
        _FEAT_CACHE[key] = _feat(series)
    return _FEAT_CACHE[key]


def clear_feat_cache() -> None:
    _FEAT_CACHE.clear()


def _vol_expansion(feat: dict, i: int) -> bool:
    ap, med = feat["atr_pct"][i], feat["atr_pct_med50"][i]
    if any(np.isnan(x) for x in (ap, med)) or med <= 0:
        return False
    return ap >= V8_VOL_ATR_MULT * med


def _compressed(feat: dict, i: int) -> bool:
    ap, med = feat["atr_pct"][i], feat["atr_pct_med50"][i]
    if any(np.isnan(x) for x in (ap, med)) or med <= 0:
        return False
    return ap <= V8_COMPRESS_MULT * med


def _recent_compression(feat: dict, i: int, look: int = 8) -> bool:
    if i < look:
        return False
    return any(_compressed(feat, j) for j in range(i - look, i))


def _atr_ok(feat: dict, i: int) -> bool:
    ap, med = feat["atr_pct"][i], feat["atr_pct_med50"][i]
    if any(np.isnan(x) for x in (ap, med)) or med <= 0:
        return False
    return ap >= V8_COMPRESS_MULT * med


def _rs_beats_spy(
    series: CandleSeries,
    spy: Optional[CandleSeries],
    i: int,
    look: int = V8_RS_LOOKBACK,
) -> Optional[bool]:
    """Causal relative strength vs SPY using matching timestamps."""
    if spy is None or i < look:
        return None
    ts = int(series.timestamps[i])
    sj = int(np.searchsorted(spy.timestamps, ts, side="right") - 1)
    if sj < look:
        return None
    r_inst = float(series.close[i]) / float(series.close[i - look]) - 1.0
    r_spy = float(spy.close[sj]) / float(spy.close[sj - look]) - 1.0
    return r_inst > r_spy


def _breakout_fail_fade(series: CandleSeries, i: int, look: int = V8_LOOKBACK) -> Optional[str]:
    """If a break within last 2 bars failed (close back inside), fade it."""
    if i < look + 3:
        return None
    for b in (i - 1, i - 2):
        prior_hi = float(np.max(series.high[b - look : b]))
        prior_lo = float(np.min(series.low[b - look : b]))
        broke_up = float(series.close[b]) > prior_hi
        broke_dn = float(series.close[b]) < prior_lo
        if broke_up and float(series.close[i]) < prior_hi and float(series.close[i]) <= float(
            series.open[i]
        ):
            return "bearish"  # fade failed upside break
        if broke_dn and float(series.close[i]) > prior_lo and float(series.close[i]) >= float(
            series.open[i]
        ):
            return "bullish"
    return None


def _raw_donchian(series: CandleSeries, i: int, look: int = V8_LOOKBACK) -> Optional[str]:
    if i < look + 1:
        return None
    prior_hi = float(np.max(series.high[i - look : i]))
    prior_lo = float(np.min(series.low[i - look : i]))
    c = float(series.close[i])
    c0 = float(series.close[i - 1])
    if c > prior_hi and c0 <= prior_hi:
        return "bullish"
    if c < prior_lo and c0 >= prior_lo:
        return "bearish"
    return None


def _signal_at(
    family: V8Family,
    series: CandleSeries,
    feat: dict,
    i: int,
    daily: Optional[CandleSeries],
    spy: Optional[CandleSeries],
) -> Optional[str]:
    key = family.key

    if key == "S1" or key == "C1" or key == "C2":
        if _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if key == "S2":
        rs = _rs_beats_spy(series, spy, i)
        if rs is None:
            return None
        if rs and _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        # bearish: underperform SPY
        if (rs is False) and _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            # for shorts, require underperformance
            r_under = _rs_beats_spy(series, spy, i)
            if r_under is False:
                return "bearish"
        return None

    if key == "S3" or key == "C3":
        if not (_recent_compression(feat, i) and _vol_expansion(feat, i)):
            return None
        if _ma_bull(feat, i) and _donchian_confirm(series, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _donchian_confirm(series, i, "bearish"):
            return "bearish"
        return None

    if key == "S4":
        dt = _daily_trend(daily, int(series.timestamps[i]))
        if dt == "bullish" and _ma_bull(feat, i) and _donchian_confirm(series, i, "bullish"):
            return "bullish"
        if dt == "bearish" and _ma_bear(feat, i) and _donchian_confirm(series, i, "bearish"):
            return "bearish"
        return None

    if key == "F1":
        dt = _daily_trend(daily, int(series.timestamps[i]))
        if dt == "bullish" and _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if dt == "bearish" and _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if key == "F2":
        fade = _breakout_fail_fade(series, i)
        if fade == "bullish" and (_ma_bull(feat, i) or feat["structure"][i] == 1):
            return "bullish"
        if fade == "bearish" and (_ma_bear(feat, i) or feat["structure"][i] == -1):
            return "bearish"
        return None

    if key == "F3":
        if not _atr_ok(feat, i) or i < 10:
            return None
        rsi = feat["rsi"][i]
        if np.isnan(rsi):
            return None
        mom = float(series.close[i]) - float(series.close[i - 10])
        if _ma_bull(feat, i) and rsi >= 52 and mom > 0 and float(series.close[i]) >= float(
            series.open[i]
        ):
            return "bullish"
        if _ma_bear(feat, i) and rsi <= 48 and mom < 0 and float(series.close[i]) <= float(
            series.open[i]
        ):
            return "bearish"
        return None

    if key == "B1":
        return _raw_donchian(series, i)

    if key == "B2":
        # Continuously allow entries while stacked (baseline needs trades)
        if _ma_bull(feat, i) and float(series.close[i]) >= float(series.open[i]):
            return "bullish"
        if _ma_bear(feat, i) and float(series.close[i]) <= float(series.open[i]):
            return "bearish"
        return None

    return None


def backtest_family(
    series: CandleSeries,
    family: V8Family,
    *,
    daily: Optional[CandleSeries] = None,
    spy: Optional[CandleSeries] = None,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V8_ATR_STOP_MULT,
    max_hold: int = V8_MAX_HOLD,
    entry_delay: int = 0,
) -> list[V2Trade]:
    feat = _feat_cached(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, V8_LOOKBACK + 5, V8_RS_LOOKBACK + 5)
    n = len(series)
    i = max(warmup, start_idx or warmup)
    last_start = n - max_hold if end_idx_exclusive is None else min(n - max_hold, end_idx_exclusive)
    trades: list[V2Trade] = []

    while i < last_start:
        direction = _signal_at(family, series, feat, i, daily, spy)
        if direction is None:
            i += 1
            continue
        entry_i = i + entry_delay
        if entry_i >= last_start or entry_i >= n:
            i += 1
            continue
        atr0 = feat["atr"][entry_i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        raw = float(series.close[entry_i])
        slip = entry_slip_atr * float(atr0)
        entry = raw + slip if direction == "bullish" else raw - slip
        atr_x = float(atr0) * (atr_stop_mult / V8_ATR_STOP_MULT)
        exit_idx, exit_px, reason, stop_dist = _adaptive_exit(
            series, feat, entry_i, direction, entry, atr_x
        )
        if exit_idx > entry_i + max_hold:
            exit_idx = min(entry_i + max_hold, n - 1)
            exit_px = float(series.close[exit_idx])
            reason = "max_hold"
            stop_dist = atr_stop_mult * float(atr0)
        trades.append(
            _make_trade(
                series=series,
                stage=family.name,
                direction=direction,
                confidence="V8",
                score=0,
                entry_idx=entry_i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=family.key,
                regime="trending",
                exit_reason=reason,
                feature_flags={"baseline": int(family.is_baseline), "delay": entry_delay},
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_family_on_map(
    series_4h: dict[tuple[str, str], CandleSeries],
    family: V8Family,
    instruments: tuple[str, ...] | list[str],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    spy: Optional[CandleSeries] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V8_ATR_STOP_MULT,
    max_hold: int = V8_MAX_HOLD,
    entry_delay: int = 0,
) -> list[V2Trade]:
    trades: list[V2Trade] = []
    for key in instruments:
        series = series_4h.get((key, "4h"))
        if series is None:
            continue
        n = len(series)
        daily = (daily_map or {}).get(key)
        trades.extend(
            backtest_family(
                series,
                family,
                daily=daily,
                spy=spy,
                start_idx=int(n * start_frac),
                end_idx_exclusive=int(n * end_frac),
                cost_mult=cost_mult,
                entry_slip_atr=entry_slip_atr,
                atr_stop_mult=atr_stop_mult,
                max_hold=max_hold,
                entry_delay=entry_delay,
            )
        )
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades


def folds_for_family(
    series_4h: dict,
    family: V8Family,
    instruments: tuple[str, ...] | list[str],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    spy: Optional[CandleSeries] = None,
    n_folds: int = 4,
    cost_mult: float = 1.0,
) -> list[dict]:
    out = []
    for k in range(n_folds):
        start, end = k / n_folds, (k + 1) / n_folds
        trades = run_family_on_map(
            series_4h,
            family,
            instruments,
            daily_map=daily_map,
            spy=spy,
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out
