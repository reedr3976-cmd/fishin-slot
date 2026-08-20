"""V11 causal market context — liquidity/MTF/FVG/SR/regime refinements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import V11_ADX_STRONG, V11_ADX_WEAK, V11_FVG_MAX_AGE, V11_PIVOT, V11_SR_ZONE_ATR
from indicators import adx, swing_structure_dir
from models import CandleSeries
from backtest.market_context import (
    FVGZone,
    _cluster_levels,
    _confirmed_pivot_indices,
    _detect_fvg_at,
    _htf_dir,
    _htf_levels,
    _mitigated,
    build_context_arrays,
)
from backtest.market_context import ContextArrays as V10Context


@dataclass
class V11Context:
    v10: V10Context
    daily_class: np.ndarray  # -1 bear, 0 range/neutral, 1 bull
    weekly_class: np.ndarray
    liq_sweep_htf_bull: np.ndarray
    liq_sweep_htf_bear: np.ndarray
    bos_after_sweep_bull: np.ndarray
    bos_after_sweep_bear: np.ndarray
    liq_at_sr_bull: np.ndarray
    liq_at_sr_bear: np.ndarray
    fvg_fresh_bull: np.ndarray
    fvg_fresh_bear: np.ndarray
    fvg_partial_bull: np.ndarray
    fvg_partial_bear: np.ndarray
    fvg_htf_bull: np.ndarray
    fvg_htf_bear: np.ndarray
    fvg_after_sweep_bull: np.ndarray
    fvg_after_sweep_bear: np.ndarray
    fvg_after_bos_bull: np.ndarray
    fvg_after_bos_bear: np.ndarray
    sr_zone_bull: np.ndarray
    sr_zone_bear: np.ndarray
    sr_flip_bull: np.ndarray
    sr_flip_bear: np.ndarray
    regime_v11: np.ndarray
    trend_eligible: np.ndarray
    htf_fvg_bull: np.ndarray
    htf_fvg_bear: np.ndarray


_V11_CACHE: dict[tuple, V11Context] = {}


def clear_v11_cache() -> None:
    _V11_CACHE.clear()


def _htf_class(series: Optional[CandleSeries], ts: int) -> int:
    """Classify completed HTF structure: bull / bear / range."""
    if series is None or len(series) < 30:
        return 0
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < 20:
        return 0
    sub_h = series.high[: idx + 1]
    sub_l = series.low[: idx + 1]
    sub_c = series.close[: idx + 1]
    sdir = swing_structure_dir(sub_h, sub_l, idx, pivot=V11_PIVOT)
    adx_arr, _, _ = adx(sub_h, sub_l, sub_c, 14)
    adx_v = float(adx_arr[idx]) if not np.isnan(adx_arr[idx]) else 0.0
    if sdir == 1 and adx_v >= V11_ADX_WEAK:
        return 1
    if sdir == -1 and adx_v >= V11_ADX_WEAK:
        return -1
    return 0


def _partial_touch(zone: FVGZone, high: np.ndarray, low: np.ndarray, i: int) -> bool:
    for j in range(zone.birth_idx + 1, i + 1):
        if zone.direction == "bullish" and float(low[j]) <= zone.top and float(low[j]) > zone.bottom:
            return True
        if zone.direction == "bearish" and float(high[j]) >= zone.bottom and float(high[j]) < zone.top:
            return True
    return False


def _build_htf_fvgs(daily: Optional[CandleSeries]) -> list[FVGZone]:
    if daily is None or len(daily) < 5:
        return []
    out: list[FVGZone] = []
    for i in range(2, len(daily)):
        z = _detect_fvg_at(daily.high, daily.low, i)
        if z:
            out.append(z)
    return out


def build_v11_context(
    series: CandleSeries,
    *,
    daily: Optional[CandleSeries] = None,
    weekly: Optional[CandleSeries] = None,
) -> V11Context:
    key = (id(series), id(daily), id(weekly))
    if key in _V11_CACHE:
        return _V11_CACHE[key]

    v10 = build_context_arrays(series, daily=daily, weekly=weekly)
    n = len(series)
    high, low, close, ts, open_ = series.high, series.low, series.close, series.timestamps, series.open

    daily_class = np.zeros(n, dtype=np.int8)
    weekly_class = np.zeros(n, dtype=np.int8)
    liq_sweep_htf_bull = np.zeros(n, dtype=bool)
    liq_sweep_htf_bear = np.zeros(n, dtype=bool)
    bos_after_sweep_bull = np.zeros(n, dtype=bool)
    bos_after_sweep_bear = np.zeros(n, dtype=bool)
    liq_at_sr_bull = np.zeros(n, dtype=bool)
    liq_at_sr_bear = np.zeros(n, dtype=bool)
    fvg_fresh_bull = np.zeros(n, dtype=bool)
    fvg_fresh_bear = np.zeros(n, dtype=bool)
    fvg_partial_bull = np.zeros(n, dtype=bool)
    fvg_partial_bear = np.zeros(n, dtype=bool)
    fvg_htf_bull = np.zeros(n, dtype=bool)
    fvg_htf_bear = np.zeros(n, dtype=bool)
    fvg_after_sweep_bull = np.zeros(n, dtype=bool)
    fvg_after_sweep_bear = np.zeros(n, dtype=bool)
    fvg_after_bos_bull = np.zeros(n, dtype=bool)
    fvg_after_bos_bear = np.zeros(n, dtype=bool)
    sr_zone_bull = np.zeros(n, dtype=bool)
    sr_zone_bear = np.zeros(n, dtype=bool)
    sr_flip_bull = np.zeros(n, dtype=bool)
    sr_flip_bear = np.zeros(n, dtype=bool)
    regime_v11 = np.zeros(n, dtype=np.int8)
    trend_eligible = np.zeros(n, dtype=bool)
    htf_fvg_bull = np.zeros(n, dtype=bool)
    htf_fvg_bear = np.zeros(n, dtype=bool)

    active_fvgs: list[FVGZone] = []
    htf_fvgs = _build_htf_fvgs(daily)
    broken_res: list[float] = []
    broken_sup: list[float] = []

    for i in range(n):
        t = int(ts[i])
        daily_class[i] = _htf_class(daily, t)
        weekly_class[i] = _htf_class(weekly, t)
        atr_v = float(v10.atr[i]) if not np.isnan(v10.atr[i]) else np.nan
        adx_v = float(v10.adx[i]) if not np.isnan(v10.adx[i]) else 0.0
        price = float(close[i])

        # Regime v11
        td = int(v10.trend_dir[i])
        rc10 = int(v10.regime_code[i])
        if v10.is_ranging[i] or td == 0:
            regime_v11[i] = 0
        elif rc10 == 3:
            regime_v11[i] = 5
        elif rc10 == 4:
            regime_v11[i] = 6
        elif td == 1 and adx_v >= V11_ADX_STRONG:
            regime_v11[i] = 3
        elif td == -1 and adx_v >= V11_ADX_STRONG:
            regime_v11[i] = 4
        elif td == 1 and adx_v >= V11_ADX_WEAK:
            regime_v11[i] = 1
        elif td == -1 and adx_v >= V11_ADX_WEAK:
            regime_v11[i] = 2
        else:
            regime_v11[i] = 0
        trend_eligible[i] = regime_v11[i] in (1, 2, 3, 4, 5)

        sh, sl = _confirmed_pivot_indices(high, low, i, pivot=V11_PIVOT)
        d_hi, d_lo = _htf_levels(daily, t)
        w_hi, w_lo = _htf_levels(weekly, t)

        if atr_v > 0:
            sup_z = _cluster_levels([p for _, p in sl] + d_lo + w_lo, price, atr_v, "support")
            res_z = _cluster_levels([p for _, p in sh] + d_hi + w_hi, price, atr_v, "resistance")

            # Track breakout flips (causal)
            if sh and i >= 2:
                lvl = sh[-1][1]
                if float(close[i - 1]) > lvl and float(close[i]) > lvl:
                    broken_res.append(lvl)
            if sl and i >= 2:
                lvl = sl[-1][1]
                if float(close[i - 1]) < lvl and float(close[i]) < lvl:
                    broken_sup.append(lvl)

            # Strict SR zone touch: wick into zone + close rejection + HTF not opposing
            zone_tol = V11_SR_ZONE_ATR * atr_v
            if sup_z and abs(float(low[i]) - sup_z[0]) <= zone_tol:
                if float(close[i]) > float(open_[i]) and daily_class[i] >= 0:
                    sr_zone_bull[i] = True
            if res_z and abs(float(high[i]) - res_z[0]) <= zone_tol:
                if float(close[i]) < float(open_[i]) and daily_class[i] <= 0:
                    sr_zone_bear[i] = True

            # Flip levels: broken resistance retest as support (and vice versa)
            for lvl in broken_res[-8:]:
                if abs(float(low[i]) - lvl) <= zone_tol and float(close[i]) > lvl and daily_class[i] >= 0:
                    sr_flip_bull[i] = True
            for lvl in broken_sup[-8:]:
                if abs(float(high[i]) - lvl) <= zone_tol and float(close[i]) < lvl and daily_class[i] <= 0:
                    sr_flip_bear[i] = True

        # HTF-aligned liquidity sweeps
        if v10.liq_sweep_bull[i] and daily_class[i] == 1:
            liq_sweep_htf_bull[i] = True
        if v10.liq_sweep_bear[i] and daily_class[i] == -1:
            liq_sweep_htf_bear[i] = True

        if v10.liq_sweep_bull[i] and atr_v > 0 and (d_lo or w_lo):
            zones = (d_lo or []) + (w_lo or [])
            if any(abs(float(low[i]) - z) <= V11_SR_ZONE_ATR * atr_v for z in zones):
                liq_at_sr_bull[i] = True
        if v10.liq_sweep_bear[i] and atr_v > 0 and (d_hi or w_hi):
            zones = (d_hi or []) + (w_hi or [])
            if any(abs(float(high[i]) - z) <= V11_SR_ZONE_ATR * atr_v for z in zones):
                liq_at_sr_bear[i] = True

        for k in (1, 2, 3):
            if i >= k:
                if v10.bos_bull[i] and v10.liq_sweep_bull[i - k]:
                    bos_after_sweep_bull[i] = True
                if v10.bos_bear[i] and v10.liq_sweep_bear[i - k]:
                    bos_after_sweep_bear[i] = True

        # FVG tracking with fresh/partial
        new_fvg = _detect_fvg_at(high, low, i)
        if new_fvg:
            active_fvgs.append(new_fvg)
        active_fvgs = [
            z for z in active_fvgs if i - z.birth_idx <= V11_FVG_MAX_AGE and not _mitigated(z, high, low, i)
        ]
        if atr_v > 0:
            for z in active_fvgs:
                in_zone = z.bottom <= price <= z.top if z.direction == "bullish" else z.bottom <= price <= z.top
                dist = abs(price - (z.top + z.bottom) / 2) / atr_v
                if dist > 1.5 and not in_zone:
                    continue
                partial = _partial_touch(z, high, low, i)
                fresh = not partial
                if z.direction == "bullish" and td >= 0:
                    if fresh:
                        fvg_fresh_bull[i] = True
                    elif partial:
                        fvg_partial_bull[i] = True
                    if daily_class[i] == 1:
                        fvg_htf_bull[i] = True
                if z.direction == "bearish" and td <= 0:
                    if fresh:
                        fvg_fresh_bear[i] = True
                    elif partial:
                        fvg_partial_bear[i] = True
                    if daily_class[i] == -1:
                        fvg_htf_bear[i] = True

        for k in (0, 1, 2, 3):
            if i >= k:
                if (fvg_fresh_bull[i] or fvg_partial_bull[i]) and v10.liq_sweep_bull[i - k]:
                    fvg_after_sweep_bull[i] = True
                if (fvg_fresh_bear[i] or fvg_partial_bear[i]) and v10.liq_sweep_bear[i - k]:
                    fvg_after_sweep_bear[i] = True
                if (fvg_fresh_bull[i] or fvg_partial_bull[i]) and v10.bos_bull[i - k]:
                    fvg_after_bos_bull[i] = True
                if (fvg_fresh_bear[i] or fvg_partial_bear[i]) and v10.bos_bear[i - k]:
                    fvg_after_bos_bear[i] = True

        # Map daily FVG zones to 4H bar (causal)
        if daily is not None and atr_v > 0:
            d_idx = int(np.searchsorted(daily.timestamps, t, side="right") - 1)
            for z in htf_fvgs:
                if z.birth_idx > d_idx:
                    continue
                if _mitigated(z, daily.high, daily.low, d_idx):
                    continue
                if z.direction == "bullish" and z.bottom <= price <= z.top and daily_class[i] == 1:
                    htf_fvg_bull[i] = True
                if z.direction == "bearish" and z.bottom <= price <= z.top and daily_class[i] == -1:
                    htf_fvg_bear[i] = True

    ctx = V11Context(
        v10=v10,
        daily_class=daily_class,
        weekly_class=weekly_class,
        liq_sweep_htf_bull=liq_sweep_htf_bull,
        liq_sweep_htf_bear=liq_sweep_htf_bear,
        bos_after_sweep_bull=bos_after_sweep_bull,
        bos_after_sweep_bear=bos_after_sweep_bear,
        liq_at_sr_bull=liq_at_sr_bull,
        liq_at_sr_bear=liq_at_sr_bear,
        fvg_fresh_bull=fvg_fresh_bull,
        fvg_fresh_bear=fvg_fresh_bear,
        fvg_partial_bull=fvg_partial_bull,
        fvg_partial_bear=fvg_partial_bear,
        fvg_htf_bull=fvg_htf_bull,
        fvg_htf_bear=fvg_htf_bear,
        fvg_after_sweep_bull=fvg_after_sweep_bull,
        fvg_after_sweep_bear=fvg_after_sweep_bear,
        fvg_after_bos_bull=fvg_after_bos_bull,
        fvg_after_bos_bear=fvg_after_bos_bear,
        sr_zone_bull=sr_zone_bull,
        sr_zone_bear=sr_zone_bear,
        sr_flip_bull=sr_flip_bull,
        sr_flip_bear=sr_flip_bear,
        regime_v11=regime_v11,
        trend_eligible=trend_eligible,
        htf_fvg_bull=htf_fvg_bull,
        htf_fvg_bear=htf_fvg_bear,
    )
    _V11_CACHE[key] = ctx
    return ctx


REGIME_V11_LABELS = {
    0: "range",
    1: "weak_trend_bull",
    2: "weak_trend_bear",
    3: "strong_trend_bull",
    4: "strong_trend_bear",
    5: "volatility_expansion",
    6: "volatility_contraction",
}
