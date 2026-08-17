"""Causal market context + price structure for V10 research.

All features at bar index i use only candles with timestamp <= series.timestamps[i].
Higher-timeframe bars must be completed (last HTF bar with ts <= decision ts).
No future data for S/R, FVG, or swing confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import V10_ADX_MIN, V10_COMPRESS_MULT, V10_FVG_MAX_AGE, V10_PIVOT, V10_SR_CLUSTER_ATR, V10_VOL_ATR_MULT
from indicators import adx, atr, sma, swing_structure_dir
from models import CandleSeries


@dataclass(frozen=True)
class FVGZone:
    direction: str  # bullish | bearish
    top: float
    bottom: float
    birth_idx: int
    mitigated_idx: Optional[int] = None

    @property
    def size(self) -> float:
        return self.top - self.bottom


def _confirmed_pivot_indices(
    high: np.ndarray, low: np.ndarray, i: int, *, pivot: int = V10_PIVOT, max_look: int = 120
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Swing highs/lows confirmed as of bar i (right edge = i - pivot)."""
    if i < pivot * 2 + 1:
        return [], []
    start = max(pivot, i - max_look)
    last = i - pivot
    swing_h: list[tuple[int, float]] = []
    swing_l: list[tuple[int, float]] = []
    for j in range(start + pivot, last + 1):
        wh = high[j - pivot : j + pivot + 1]
        wl = low[j - pivot : j + pivot + 1]
        if high[j] >= np.max(wh):
            swing_h.append((j, float(high[j])))
        if low[j] <= np.min(wl):
            swing_l.append((j, float(low[j])))
    return swing_h, swing_l


def _htf_dir(series: Optional[CandleSeries], ts: int, *, pivot: int = V10_PIVOT) -> int:
    """+1 bullish structure, -1 bearish, 0 unknown/neutral on completed HTF bars."""
    if series is None or len(series) < 60:
        return 0
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < pivot * 4 + 2:
        return 0
    return swing_structure_dir(series.high[: idx + 1], series.low[: idx + 1], idx, pivot=pivot)


def _htf_levels(series: Optional[CandleSeries], ts: int, n_levels: int = 5) -> tuple[list[float], list[float]]:
    """Recent completed HTF swing highs/lows available at ts."""
    if series is None:
        return [], []
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < 10:
        return [], []
    sh, sl = _confirmed_pivot_indices(series.high, series.low, idx)
    highs = [p for _, p in sh[-n_levels:]]
    lows = [p for _, p in sl[-n_levels:]]
    if idx >= 20:
        highs.append(float(np.max(series.high[max(0, idx - 20) : idx + 1])))
        lows.append(float(np.min(series.low[max(0, idx - 20) : idx + 1])))
    return highs, lows


def _detect_fvg_at(
    high: np.ndarray, low: np.ndarray, i: int
) -> Optional[FVGZone]:
    """Strict 3-candle FVG at bar i (causal — gap formed at i)."""
    if i < 2:
        return None
    if float(low[i]) > float(high[i - 2]):
        return FVGZone("bullish", float(low[i]), float(high[i - 2]), i)
    if float(high[i]) < float(low[i - 2]):
        return FVGZone("bearish", float(low[i - 2]), float(high[i]), i)
    return None


def _mitigated(zone: FVGZone, high: np.ndarray, low: np.ndarray, i: int) -> bool:
    for j in range(zone.birth_idx + 1, i + 1):
        if zone.direction == "bullish" and float(low[j]) <= zone.bottom:
            return True
        if zone.direction == "bearish" and float(high[j]) >= zone.top:
            return True
    return False


def _cluster_levels(levels: list[float], price: float, atr_v: float, side: str) -> list[float]:
    if not levels or atr_v <= 0 or np.isnan(atr_v):
        return []
    tol = V10_SR_CLUSTER_ATR * atr_v
    filtered = sorted({round(x, 8) for x in levels if (x < price if side == "support" else x > price)})
    if not filtered:
        return []
    clusters: list[list[float]] = [[filtered[0]]]
    for lv in filtered[1:]:
        if abs(lv - clusters[-1][-1]) <= tol:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [float(np.mean(c)) for c in clusters]


@dataclass
class ContextArrays:
    trend_dir: np.ndarray
    bos_bull: np.ndarray
    bos_bear: np.ndarray
    choch_bull: np.ndarray
    choch_bear: np.ndarray
    is_ranging: np.ndarray
    failed_break_up: np.ndarray
    failed_break_dn: np.ndarray
    regime_code: np.ndarray  # 0=ranging 1=trend_bull 2=trend_bear 3=vol_exp 4=vol_contract
    daily_dir: np.ndarray
    weekly_dir: np.ndarray
    support_dist_atr: np.ndarray
    resist_dist_atr: np.ndarray
    sr_retest_bull: np.ndarray
    sr_retest_bear: np.ndarray
    fvg_near_bull: np.ndarray
    fvg_near_bear: np.ndarray
    fvg_align_daily: np.ndarray
    liq_sweep_bull: np.ndarray
    liq_sweep_bear: np.ndarray
    liq_break_bull: np.ndarray
    liq_break_bear: np.ndarray
    atr: np.ndarray
    adx: np.ndarray


_CTX_CACHE: dict[tuple, ContextArrays] = {}


def clear_context_cache() -> None:
    _CTX_CACHE.clear()


def build_context_arrays(
    series: CandleSeries,
    *,
    daily: Optional[CandleSeries] = None,
    weekly: Optional[CandleSeries] = None,
) -> ContextArrays:
    key = (id(series), id(daily), id(weekly))
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]

    n = len(series)
    high, low, close, ts = series.high, series.low, series.close, series.timestamps
    atr_arr = atr(high, low, close, 14)
    adx_arr, pdi, mdi = adx(high, low, close, 14)
    atr_pct = np.divide(atr_arr, close, out=np.full(n, np.nan), where=close > 0)
    med50 = np.full(n, np.nan)
    for i in range(50, n):
        w = atr_pct[i - 50 : i]
        v = w[~np.isnan(w)]
        if len(v):
            med50[i] = float(np.median(v))

    trend_dir = np.zeros(n, dtype=np.int8)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)
    is_ranging = np.zeros(n, dtype=bool)
    failed_break_up = np.zeros(n, dtype=bool)
    failed_break_dn = np.zeros(n, dtype=bool)
    regime_code = np.zeros(n, dtype=np.int8)
    daily_dir = np.zeros(n, dtype=np.int8)
    weekly_dir = np.zeros(n, dtype=np.int8)
    support_dist_atr = np.full(n, np.nan)
    resist_dist_atr = np.full(n, np.nan)
    sr_retest_bull = np.zeros(n, dtype=bool)
    sr_retest_bear = np.zeros(n, dtype=bool)
    fvg_near_bull = np.zeros(n, dtype=bool)
    fvg_near_bear = np.zeros(n, dtype=bool)
    fvg_align_daily = np.zeros(n, dtype=bool)
    liq_sweep_bull = np.zeros(n, dtype=bool)
    liq_sweep_bear = np.zeros(n, dtype=bool)
    liq_break_bull = np.zeros(n, dtype=bool)
    liq_break_bear = np.zeros(n, dtype=bool)

    active_fvgs: list[FVGZone] = []
    prev_trend = 0

    for i in range(n):
        t = int(ts[i])
        trend_dir[i] = swing_structure_dir(high, low, i)
        daily_dir[i] = _htf_dir(daily, t)
        weekly_dir[i] = _htf_dir(weekly, t)

        sh, sl = _confirmed_pivot_indices(high, low, i)
        atr_v = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else np.nan

        # BOS / CHoCH
        if len(sh) >= 2 and not np.isnan(close[i]):
            last_sh = sh[-1][1]
            prior_sh = sh[-2][1]
            if float(close[i]) > last_sh and float(close[i - 1]) <= last_sh if i > 0 else False:
                bos_bull[i] = True
            if prev_trend == -1 and float(close[i]) > prior_sh:
                choch_bull[i] = True
        if len(sl) >= 2 and not np.isnan(close[i]):
            last_sl = sl[-1][1]
            prior_sl = sl[-2][1]
            if float(close[i]) < last_sl and float(close[i - 1]) >= last_sl if i > 0 else False:
                bos_bear[i] = True
            if prev_trend == 1 and float(close[i]) < prior_sl:
                choch_bear[i] = True
        prev_trend = int(trend_dir[i])

        # Range: narrow recent range vs ATR OR low ADX
        adx_v = float(adx_arr[i]) if not np.isnan(adx_arr[i]) else 0.0
        look = min(i, 20)
        if look >= 5 and atr_v > 0:
            rng = float(np.max(high[i - look : i + 1]) - np.min(low[i - look : i + 1]))
            is_ranging[i] = adx_v < V10_ADX_MIN or (trend_dir[i] == 0 and rng < 3.0 * atr_v)
        else:
            is_ranging[i] = adx_v < V10_ADX_MIN

        # Failed breakout (2-bar lookback)
        if i >= 22 and atr_v > 0:
            prior_hi = float(np.max(high[i - 21 : i - 1]))
            prior_lo = float(np.min(low[i - 21 : i - 1]))
            for b in (i - 1, i - 2):
                if b < 1:
                    continue
                if float(close[b]) > prior_hi and float(close[i]) < prior_hi:
                    failed_break_up[i] = True
                if float(close[b]) < prior_lo and float(close[i]) > prior_lo:
                    failed_break_dn[i] = True

        # Regime classification
        ap, med = atr_pct[i], med50[i]
        vol_exp = bool(not np.isnan(ap) and not np.isnan(med) and med > 0 and ap >= V10_VOL_ATR_MULT * med)
        vol_contract = bool(not np.isnan(ap) and not np.isnan(med) and med > 0 and ap <= V10_COMPRESS_MULT * med)
        if is_ranging[i]:
            regime_code[i] = 0
        elif vol_exp and trend_dir[i] == 1:
            regime_code[i] = 3
        elif vol_exp and trend_dir[i] == -1:
            regime_code[i] = 3
        elif vol_contract:
            regime_code[i] = 4
        elif trend_dir[i] == 1 and adx_v >= V10_ADX_MIN:
            regime_code[i] = 1
        elif trend_dir[i] == -1 and adx_v >= V10_ADX_MIN:
            regime_code[i] = 2
        else:
            regime_code[i] = 0

        # S/R from 4H swings + HTF levels
        d_hi, d_lo = _htf_levels(daily, t)
        w_hi, w_lo = _htf_levels(weekly, t)
        price = float(close[i])
        if atr_v > 0:
            sup_levels = _cluster_levels([p for _, p in sl] + d_lo + w_lo, price, atr_v, "support")
            res_levels = _cluster_levels([p for _, p in sh] + d_hi + w_hi, price, atr_v, "resistance")
            if sup_levels:
                support_dist_atr[i] = (price - sup_levels[0]) / atr_v
            if res_levels:
                resist_dist_atr[i] = (res_levels[0] - price) / atr_v
            # Retest: touched support/resistance within 0.5 ATR and rejection candle
            if sup_levels and abs(price - sup_levels[0]) <= 0.5 * atr_v:
                o = float(series.open[i])
                if float(close[i]) > o and float(low[i]) <= sup_levels[0] + 0.1 * atr_v:
                    sr_retest_bull[i] = True
            if res_levels and abs(price - res_levels[0]) <= 0.5 * atr_v:
                if float(close[i]) < float(series.open[i]) and float(high[i]) >= res_levels[0] - 0.1 * atr_v:
                    sr_retest_bear[i] = True

        # FVG tracking
        new_fvg = _detect_fvg_at(high, low, i)
        if new_fvg:
            active_fvgs.append(new_fvg)
        active_fvgs = [z for z in active_fvgs if i - z.birth_idx <= V10_FVG_MAX_AGE and not _mitigated(z, high, low, i)]
        if atr_v > 0:
            for z in active_fvgs:
                mid = (z.top + z.bottom) / 2.0
                dist = abs(price - mid) / atr_v
                if dist <= 1.5:
                    if z.direction == "bullish" and price >= z.bottom:
                        fvg_near_bull[i] = True
                        if daily_dir[i] == 1:
                            fvg_align_daily[i] = True
                    if z.direction == "bearish" and price <= z.top:
                        fvg_near_bear[i] = True
                        if daily_dir[i] == -1:
                            fvg_align_daily[i] = True

        # Liquidity sweep: wick beyond last swing then close back inside
        if sh and sl and i >= 1:
            last_sh_p = sh[-1][1]
            last_sl_p = sl[-1][1]
            if float(high[i]) > last_sh_p and float(close[i]) < last_sh_p and float(close[i]) <= float(series.open[i]):
                liq_sweep_bear[i] = True
            if float(low[i]) < last_sl_p and float(close[i]) > last_sl_p and float(close[i]) >= float(series.open[i]):
                liq_sweep_bull[i] = True

        # Breakout continuation: close beyond swing + bullish/bearish hold bar
        if sh and i >= 2:
            last_sh_p = sh[-1][1]
            if float(close[i - 1]) > last_sh_p and float(close[i]) > last_sh_p and float(close[i]) >= float(series.open[i]):
                liq_break_bull[i] = True
        if sl and i >= 2:
            last_sl_p = sl[-1][1]
            if float(close[i - 1]) < last_sl_p and float(close[i]) < last_sl_p and float(close[i]) <= float(series.open[i]):
                liq_break_bear[i] = True

    ctx = ContextArrays(
        trend_dir=trend_dir,
        bos_bull=bos_bull,
        bos_bear=bos_bear,
        choch_bull=choch_bull,
        choch_bear=choch_bear,
        is_ranging=is_ranging,
        failed_break_up=failed_break_up,
        failed_break_dn=failed_break_dn,
        regime_code=regime_code,
        daily_dir=daily_dir,
        weekly_dir=weekly_dir,
        support_dist_atr=support_dist_atr,
        resist_dist_atr=resist_dist_atr,
        sr_retest_bull=sr_retest_bull,
        sr_retest_bear=sr_retest_bear,
        fvg_near_bull=fvg_near_bull,
        fvg_near_bear=fvg_near_bear,
        fvg_align_daily=fvg_align_daily,
        liq_sweep_bull=liq_sweep_bull,
        liq_sweep_bear=liq_sweep_bear,
        liq_break_bull=liq_break_bull,
        liq_break_bear=liq_break_bear,
        atr=atr_arr,
        adx=adx_arr,
    )
    _CTX_CACHE[key] = ctx
    return ctx


REGIME_LABELS = {
    0: "ranging",
    1: "trending_bullish",
    2: "trending_bearish",
    3: "volatility_expansion",
    4: "volatility_contraction",
}


def snapshot_at(ctx: ContextArrays, i: int) -> dict:
    """Human-readable snapshot for scanner output design (research)."""
    rc = int(ctx.regime_code[i])
    return {
        "trend_dir": int(ctx.trend_dir[i]),
        "regime": REGIME_LABELS.get(rc, "unknown"),
        "daily_dir": int(ctx.daily_dir[i]),
        "weekly_dir": int(ctx.weekly_dir[i]),
        "support_dist_atr": float(ctx.support_dist_atr[i]) if not np.isnan(ctx.support_dist_atr[i]) else None,
        "resist_dist_atr": float(ctx.resist_dist_atr[i]) if not np.isnan(ctx.resist_dist_atr[i]) else None,
        "bos": bool(ctx.bos_bull[i] or ctx.bos_bear[i]),
        "fvg_context": bool(ctx.fvg_near_bull[i] or ctx.fvg_near_bear[i]),
    }
