"""Scanner V6 — CLEAN STRATEGY-FAMILY RESET (research only).

V4/V5 hypotheses are treated as falsified. Do not retune V4_S1_STOCK.
Research stocks, commodities, and forex as separate families.

Does NOT modify live ORIGINAL. No paper/live enablement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V6_ATR_STOP_MULT,
    V6_LOOKBACK,
    V6_MAX_HOLD,
    V6_VOL_ATR_MULT,
)
from models import CandleSeries
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade, _precompute
from backtest.scanner_v3 import _breakout_dir, _simple_regime
from indicators import swing_structure_dir


@dataclass(frozen=True)
class V6Family:
    key: str
    name: str
    notes: str


FAMILIES: tuple[V6Family, ...] = (
    V6Family(
        "A",
        "V6_A_MA_PULLBACK",
        "MA trend (SMA20/50 + slope) + pullback-to-SMA20 continuation.",
    ),
    V6Family(
        "B",
        "V6_B_DONCHIAN_CONFIRM",
        "Structure+MA regime; 20-bar breakout with next-bar confirmation.",
    ),
    V6Family(
        "C",
        "V6_C_VOL_EXPANSION",
        "ATR expansion vs 50-bar median + MA-aligned Donchian breakout.",
    ),
    V6Family(
        "D",
        "V6_D_MOMENTUM_TREND",
        "SMA stack + RSI side + 10-bar momentum continuation.",
    ),
    V6Family(
        "E",
        "V6_E_MTF_DAILY_4H",
        "Daily SMA20/50 trend filter + 4H MA pullback entry.",
    ),
)


def _feat(series: CandleSeries) -> dict:
    feat = dict(_precompute(series))
    n = len(series)
    structure = np.zeros(n, dtype=np.int8)
    for i in range(n):
        structure[i] = swing_structure_dir(series.high, series.low, i)
    feat["structure"] = structure
    return feat


# Cache causal features per series object id (research runs only)
_FEAT_CACHE: dict[int, dict] = {}


def _feat_cached(series: CandleSeries) -> dict:
    key = id(series)
    if key not in _FEAT_CACHE:
        _FEAT_CACHE[key] = _feat(series)
    return _FEAT_CACHE[key]


def _ma_bull(feat: dict, i: int) -> bool:
    f, s, slope = feat["sma_fast"][i], feat["sma_slow"][i], feat["sma_slope"][i]
    if any(np.isnan(x) for x in (f, s, slope)):
        return False
    return f > s and slope > 0


def _ma_bear(feat: dict, i: int) -> bool:
    f, s, slope = feat["sma_fast"][i], feat["sma_slow"][i], feat["sma_slope"][i]
    if any(np.isnan(x) for x in (f, s, slope)):
        return False
    return f < s and slope < 0


def _pullback(series: CandleSeries, feat: dict, i: int, direction: str) -> bool:
    sma20 = feat["sma_fast"][i]
    if np.isnan(sma20) or i < 2:
        return False
    if direction == "bullish":
        touched = float(series.low[i]) <= sma20 * 1.001 or float(series.low[i - 1]) <= float(
            feat["sma_fast"][i - 1]
        ) * 1.001
        return touched and float(series.close[i]) > sma20 and float(series.close[i]) >= float(
            series.open[i]
        )
    touched = float(series.high[i]) >= sma20 * 0.999 or float(series.high[i - 1]) >= float(
        feat["sma_fast"][i - 1]
    ) * 0.999
    return touched and float(series.close[i]) < sma20 and float(series.close[i]) <= float(
        series.open[i]
    )


def _donchian_confirm(series: CandleSeries, i: int, direction: str, look: int = V6_LOOKBACK) -> bool:
    if i < look + 2:
        return False
    if direction == "bullish":
        prior = float(np.max(series.high[i - look - 1 : i - 1]))
        broke = float(series.close[i - 1]) > prior
        holds = float(series.close[i]) > prior and float(series.close[i]) >= float(series.open[i])
        return broke and holds
    prior = float(np.min(series.low[i - look - 1 : i - 1]))
    broke = float(series.close[i - 1]) < prior
    holds = float(series.close[i]) < prior and float(series.close[i]) <= float(series.open[i])
    return broke and holds


def _vol_expansion(feat: dict, i: int) -> bool:
    atr = feat["atr"][i]
    med = feat["atr_pct_med50"][i]
    # atr_pct_med50 is median of atr_pct; compare atr_pct to med
    ap = feat["atr_pct"][i]
    if any(np.isnan(x) for x in (atr, med, ap)) or med <= 0:
        return False
    return ap >= V6_VOL_ATR_MULT * med


def _momentum_ok(series: CandleSeries, feat: dict, i: int, direction: str) -> bool:
    rsi = feat["rsi"][i]
    if np.isnan(rsi) or i < 10:
        return False
    mom = float(series.close[i]) - float(series.close[i - 10])
    if direction == "bullish":
        return rsi >= 55 and mom > 0 and _ma_bull(feat, i)
    return rsi <= 45 and mom < 0 and _ma_bear(feat, i)


def _daily_trend(daily: Optional[CandleSeries], ts: int) -> Optional[str]:
    """Causal daily SMA stack at last daily bar with timestamp <= ts."""
    if daily is None or len(daily) < SMA_SLOW + 5:
        return None
    # find last daily index with timestamp <= ts
    idx = int(np.searchsorted(daily.timestamps, ts, side="right") - 1)
    if idx < SMA_SLOW:
        return None
    # use only bars <= idx
    close = daily.close[: idx + 1]
    from indicators import sma

    f = sma(close, 20)
    s = sma(close, 50)
    if np.isnan(f[-1]) or np.isnan(s[-1]):
        return None
    if f[-1] > s[-1]:
        return "bullish"
    if f[-1] < s[-1]:
        return "bearish"
    return None


def _signal_at(
    family: V6Family,
    series: CandleSeries,
    feat: dict,
    i: int,
    daily: Optional[CandleSeries],
) -> Optional[str]:
    if family.key == "A":
        if _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if family.key == "B":
        regime = _simple_regime(
            feat, i, require_structure=True, require_ma=True, require_adx=False
        )
        if regime in ("bullish", "bearish") and _donchian_confirm(series, i, regime):
            return regime
        return None

    if family.key == "C":
        if not _vol_expansion(feat, i):
            return None
        if _ma_bull(feat, i):
            d = _breakout_dir(series, i, lookback=V6_LOOKBACK)
            if d == "bullish":
                return "bullish"
        if _ma_bear(feat, i):
            d = _breakout_dir(series, i, lookback=V6_LOOKBACK)
            if d == "bearish":
                return "bearish"
        return None

    if family.key == "D":
        if _momentum_ok(series, feat, i, "bullish"):
            # enter on bullish continuation candle
            if float(series.close[i]) >= float(series.open[i]):
                return "bullish"
        if _momentum_ok(series, feat, i, "bearish"):
            if float(series.close[i]) <= float(series.open[i]):
                return "bearish"
        return None

    if family.key == "E":
        dt = _daily_trend(daily, int(series.timestamps[i]))
        if dt == "bullish" and _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if dt == "bearish" and _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    return None


def backtest_family(
    series: CandleSeries,
    family: V6Family,
    *,
    daily: Optional[CandleSeries] = None,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V6_ATR_STOP_MULT,
    lookback: int = V6_LOOKBACK,  # reserved for sensitivity hooks
    max_hold: int = V6_MAX_HOLD,
) -> list[V2Trade]:
    _ = lookback  # families use module constants; sensitivity may patch via globals later
    feat = _feat(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, V6_LOOKBACK + 5)
    n = len(series)
    i = max(warmup, start_idx or warmup)
    last_start = n - max_hold if end_idx_exclusive is None else min(n - max_hold, end_idx_exclusive)
    trades: list[V2Trade] = []

    while i < last_start:
        direction = _signal_at(family, series, feat, i, daily)
        if direction is None:
            i += 1
            continue
        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        raw = float(series.close[i])
        slip = entry_slip_atr * float(atr0)
        entry = raw + slip if direction == "bullish" else raw - slip
        atr_x = float(atr0) * (atr_stop_mult / V6_ATR_STOP_MULT)
        exit_idx, exit_px, reason, stop_dist = _adaptive_exit(
            series, feat, i, direction, entry, atr_x
        )
        if exit_idx > i + max_hold:
            exit_idx = min(i + max_hold, n - 1)
            exit_px = float(series.close[exit_idx])
            reason = "max_hold"
            stop_dist = atr_stop_mult * float(atr0)
        trades.append(
            _make_trade(
                series=series,
                stage=family.name,
                direction=direction,
                confidence="V6",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=family.key,
                regime="trending",
                exit_reason=reason,
                feature_flags={"family_a": int(family.key == "A"), "family_e": int(family.key == "E")},
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_family_on_map(
    series_4h: dict[tuple[str, str], CandleSeries],
    family: V6Family,
    instruments: tuple[str, ...] | list[str],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V6_ATR_STOP_MULT,
    max_hold: int = V6_MAX_HOLD,
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
                start_idx=int(n * start_frac),
                end_idx_exclusive=int(n * end_frac),
                cost_mult=cost_mult,
                entry_slip_atr=entry_slip_atr,
                atr_stop_mult=atr_stop_mult,
                max_hold=max_hold,
            )
        )
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades


def folds_for_family(
    series_4h: dict,
    family: V6Family,
    instruments: tuple[str, ...] | list[str],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
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
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out
