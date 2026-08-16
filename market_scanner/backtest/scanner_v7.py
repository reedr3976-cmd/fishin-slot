"""Scanner V7 — ROBUSTNESS RESEARCH after V6 FAIL (research only).

Addresses V6 near-miss failure modes (stocks C thin-sample / period lottery;
commodities C single-symbol concentration) with new simple family combinations.

Does NOT modify live ORIGINAL. No paper/live enablement. No auto-promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V7_ADX_MIN,
    V7_ATR_STOP_MULT,
    V7_LOOKBACK,
    V7_MAX_HOLD,
    V7_VOL_ATR_MULT,
    V7_VOL_LOOKBACK,
)
from models import CandleSeries
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade, _precompute
from backtest.scanner_v3 import _simple_regime
from backtest.scanner_v6 import _daily_trend, _donchian_confirm, _ma_bear, _ma_bull, _pullback
from indicators import swing_structure_dir


@dataclass(frozen=True)
class V7Family:
    key: str
    name: str
    notes: str
    addresses: str


FAMILIES: tuple[V7Family, ...] = (
    V7Family(
        "A",
        "V7_A_VOL_TREND_CONFIRM",
        "Vol expansion + ADX trend strength + MA + Donchian next-bar confirm.",
        "Stocks C: raw expansion breakouts failed held-out/folds; add confirm+ADX.",
    ),
    V7Family(
        "B",
        "V7_B_VOL_PULLBACK",
        "Recent vol expansion + MA trend + pullback-to-SMA20 (no chase).",
        "Stocks C small-n chase; pullback after expansion aims for more stable entries.",
    ),
    V7Family(
        "C",
        "V7_C_BREAKOUT_RETEST",
        "Donchian break then retest-hold + MA structure alignment.",
        "False-breakout noise; require retest acceptance of break level.",
    ),
    V7Family(
        "D",
        "V7_D_VOL_MTF",
        "Daily SMA20/50 filter + 4H vol expansion + MA-aligned Donchian confirm.",
        "Fold/period lottery; higher-TF regime must agree before 4H expansion entry.",
    ),
    V7Family(
        "E",
        "V7_E_REGIME_MOM_STRUCT",
        "ATR not compressed + structure+MA + RSI/momentum continuation.",
        "Over-rare vol filters; regime-aware momentum with structural confirmation.",
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


def _vol_expansion_at(feat: dict, i: int) -> bool:
    ap = feat["atr_pct"][i]
    med = feat["atr_pct_med50"][i]
    if any(np.isnan(x) for x in (ap, med)) or med <= 0:
        return False
    return ap >= V7_VOL_ATR_MULT * med


def _recent_vol_expansion(feat: dict, i: int, look: int = V7_VOL_LOOKBACK) -> bool:
    if i < look:
        return False
    return any(_vol_expansion_at(feat, j) for j in range(i - look + 1, i + 1))


def _adx_ok(feat: dict, i: int, direction: str) -> bool:
    adx_v = feat["adx"][i]
    pdi = feat["plus_di"][i]
    mdi = feat["minus_di"][i]
    if any(np.isnan(x) for x in (adx_v, pdi, mdi)) or adx_v < V7_ADX_MIN:
        return False
    if direction == "bullish":
        return pdi > mdi
    return mdi > pdi


def _atr_not_compressed(feat: dict, i: int, mult: float = 0.85) -> bool:
    ap = feat["atr_pct"][i]
    med = feat["atr_pct_med50"][i]
    if any(np.isnan(x) for x in (ap, med)) or med <= 0:
        return False
    return ap >= mult * med


def _breakout_retest(series: CandleSeries, i: int, direction: str, look: int = V7_LOOKBACK) -> bool:
    """Break within prior 3 bars, then retest near break level and hold."""
    if i < look + 4:
        return False
    if direction == "bullish":
        # break bar among i-3..i-1
        for b in range(i - 3, i):
            prior = float(np.max(series.high[b - look : b]))
            if float(series.close[b]) <= prior:
                continue
            # retest: low comes back near prior within ~0.35% or pierces slightly, close holds above
            lvl = prior
            touched = float(series.low[i]) <= lvl * 1.002 or float(series.low[i - 1]) <= lvl * 1.002
            holds = float(series.close[i]) >= lvl and float(series.close[i]) >= float(series.open[i])
            if touched and holds:
                return True
        return False
    for b in range(i - 3, i):
        prior = float(np.min(series.low[b - look : b]))
        if float(series.close[b]) >= prior:
            continue
        lvl = prior
        touched = float(series.high[i]) >= lvl * 0.998 or float(series.high[i - 1]) >= lvl * 0.998
        holds = float(series.close[i]) <= lvl and float(series.close[i]) <= float(series.open[i])
        if touched and holds:
            return True
    return False


def _momentum_struct(series: CandleSeries, feat: dict, i: int, direction: str) -> bool:
    rsi = feat["rsi"][i]
    if np.isnan(rsi) or i < 10:
        return False
    mom = float(series.close[i]) - float(series.close[i - 10])
    regime = _simple_regime(
        feat, i, require_structure=True, require_ma=True, require_adx=False
    )
    if regime != direction:
        return False
    if direction == "bullish":
        return rsi >= 52 and mom > 0 and float(series.close[i]) >= float(series.open[i])
    return rsi <= 48 and mom < 0 and float(series.close[i]) <= float(series.open[i])


def _signal_at(
    family: V7Family,
    series: CandleSeries,
    feat: dict,
    i: int,
    daily: Optional[CandleSeries],
) -> Optional[str]:
    if family.key == "A":
        if not _vol_expansion_at(feat, i):
            return None
        if _ma_bull(feat, i) and _adx_ok(feat, i, "bullish") and _donchian_confirm(
            series, i, "bullish"
        ):
            return "bullish"
        if _ma_bear(feat, i) and _adx_ok(feat, i, "bearish") and _donchian_confirm(
            series, i, "bearish"
        ):
            return "bearish"
        return None

    if family.key == "B":
        if not _recent_vol_expansion(feat, i):
            return None
        if _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if family.key == "C":
        if _ma_bull(feat, i) and _breakout_retest(series, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _breakout_retest(series, i, "bearish"):
            return "bearish"
        return None

    if family.key == "D":
        dt = _daily_trend(daily, int(series.timestamps[i]))
        if dt is None or not _vol_expansion_at(feat, i):
            return None
        if (
            dt == "bullish"
            and _ma_bull(feat, i)
            and _donchian_confirm(series, i, "bullish")
        ):
            return "bullish"
        if (
            dt == "bearish"
            and _ma_bear(feat, i)
            and _donchian_confirm(series, i, "bearish")
        ):
            return "bearish"
        return None

    if family.key == "E":
        if not _atr_not_compressed(feat, i):
            return None
        if _momentum_struct(series, feat, i, "bullish"):
            return "bullish"
        if _momentum_struct(series, feat, i, "bearish"):
            return "bearish"
        return None

    return None


def backtest_family(
    series: CandleSeries,
    family: V7Family,
    *,
    daily: Optional[CandleSeries] = None,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V7_ATR_STOP_MULT,
    max_hold: int = V7_MAX_HOLD,
) -> list[V2Trade]:
    feat = _feat_cached(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, V7_LOOKBACK + 5)
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
        atr_x = float(atr0) * (atr_stop_mult / V7_ATR_STOP_MULT)
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
                confidence="V7",
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
                feature_flags={
                    "family_a": int(family.key == "A"),
                    "family_d": int(family.key == "D"),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_family_on_map(
    series_4h: dict[tuple[str, str], CandleSeries],
    family: V7Family,
    instruments: tuple[str, ...] | list[str],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V7_ATR_STOP_MULT,
    max_hold: int = V7_MAX_HOLD,
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
    family: V7Family,
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
