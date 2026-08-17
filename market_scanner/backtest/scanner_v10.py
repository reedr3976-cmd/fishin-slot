"""Scanner V10 — MARKET CONTEXT + PRICE STRUCTURE research (research only).

Tests causal structure, S/R, FVG, MTF alignment, regime, liquidity, and limited
combinations. Retains V9 macro/event layer as optional overlay. Does NOT modify
live ORIGINAL. No paper/live enablement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V10_ATR_STOP_MULT,
    V10_MAX_HOLD,
    V10_RR_TARGET,
)
from models import CandleSeries
from backtest.market_context import ContextArrays, build_context_arrays
from backtest.macro_features import MacroContext, fx_relative_ok, gold_filter_ok
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade, _precompute
from backtest.scanner_v6 import _daily_trend, _donchian_confirm, _ma_bear, _ma_bull, _pullback
from backtest.scanner_v8 import _feat_cached, _raw_donchian
from indicators import last_confirmed_swing


@dataclass(frozen=True)
class V10Spec:
    key: str
    name: str
    market_class: str
    component: str
    rationale: str
    is_control: bool = False
    is_combo: bool = False
    exit_mode: str = "atr"  # atr | struct | r_multiple
    event_window: Optional[dict] = None
    gold_mode: str = "none"
    fx_mode: str = "none"
    universe_tag: str = "default"


def _exit_for_mode(
    mode: str,
    series: CandleSeries,
    feat: dict,
    ctx: ContextArrays,
    entry_idx: int,
    direction: str,
    entry: float,
    atr0: float,
    max_hold: int,
) -> tuple[int, float, str, float]:
    n = len(series)
    stop_dist = V10_ATR_STOP_MULT * atr0

    if mode == "struct":
        sh, sl = last_confirmed_swing(series.high, series.low, entry_idx)
        if direction == "bullish" and sl is not None:
            stop_dist = max(stop_dist * 0.5, entry - sl)
        elif direction == "bearish" and sh is not None:
            stop_dist = max(stop_dist * 0.5, sh - entry)
        atr_x = float(atr0) * (stop_dist / (V10_ATR_STOP_MULT * atr0)) if atr0 > 0 else atr0
        return _adaptive_exit(series, feat, entry_idx, direction, entry, atr_x)

    if mode == "r_multiple":
        if direction == "bullish":
            stop = entry - stop_dist
            target = entry + V10_RR_TARGET * stop_dist
        else:
            stop = entry + stop_dist
            target = entry - V10_RR_TARGET * stop_dist
        last = min(entry_idx + max_hold, n - 1)
        for j in range(entry_idx + 1, last + 1):
            hi, lo = float(series.high[j]), float(series.low[j])
            if direction == "bullish":
                if lo <= stop:
                    return j, stop, "atr_stop", stop_dist
                if hi >= target:
                    return j, target, "r_target", stop_dist
            else:
                if hi >= stop:
                    return j, stop, "atr_stop", stop_dist
                if lo <= target:
                    return j, target, "r_target", stop_dist
        return last, float(series.close[last]), "max_hold", stop_dist

    atr_x = float(atr0)
    exit_idx, exit_px, reason, sd = _adaptive_exit(series, feat, entry_idx, direction, entry, atr_x)
    return exit_idx, exit_px, reason, sd


def _trending_regime(ctx: ContextArrays, i: int) -> bool:
    rc = int(ctx.regime_code[i])
    return rc in (1, 2, 3) and not ctx.is_ranging[i]


def _signal_at(
    spec: V10Spec,
    series: CandleSeries,
    feat: dict,
    ctx: ContextArrays,
    i: int,
    daily: Optional[CandleSeries],
) -> Optional[str]:
    c = spec.component

    if c == "CTRL_TREND":
        if spec.market_class == "forex":
            dt = _daily_trend(daily, int(series.timestamps[i]))
            if dt == "bullish" and _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
                return "bullish"
            if dt == "bearish" and _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
                return "bearish"
            return None
        if _ma_bull(feat, i) and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if _ma_bear(feat, i) and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if c == "CTRL_DON":
        return _raw_donchian(series, i)

    if c == "STRUCT_BOS":
        if ctx.bos_bull[i] and ctx.trend_dir[i] == 1:
            return "bullish"
        if ctx.bos_bear[i] and ctx.trend_dir[i] == -1:
            return "bearish"
        return None

    if c == "STRUCT_CHOCH":
        if ctx.choch_bull[i]:
            return "bullish"
        if ctx.choch_bear[i]:
            return "bearish"
        return None

    if c == "SR_RETEST":
        if ctx.sr_retest_bull[i] and ctx.trend_dir[i] >= 0:
            return "bullish"
        if ctx.sr_retest_bear[i] and ctx.trend_dir[i] <= 0:
            return "bearish"
        return None

    if c == "FVG_PB":
        if ctx.fvg_near_bull[i] and ctx.trend_dir[i] == 1:
            return "bullish"
        if ctx.fvg_near_bear[i] and ctx.trend_dir[i] == -1:
            return "bearish"
        return None

    if c == "MTF_DAILY":
        if ctx.daily_dir[i] == 1 and ctx.trend_dir[i] == 1 and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if ctx.daily_dir[i] == -1 and ctx.trend_dir[i] == -1 and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if c == "MTF_DAILY_WEEKLY":
        if ctx.daily_dir[i] == 1 and ctx.weekly_dir[i] == 1 and ctx.trend_dir[i] == 1:
            if _pullback(series, feat, i, "bullish") or ctx.bos_bull[i]:
                return "bullish"
        if ctx.daily_dir[i] == -1 and ctx.weekly_dir[i] == -1 and ctx.trend_dir[i] == -1:
            if _pullback(series, feat, i, "bearish") or ctx.bos_bear[i]:
                return "bearish"
        return None

    if c == "REGIME_TREND":
        if not _trending_regime(ctx, i):
            return None
        if int(ctx.trend_dir[i]) == 1 and _donchian_confirm(series, i, "bullish"):
            return "bullish"
        if int(ctx.trend_dir[i]) == -1 and _donchian_confirm(series, i, "bearish"):
            return "bearish"
        return None

    if c == "LIQ_SWEEP":
        if ctx.liq_sweep_bull[i] and ctx.trend_dir[i] >= 0:
            return "bullish"
        if ctx.liq_sweep_bear[i] and ctx.trend_dir[i] <= 0:
            return "bearish"
        return None

    if c == "LIQ_BREAK":
        if ctx.liq_break_bull[i] and _trending_regime(ctx, i):
            return "bullish"
        if ctx.liq_break_bear[i] and _trending_regime(ctx, i):
            return "bearish"
        return None

    if c == "COMBO_HTF_BOS":
        if ctx.daily_dir[i] == 1 and ctx.bos_bull[i] and _trending_regime(ctx, i):
            return "bullish"
        if ctx.daily_dir[i] == -1 and ctx.bos_bear[i] and _trending_regime(ctx, i):
            return "bearish"
        return None

    if c == "COMBO_SR_FVG":
        if ctx.sr_retest_bull[i] and ctx.fvg_near_bull[i] and ctx.trend_dir[i] == 1:
            return "bullish"
        if ctx.sr_retest_bear[i] and ctx.fvg_near_bear[i] and ctx.trend_dir[i] == -1:
            return "bearish"
        return None

    if c == "COMBO_FULL":
        if (
            ctx.daily_dir[i] == 1
            and ctx.trend_dir[i] == 1
            and _trending_regime(ctx, i)
            and (ctx.bos_bull[i] or ctx.sr_retest_bull[i] or ctx.fvg_near_bull[i])
        ):
            return "bullish"
        if (
            ctx.daily_dir[i] == -1
            and ctx.trend_dir[i] == -1
            and _trending_regime(ctx, i)
            and (ctx.bos_bear[i] or ctx.sr_retest_bear[i] or ctx.fvg_near_bear[i])
        ):
            return "bearish"
        return None

    return None


def backtest_spec(
    series: CandleSeries,
    spec: V10Spec,
    ctx_arr: ContextArrays,
    macro: Optional[MacroContext],
    *,
    daily: Optional[CandleSeries] = None,
    weekly: Optional[CandleSeries] = None,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V10_ATR_STOP_MULT,
    max_hold: int = V10_MAX_HOLD,
) -> list[V2Trade]:
    feat = _feat_cached(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, 30)
    n = len(series)
    i = max(warmup, start_idx or warmup)
    last_start = n - max_hold if end_idx_exclusive is None else min(n - max_hold, end_idx_exclusive)
    trades: list[V2Trade] = []
    win = spec.event_window or {}
    before = int(win.get("before_sec") or 0)
    after = int(win.get("after_sec") or 0)
    cal_day = bool(win.get("calendar_day"))
    skip_bar = bool(win.get("skip_event_bar"))

    while i < last_start:
        direction = _signal_at(spec, series, feat, ctx_arr, i, daily)
        if direction is None:
            i += 1
            continue
        ts = int(series.timestamps[i])
        if macro is not None and win:
            if macro.in_blackout(
                series.instrument, ts, before_sec=before, after_sec=after, calendar_day=cal_day, skip_event_bar=skip_bar
            ):
                i += 1
                continue
            if spec.gold_mode != "none" and series.instrument in ("XAUUSD", "XAGUSD", "COPPER"):
                if not gold_filter_ok(macro, ts, spec.gold_mode):
                    i += 1
                    continue
            if spec.fx_mode != "none" and series.asset_class == "forex":
                if not fx_relative_ok(macro, series.instrument, direction, ts, spec.fx_mode):
                    i += 1
                    continue

        atr0 = ctx_arr.atr[i] if not np.isnan(ctx_arr.atr[i]) else feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        raw = float(series.close[i])
        slip = entry_slip_atr * float(atr0)
        entry = raw + slip if direction == "bullish" else raw - slip
        exit_idx, exit_px, reason, stop_dist = _exit_for_mode(
            spec.exit_mode, series, feat, ctx_arr, i, direction, entry, float(atr0), max_hold
        )
        if exit_idx > i + max_hold:
            exit_idx = min(i + max_hold, n - 1)
            exit_px = float(series.close[exit_idx])
            reason = "max_hold"
            stop_dist = atr_stop_mult * float(atr0)
        snap = {
            "regime": int(ctx_arr.regime_code[i]),
            "daily_dir": int(ctx_arr.daily_dir[i]),
            "weekly_dir": int(ctx_arr.weekly_dir[i]),
        }
        trades.append(
            _make_trade(
                series=series,
                stage=spec.name,
                direction=direction,
                confidence="V10",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=spec.key,
                regime=snap.get("regime", "trending"),
                exit_reason=reason,
                feature_flags={
                    "component": spec.component,
                    "exit_mode": spec.exit_mode,
                    "combo": int(spec.is_combo),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_spec_on_map(
    series_4h: dict[tuple[str, str], CandleSeries],
    spec: V10Spec,
    instruments: tuple[str, ...] | list[str],
    ctx_map: dict[str, ContextArrays],
    macro: Optional[MacroContext],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    weekly_map: Optional[dict[str, CandleSeries]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V10_ATR_STOP_MULT,
    max_hold: int = V10_MAX_HOLD,
) -> list[V2Trade]:
    trades: list[V2Trade] = []
    for key in instruments:
        series = series_4h.get((key, "4h"))
        if series is None:
            continue
        ctx_arr = ctx_map.get(key)
        if ctx_arr is None:
            daily = (daily_map or {}).get(key)
            weekly = (weekly_map or {}).get(key)
            ctx_arr = build_context_arrays(series, daily=daily, weekly=weekly)
            ctx_map[key] = ctx_arr
        n = len(series)
        daily = (daily_map or {}).get(key)
        weekly = (weekly_map or {}).get(key)
        trades.extend(
            backtest_spec(
                series,
                spec,
                ctx_arr,
                macro,
                daily=daily,
                weekly=weekly,
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


def folds_for_spec(
    series_4h: dict,
    spec: V10Spec,
    instruments: tuple[str, ...] | list[str],
    ctx_map: dict[str, ContextArrays],
    macro: Optional[MacroContext],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    weekly_map: Optional[dict[str, CandleSeries]] = None,
    n_folds: int = 4,
) -> list[dict]:
    out = []
    for k in range(n_folds):
        start, end = k / n_folds, (k + 1) / n_folds
        trades = run_spec_on_map(
            series_4h,
            spec,
            instruments,
            ctx_map,
            macro,
            daily_map=daily_map,
            weekly_map=weekly_map,
            start_frac=start,
            end_frac=end,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out
