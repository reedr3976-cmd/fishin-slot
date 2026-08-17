"""Scanner V9 — MACRO / EVENT-LAYER RESEARCH (research only).

Overlays official-calendar event windows and market-observed policy context
on simple V8 trend/breakout entries. Does NOT modify live ORIGINAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import BACKTEST_WARMUP_BARS, SMA_SLOW, V9_ATR_STOP_MULT, V9_MAX_HOLD
from models import CandleSeries
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade
from backtest.scanner_v8 import V8Family, _feat_cached, _signal_at
from backtest.macro_features import MacroContext, fx_relative_ok, gold_filter_ok


@dataclass(frozen=True)
class V9Spec:
    key: str
    name: str
    market_class: str
    base: V8Family
    rationale: str
    event_window: dict
    gold_mode: str = "none"
    fx_mode: str = "none"
    is_control: bool = False
    universe_tag: str = "default"


def backtest_spec(
    series: CandleSeries,
    spec: V9Spec,
    ctx: Optional[MacroContext],
    *,
    daily: Optional[CandleSeries] = None,
    spy: Optional[CandleSeries] = None,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V9_ATR_STOP_MULT,
    max_hold: int = V9_MAX_HOLD,
) -> list[V2Trade]:
    feat = _feat_cached(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, 25)
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
        direction = _signal_at(spec.base, series, feat, i, daily, spy)
        if direction is None:
            i += 1
            continue
        ts = int(series.timestamps[i])
        if ctx is not None:
            blocked = ctx.in_blackout(
                series.instrument,
                ts,
                before_sec=before,
                after_sec=after,
                calendar_day=cal_day,
                skip_event_bar=skip_bar,
            )
            if blocked:
                i += 1
                continue
            if spec.market_class in ("commodities", "metals") or series.instrument in (
                "XAUUSD",
                "XAGUSD",
            ):
                if not gold_filter_ok(ctx, ts, spec.gold_mode):
                    i += 1
                    continue
            if spec.market_class == "forex" or series.instrument in (
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "AUDUSD",
                "USDCAD",
                "USDCHF",
            ):
                if not fx_relative_ok(ctx, series.instrument, direction, ts, spec.fx_mode):
                    i += 1
                    continue
        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        raw = float(series.close[i])
        slip = entry_slip_atr * float(atr0)
        entry = raw + slip if direction == "bullish" else raw - slip
        atr_x = float(atr0) * (atr_stop_mult / V9_ATR_STOP_MULT)
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
                stage=spec.name,
                direction=direction,
                confidence="V9",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=spec.key,
                regime="trending",
                exit_reason=reason,
                feature_flags={
                    "event_filtered": int(win.get("key", "none") != "none"),
                    "gold_filtered": int(spec.gold_mode != "none"),
                    "fx_filtered": int(spec.fx_mode != "none"),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_spec_on_map(
    series_4h: dict[tuple[str, str], CandleSeries],
    spec: V9Spec,
    instruments: tuple[str, ...] | list[str],
    ctx: Optional[MacroContext],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    spy: Optional[CandleSeries] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    atr_stop_mult: float = V9_ATR_STOP_MULT,
    max_hold: int = V9_MAX_HOLD,
) -> list[V2Trade]:
    trades: list[V2Trade] = []
    for key in instruments:
        series = series_4h.get((key, "4h"))
        if series is None:
            continue
        n = len(series)
        daily = (daily_map or {}).get(key)
        trades.extend(
            backtest_spec(
                series,
                spec,
                ctx,
                daily=daily,
                spy=spy,
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
    spec: V9Spec,
    instruments: tuple[str, ...] | list[str],
    ctx: Optional[MacroContext],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    spy: Optional[CandleSeries] = None,
    n_folds: int = 4,
) -> list[dict]:
    out = []
    for k in range(n_folds):
        start, end = k / n_folds, (k + 1) / n_folds
        trades = run_spec_on_map(
            series_4h,
            spec,
            instruments,
            ctx,
            daily_map=daily_map,
            spy=spy,
            start_frac=start,
            end_frac=end,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out
