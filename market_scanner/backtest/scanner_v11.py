"""Scanner V11 — near-miss refinement research (research only).

Deepens V10 LIQ_SWEEP, MTF_DAILY, and FVG_PULLBACK families with causal HTF
context, strict S/R zones, redesigned regime gating, separate exit research,
and V9 macro/event timing overlays. Does NOT modify live ORIGINAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import BACKTEST_WARMUP_BARS, SMA_SLOW, V11_ATR_STOP_MULT, V11_MAX_HOLD, V11_RR_TARGET
from models import CandleSeries
from backtest.macro_features import MacroContext, fx_relative_ok, gold_filter_ok
from backtest.market_context_v11 import V11Context, build_v11_context
from backtest.scanner_v2 import V2Trade, _adaptive_exit, _make_trade
from backtest.scanner_v6 import _pullback
from backtest.scanner_v8 import _feat_cached
from indicators import last_confirmed_swing


@dataclass(frozen=True)
class V11Spec:
    key: str
    name: str
    market_class: str
    component: str
    rationale: str
    baseline: str = "none"  # ablation baseline tag: liq | mtf | fvg
    is_control: bool = False
    is_combo: bool = False
    is_exit_variant: bool = False
    exit_mode: str = "atr"
    entry_component: str = ""  # for exit-only variants
    regime_gate: str = "none"  # none | trend | strong
    event_mode: str = "none"  # none | avoid_1h | avoid_pre | avoid_event | post_4h
    gold_mode: str = "none"
    fx_mode: str = "none"
    stock_macro: str = "none"
    universe_tag: str = "default"


def _regime_ok(ctx: V11Context, i: int, gate: str) -> bool:
    if gate == "none":
        return True
    rv = int(ctx.regime_v11[i])
    if gate == "trend":
        return rv in (1, 2, 3, 4, 5)
    if gate == "strong":
        return rv in (3, 4, 5)
    return True


def _in_post_event_window(macro: MacroContext, instrument: str, ts: int, *, bar_sec: int = 14400) -> bool:
    """First 4H bar after a HIGH event (causal)."""
    evs = macro.events_by_asset.get(instrument) or macro.events
    for ev in evs:
        if ev.importance != "HIGH" or not ev.ts_unix:
            continue
        if ev.time_precision in ("date_only", "UNKNOWN"):
            continue
        if ev.ts_unix <= ts < ev.ts_unix + bar_sec:
            return True
    return False


def _event_allows(macro: Optional[MacroContext], instrument: str, ts: int, mode: str) -> bool:
    if macro is None or mode in ("none", ""):
        return True
    if mode == "avoid_1h":
        return not macro.in_blackout(instrument, ts, before_sec=3600, after_sec=3600)
    if mode == "avoid_pre":
        return not macro.in_blackout(instrument, ts, before_sec=4 * 3600, after_sec=0)
    if mode == "avoid_event":
        return not macro.in_blackout(
            instrument, ts, before_sec=0, after_sec=0, skip_event_bar=True
        )
    if mode == "post_4h":
        return _in_post_event_window(macro, instrument, ts)
    return True


def _signal_component(ctx: V11Context, feat: dict, series: CandleSeries, i: int, comp: str) -> Optional[str]:
    v = ctx.v10
    dc, wc = int(ctx.daily_class[i]), int(ctx.weekly_class[i])

    if comp == "LIQ_BASE":
        if v.liq_sweep_bull[i]:
            return "bullish"
        if v.liq_sweep_bear[i]:
            return "bearish"
        return None

    if comp == "LIQ_HTF":
        if ctx.liq_sweep_htf_bull[i]:
            return "bullish"
        if ctx.liq_sweep_htf_bear[i]:
            return "bearish"
        return None

    if comp == "LIQ_HTF_DW":
        if ctx.liq_sweep_htf_bull[i] and dc == 1 and wc == 1:
            return "bullish"
        if ctx.liq_sweep_htf_bear[i] and dc == -1 and wc == -1:
            return "bearish"
        return None

    if comp == "LIQ_BOS":
        if ctx.bos_after_sweep_bull[i] and dc >= 0:
            return "bullish"
        if ctx.bos_after_sweep_bear[i] and dc <= 0:
            return "bearish"
        return None

    if comp == "LIQ_FVG":
        if ctx.liq_sweep_htf_bull[i] and (ctx.fvg_fresh_bull[i] or ctx.fvg_partial_bull[i]):
            return "bullish"
        if ctx.liq_sweep_htf_bear[i] and (ctx.fvg_fresh_bear[i] or ctx.fvg_partial_bear[i]):
            return "bearish"
        return None

    if comp == "LIQ_SR":
        if ctx.liq_at_sr_bull[i] and dc == 1:
            return "bullish"
        if ctx.liq_at_sr_bear[i] and dc == -1:
            return "bearish"
        return None

    if comp == "LIQ_COMBO":
        if ctx.liq_sweep_htf_bull[i] and ctx.bos_after_sweep_bull[i] and dc == 1:
            return "bullish"
        if ctx.liq_sweep_htf_bear[i] and ctx.bos_after_sweep_bear[i] and dc == -1:
            return "bearish"
        return None

    if comp == "MTF_D_PB":
        if dc == 1 and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if dc == -1 and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if comp == "MTF_DW_PB":
        if dc == 1 and wc == 1 and _pullback(series, feat, i, "bullish"):
            return "bullish"
        if dc == -1 and wc == -1 and _pullback(series, feat, i, "bearish"):
            return "bearish"
        return None

    if comp == "MTF_BOS":
        if dc == 1 and v.bos_bull[i]:
            return "bullish"
        if dc == -1 and v.bos_bear[i]:
            return "bearish"
        return None

    if comp == "MTF_PB_FVG":
        if dc == 1 and (ctx.fvg_fresh_bull[i] or ctx.fvg_partial_bull[i] or ctx.htf_fvg_bull[i]):
            return "bullish"
        if dc == -1 and (ctx.fvg_fresh_bear[i] or ctx.fvg_partial_bear[i] or ctx.htf_fvg_bear[i]):
            return "bearish"
        return None

    if comp == "MTF_PB_SR":
        if dc == 1 and (ctx.sr_zone_bull[i] or ctx.sr_flip_bull[i]):
            return "bullish"
        if dc == -1 and (ctx.sr_zone_bear[i] or ctx.sr_flip_bear[i]):
            return "bearish"
        return None

    if comp == "FVG_FRESH":
        if ctx.fvg_fresh_bull[i] and int(v.trend_dir[i]) == 1:
            return "bullish"
        if ctx.fvg_fresh_bear[i] and int(v.trend_dir[i]) == -1:
            return "bearish"
        return None

    if comp == "FVG_PARTIAL":
        if ctx.fvg_partial_bull[i] and int(v.trend_dir[i]) == 1:
            return "bullish"
        if ctx.fvg_partial_bear[i] and int(v.trend_dir[i]) == -1:
            return "bearish"
        return None

    if comp == "FVG_HTF":
        if ctx.fvg_htf_bull[i] or ctx.htf_fvg_bull[i]:
            return "bullish"
        if ctx.fvg_htf_bear[i] or ctx.htf_fvg_bear[i]:
            return "bearish"
        return None

    if comp == "FVG_SWEEP":
        if ctx.fvg_after_sweep_bull[i]:
            return "bullish"
        if ctx.fvg_after_sweep_bear[i]:
            return "bearish"
        return None

    if comp == "FVG_BOS":
        if ctx.fvg_after_bos_bull[i]:
            return "bullish"
        if ctx.fvg_after_bos_bear[i]:
            return "bearish"
        return None

    if comp == "SR_ZONE":
        if ctx.sr_zone_bull[i] and dc >= 0:
            return "bullish"
        if ctx.sr_zone_bear[i] and dc <= 0:
            return "bearish"
        return None

    if comp == "SR_FLIP":
        if ctx.sr_flip_bull[i] and dc == 1:
            return "bullish"
        if ctx.sr_flip_bear[i] and dc == -1:
            return "bearish"
        return None

    if comp == "SR_LIQ_FVG":
        if (ctx.sr_zone_bull[i] or ctx.sr_flip_bull[i]) and (
            ctx.liq_at_sr_bull[i] or ctx.fvg_fresh_bull[i] or ctx.fvg_partial_bull[i]
        ):
            return "bullish"
        if (ctx.sr_zone_bear[i] or ctx.sr_flip_bear[i]) and (
            ctx.liq_at_sr_bear[i] or ctx.fvg_fresh_bear[i] or ctx.fvg_partial_bear[i]
        ):
            return "bearish"
        return None

    return None


def _signal_at(spec: V11Spec, ctx: V11Context, feat: dict, series: CandleSeries, i: int) -> Optional[str]:
    comp = spec.entry_component or spec.component
    if not _regime_ok(ctx, i, spec.regime_gate):
        return None
    return _signal_component(ctx, feat, series, i, comp)


def _trail_structure_exit(
    series: CandleSeries,
    feat: dict,
    entry_idx: int,
    direction: str,
    entry: float,
    stop_dist: float,
    max_hold: int,
) -> tuple[int, float, str, float]:
    n = len(series)
    if direction == "bullish":
        stop = entry - stop_dist
        last = min(entry_idx + max_hold, n - 1)
        for j in range(entry_idx + 1, last + 1):
            lo, hi = float(series.low[j]), float(series.high[j])
            if lo <= stop:
                return j, stop, "atr_stop", stop_dist
            _, swing_lo = last_confirmed_swing(series.high, series.low, j)
            if swing_lo is not None:
                stop = max(stop, swing_lo)
            if int(feat["structure"][j]) == -1:
                return j, float(series.close[j]), "structure_break", stop_dist
        return last, float(series.close[last]), "max_hold", stop_dist
    stop = entry + stop_dist
    last = min(entry_idx + max_hold, n - 1)
    for j in range(entry_idx + 1, last + 1):
        hi, lo = float(series.high[j]), float(series.low[j])
        if hi >= stop:
            return j, stop, "atr_stop", stop_dist
        swing_hi, _ = last_confirmed_swing(series.high, series.low, j)
        if swing_hi is not None:
            stop = min(stop, swing_hi)
        if int(feat["structure"][j]) == 1:
            return j, float(series.close[j]), "structure_break", stop_dist
    return last, float(series.close[last]), "max_hold", stop_dist


def _partial_1r_exit(
    series: CandleSeries,
    entry_idx: int,
    direction: str,
    entry: float,
    stop_dist: float,
    max_hold: int,
) -> tuple[int, float, str, float]:
    n = len(series)
    target = entry + stop_dist if direction == "bullish" else entry - stop_dist
    stop = entry - stop_dist if direction == "bullish" else entry + stop_dist
    last = min(entry_idx + max_hold, n - 1)
    partial_taken = False
    for j in range(entry_idx + 1, last + 1):
        hi, lo = float(series.high[j]), float(series.low[j])
        if direction == "bullish":
            if lo <= stop:
                px = (entry + target) / 2 if partial_taken else stop
                return j, px, "atr_stop", stop_dist
            if not partial_taken and hi >= target:
                partial_taken = True
                stop = entry
        else:
            if hi >= stop:
                px = (entry + target) / 2 if partial_taken else stop
                return j, px, "atr_stop", stop_dist
            if not partial_taken and lo <= target:
                partial_taken = True
                stop = entry
    px = float(series.close[last])
    if partial_taken:
        px = (px + (target if direction == "bullish" else target)) / 2
    return last, px, "max_hold", stop_dist


def _opp_structure_exit(
    series: CandleSeries,
    ctx: V11Context,
    entry_idx: int,
    direction: str,
    entry: float,
    stop_dist: float,
    max_hold: int,
) -> tuple[int, float, str, float]:
    n = len(series)
    stop = entry - stop_dist if direction == "bullish" else entry + stop_dist
    last = min(entry_idx + max_hold, n - 1)
    v = ctx.v10
    for j in range(entry_idx + 1, last + 1):
        hi, lo = float(series.high[j]), float(series.low[j])
        if direction == "bullish":
            if lo <= stop:
                return j, stop, "atr_stop", stop_dist
            if v.bos_bear[j] or v.choch_bear[j]:
                return j, float(series.close[j]), "opp_bos_choch", stop_dist
        else:
            if hi >= stop:
                return j, stop, "atr_stop", stop_dist
            if v.bos_bull[j] or v.choch_bull[j]:
                return j, float(series.close[j]), "opp_bos_choch", stop_dist
    return last, float(series.close[last]), "max_hold", stop_dist


def _exit_for_mode(
    mode: str,
    series: CandleSeries,
    feat: dict,
    ctx: V11Context,
    entry_idx: int,
    direction: str,
    entry: float,
    atr0: float,
    max_hold: int,
) -> tuple[int, float, str, float]:
    stop_dist = V11_ATR_STOP_MULT * atr0
    if mode == "struct":
        sh, sl = last_confirmed_swing(series.high, series.low, entry_idx)
        if direction == "bullish" and sl is not None:
            stop_dist = max(stop_dist * 0.5, entry - sl)
        elif direction == "bearish" and sh is not None:
            stop_dist = max(stop_dist * 0.5, sh - entry)
        atr_x = float(atr0) * (stop_dist / (V11_ATR_STOP_MULT * atr0)) if atr0 > 0 else atr0
        return _adaptive_exit(series, feat, entry_idx, direction, entry, atr_x)
    if mode == "trail_struct":
        return _trail_structure_exit(series, feat, entry_idx, direction, entry, stop_dist, max_hold)
    if mode == "r2":
        last = min(entry_idx + max_hold, len(series) - 1)
        tgt = entry + V11_RR_TARGET * stop_dist if direction == "bullish" else entry - V11_RR_TARGET * stop_dist
        stop = entry - stop_dist if direction == "bullish" else entry + stop_dist
        for j in range(entry_idx + 1, last + 1):
            hi, lo = float(series.high[j]), float(series.low[j])
            if direction == "bullish":
                if lo <= stop:
                    return j, stop, "atr_stop", stop_dist
                if hi >= tgt:
                    return j, tgt, "r_target", stop_dist
            else:
                if hi >= stop:
                    return j, stop, "atr_stop", stop_dist
                if lo <= tgt:
                    return j, tgt, "r_target", stop_dist
        return last, float(series.close[last]), "max_hold", stop_dist
    if mode == "partial_1r":
        return _partial_1r_exit(series, entry_idx, direction, entry, stop_dist, max_hold)
    if mode == "opp_bos":
        return _opp_structure_exit(series, ctx, entry_idx, direction, entry, stop_dist, max_hold)
    return _adaptive_exit(series, feat, entry_idx, direction, entry, float(atr0))


def backtest_spec(
    series: CandleSeries,
    spec: V11Spec,
    ctx: V11Context,
    macro: Optional[MacroContext],
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    max_hold: int = V11_MAX_HOLD,
) -> list[V2Trade]:
    feat = _feat_cached(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, 30)
    n = len(series)
    i = max(warmup, start_idx or warmup)
    last_start = n - max_hold if end_idx_exclusive is None else min(n - max_hold, end_idx_exclusive)
    trades: list[V2Trade] = []

    while i < last_start:
        direction = _signal_at(spec, ctx, feat, series, i)
        if direction is None:
            i += 1
            continue
        ts = int(series.timestamps[i])
        if not _event_allows(macro, series.instrument, ts, spec.event_mode):
            i += 1
            continue
        if macro is not None:
            if spec.gold_mode != "none" and series.instrument in ("XAUUSD", "XAGUSD", "COPPER"):
                if not gold_filter_ok(macro, ts, spec.gold_mode):
                    i += 1
                    continue
            if spec.fx_mode != "none" and series.asset_class == "forex":
                if not fx_relative_ok(macro, series.instrument, direction, ts, spec.fx_mode):
                    i += 1
                    continue

        atr0 = ctx.v10.atr[i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue
        slip = entry_slip_atr * float(atr0)
        raw = float(series.close[i])
        entry = raw + slip if direction == "bullish" else raw - slip
        exit_idx, exit_px, reason, stop_dist = _exit_for_mode(
            spec.exit_mode, series, feat, ctx, i, direction, entry, float(atr0), max_hold
        )
        if exit_idx > i + max_hold:
            exit_idx = min(i + max_hold, n - 1)
            exit_px = float(series.close[exit_idx])
            reason = "max_hold"
        trades.append(
            _make_trade(
                series=series,
                stage=spec.name,
                direction=direction,
                confidence="V11",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger=spec.key,
                regime=int(ctx.regime_v11[i]),
                exit_reason=reason,
                feature_flags={
                    "component": spec.component,
                    "baseline": spec.baseline,
                    "exit_mode": spec.exit_mode,
                    "regime_v11": int(ctx.regime_v11[i]),
                },
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_spec_on_map(
    series_4h: dict,
    spec: V11Spec,
    instruments: tuple[str, ...] | list[str],
    ctx_map: dict[str, V11Context],
    macro: Optional[MacroContext],
    *,
    daily_map: Optional[dict[str, CandleSeries]] = None,
    weekly_map: Optional[dict[str, CandleSeries]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    max_hold: int = V11_MAX_HOLD,
) -> list[V2Trade]:
    trades: list[V2Trade] = []
    for key in instruments:
        series = series_4h.get((key, "4h"))
        if series is None:
            continue
        if key not in ctx_map:
            ctx_map[key] = build_v11_context(
                series, daily=(daily_map or {}).get(key), weekly=(weekly_map or {}).get(key)
            )
        n = len(series)
        trades.extend(
            backtest_spec(
                series,
                spec,
                ctx_map[key],
                macro,
                start_idx=int(n * start_frac),
                end_idx_exclusive=int(n * end_frac),
                cost_mult=cost_mult,
                entry_slip_atr=entry_slip_atr,
                max_hold=max_hold,
            )
        )
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades


def folds_for_spec(
    series_4h: dict,
    spec: V11Spec,
    instruments: tuple[str, ...] | list[str],
    ctx_map: dict[str, V11Context],
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
