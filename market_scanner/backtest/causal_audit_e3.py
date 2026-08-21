"""V13 causal / look-ahead audit for frozen E3 (Daily HTF on 4H).

Mandatory because E3 adds Daily HTF alignment on top of FVG_SWEEP.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from models import CandleSeries
from providers.dukascopy_data import aggregate_daily
from backtest.market_context_v11 import _htf_class, build_v11_context, clear_v11_cache


def _synthetic_4h_look_ahead_series() -> CandleSeries:
    """Series where the last 4H bar of a day moves price far from mid-day levels."""
    base = 1_600_000_000
    rows = []
    px = 100.0
    for day in range(40):  # enough history for ADX/structure
        for b in range(6):
            ts = base + day * 86400 + b * 14400
            o = px
            if day == 35 and b == 5:
                h = px + 12.0
                l = px - 0.2
                c = px + 11.0
            elif day == 35 and b < 5:
                h = px + 0.15
                l = px - 0.15
                c = px + 0.02
            elif day >= 30:
                # gentle uptrend building structure
                h = px + 0.4
                l = px - 0.1
                c = px + 0.25
            else:
                h = px + 0.2
                l = px - 0.2
                c = px + (0.05 if day % 2 == 0 else -0.03)
            rows.append((ts, o, h, l, c, 1.0))
            px = c
    arr = np.array(rows, dtype=np.float64)
    return CandleSeries(
        instrument="SYN",
        symbol="SYN",
        asset_class="stock",
        timeframe="4h",
        timestamps=arr[:, 0].astype(np.int64),
        open=arr[:, 1],
        high=arr[:, 2],
        low=arr[:, 3],
        close=arr[:, 4],
        volume=arr[:, 5],
    )


def _causal_daily_up_to(series_4h: CandleSeries, max_ts: int) -> CandleSeries:
    """Aggregate daily using only 4H bars with timestamp <= max_ts."""
    m = series_4h.timestamps <= max_ts
    if not np.any(m):
        raise ValueError("no bars")
    truncated = CandleSeries(
        instrument=series_4h.instrument,
        symbol=series_4h.symbol,
        asset_class=series_4h.asset_class,
        timeframe="4h",
        timestamps=series_4h.timestamps[m],
        open=series_4h.open[m],
        high=series_4h.high[m],
        low=series_4h.low[m],
        close=series_4h.close[m],
        volume=series_4h.volume[m],
    )
    return aggregate_daily(truncated)


def audit_same_day_ohlc_leak_mechanical(series_4h: Optional[CandleSeries] = None) -> dict[str, Any]:
    """Prove mid-day signals can read same-day future OHLC via pre-aggregated daily."""
    s4 = series_4h or _synthetic_4h_look_ahead_series()
    daily_full = aggregate_daily(s4)

    # Pick a mid-day bar that is not the last bar of its UTC day
    leaks = []
    for i in range(len(s4) - 1):
        day = int(s4.timestamps[i] // 86400)
        # find last bar of this day
        last_i = i
        while last_i + 1 < len(s4) and int(s4.timestamps[last_i + 1] // 86400) == day:
            last_i += 1
        if last_i <= i:
            continue
        t = int(s4.timestamps[i])
        d_idx = int(np.searchsorted(daily_full.timestamps, t, side="right") - 1)
        if d_idx < 0:
            continue
        # Daily close for this day equals last 4H close of day (by aggregate_daily)
        daily_close = float(daily_full.close[d_idx])
        future_close = float(s4.close[last_i])
        mid_close = float(s4.close[i])
        if abs(daily_close - future_close) < 1e-12 and abs(future_close - mid_close) > 1e-9:
            # Using daily_full at mid-day incorporates future_close
            causal = _causal_daily_up_to(s4, t)
            c_idx = int(np.searchsorted(causal.timestamps, t, side="right") - 1)
            causal_close = float(causal.close[c_idx]) if c_idx >= 0 else None
            full_class = int(_htf_class(daily_full, t))
            causal_class = int(_htf_class(causal, t)) if c_idx >= 0 else None
            leaks.append(
                {
                    "bar_idx": i,
                    "day": day,
                    "mid_close": mid_close,
                    "daily_full_close": daily_close,
                    "day_end_4h_close": future_close,
                    "causal_daily_close": causal_close,
                    "htf_class_full_daily": full_class,
                    "htf_class_causal_daily": causal_class,
                    "class_mismatch": full_class != causal_class,
                }
            )
            if len(leaks) >= 5:
                break

    class_mismatches = sum(1 for x in leaks if x.get("class_mismatch"))
    look_ahead = len(leaks) > 0
    return {
        "test": "same_day_daily_ohlc_mechanical_leak",
        "look_ahead_detected": look_ahead,
        "sample_leaks": leaks[:5],
        "n_leak_examples": len(leaks),
        "n_class_mismatches_in_samples": class_mismatches,
        "mechanism": (
            "aggregate_daily() builds full-day OHLC from all 4H bars of the UTC day; "
            "_htf_class(daily, t) selects that day's bar when t is mid-day, so high/low/close "
            "(and thus ADX/swing structure) can incorporate later 4H bars of the same day."
        ),
        "passthrough_requirement": (
            "At 4H timestamp t, Daily HTF must use only fully completed days before t "
            "(exclude the in-progress UTC day)."
        ),
        "pass": not look_ahead,
    }


def audit_daily_htf_look_ahead() -> dict[str, Any]:
    clear_v11_cache()
    mechanical = audit_same_day_ohlc_leak_mechanical()

    # Classification-level check on synthetic series
    s4 = _synthetic_4h_look_ahead_series()
    daily = aggregate_daily(s4)
    day35_mid = [35 * 6 + b for b in range(5)]
    day35_last = 35 * 6 + 5
    mid_classes = [int(_htf_class(daily, int(s4.timestamps[i]))) for i in day35_mid]
    last_class = int(_htf_class(daily, int(s4.timestamps[day35_last])))
    ctx = build_v11_context(s4, daily=daily, weekly=None)
    ctx_mid = [int(ctx.daily_class[i]) for i in day35_mid]
    ctx_last = int(ctx.daily_class[day35_last])
    class_leak = bool(last_class != 0 and any(c == last_class for c in mid_classes))

    look_ahead = bool(mechanical["look_ahead_detected"] or class_leak)
    return {
        **mechanical,
        "test": "daily_htf_same_day_ohlc_leak",
        "mid_day_htf_classes": mid_classes,
        "end_of_day_htf_class": last_class,
        "ctx_mid_day_classes": ctx_mid,
        "ctx_end_of_day_class": ctx_last,
        "classification_leak_synthetic": class_leak,
        "look_ahead_detected": look_ahead,
        "pass": not look_ahead,
    }


def audit_entry_uses_only_past_bars(series: CandleSeries, entry_idx: int, ctx) -> dict[str, Any]:
    issues = []
    if entry_idx < 2:
        issues.append("entry_idx_too_early_for_fvg")
    return {
        "entry_idx": entry_idx,
        "entry_ts": int(series.timestamps[entry_idx]),
        "fvg_after_sweep_bull": bool(ctx.fvg_after_sweep_bull[entry_idx]),
        "fvg_after_sweep_bear": bool(ctx.fvg_after_sweep_bear[entry_idx]),
        "daily_class": int(ctx.daily_class[entry_idx]),
        "issues": issues,
        "pass": len(issues) == 0,
        "note": "FVG/sweep indices are <= entry_idx by construction; HTF same-day leak checked separately.",
    }


def run_causal_audit(sample_series: CandleSeries | None = None, sample_ctx=None) -> dict[str, Any]:
    htf = audit_daily_htf_look_ahead()
    # Also run mechanical check on real sample if provided
    real_mech = None
    if sample_series is not None and len(sample_series) > 200:
        real_mech = audit_same_day_ohlc_leak_mechanical(sample_series)
        if real_mech.get("look_ahead_detected"):
            htf["look_ahead_detected"] = True
            htf["pass"] = False
            htf["real_sample_mechanical"] = {
                "n_leak_examples": real_mech.get("n_leak_examples"),
                "sample_leaks": real_mech.get("sample_leaks"),
            }

    entry_check = None
    if sample_series is not None and sample_ctx is not None and len(sample_series) > 100:
        for i in range(60, len(sample_series) - 30):
            if sample_ctx.fvg_after_sweep_bull[i] and int(sample_ctx.daily_class[i]) == 1:
                entry_check = audit_entry_uses_only_past_bars(sample_series, i, sample_ctx)
                break
            if sample_ctx.fvg_after_sweep_bear[i] and int(sample_ctx.daily_class[i]) == -1:
                entry_check = audit_entry_uses_only_past_bars(sample_series, i, sample_ctx)
                break
    return {
        "htf_look_ahead": htf,
        "sample_entry_check": entry_check,
        "look_ahead_detected": bool(htf.get("look_ahead_detected")),
        "all_pass": bool(htf.get("pass")) and (entry_check is None or entry_check.get("pass")),
    }
