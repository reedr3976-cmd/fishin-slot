"""Data integrity checks for V12 extended research series."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import CandleSeries


def validate_series(series: CandleSeries) -> dict[str, Any]:
    """Run integrity checks; return findings (does not mutate data)."""
    n = len(series)
    issues: list[str] = []
    if n == 0:
        return {"instrument": series.instrument, "timeframe": series.timeframe, "bars": 0, "ok": False, "issues": ["empty"]}

    ts = series.timestamps.astype(np.int64)
    o, h, l, c = series.open, series.high, series.low, series.close

    # duplicates
    dup = int(n - len(np.unique(ts)))
    if dup:
        issues.append(f"duplicate_timestamps={dup}")

    # monotonic
    if np.any(ts[1:] <= ts[:-1]):
        issues.append("non_monotonic_timestamps")

    # OHLC sanity
    bad_hl = int(np.sum(h < l))
    bad_oh = int(np.sum(o > h) + np.sum(o < l))
    bad_ch = int(np.sum(c > h) + np.sum(c < l))
    if bad_hl:
        issues.append(f"high_lt_low={bad_hl}")
    if bad_oh:
        issues.append(f"open_outside_hl={bad_oh}")
    if bad_ch:
        issues.append(f"close_outside_hl={bad_ch}")

    non_pos = int(np.sum((c <= 0) | (h <= 0) | (l <= 0)))
    if non_pos:
        issues.append(f"non_positive_prices={non_pos}")

    # gap stats (4h expects ~14400 sec; 1h ~3600)
    step = 14400 if series.timeframe == "4h" else 3600 if series.timeframe == "1h" else 86400
    diffs = np.diff(ts)
    expected = step
    large_gaps = int(np.sum(diffs > expected * 3))
    median_diff = float(np.median(diffs)) if len(diffs) else 0.0

    span_days = (int(ts[-1]) - int(ts[0])) / 86400.0

    return {
        "instrument": series.instrument,
        "timeframe": series.timeframe,
        "bars": n,
        "span_days": round(span_days, 1),
        "first_ts": int(ts[0]),
        "last_ts": int(ts[-1]),
        "duplicate_timestamps": dup,
        "median_bar_spacing_sec": median_diff,
        "large_gaps": large_gaps,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def validate_panel(series_map: dict[tuple[str, str], CandleSeries]) -> dict[str, Any]:
    rows = [validate_series(s) for s in series_map.values()]
    return {
        "series_checked": len(rows),
        "all_ok": all(r["ok"] for r in rows),
        "by_series": rows,
        "instruments_with_issues": [r["instrument"] for r in rows if r["issues"]],
    }
