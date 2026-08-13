"""Simple support / resistance from recent swing pivots (alerts only)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import SR_LOOKBACK, SR_PIVOT_LEFT, SR_PIVOT_RIGHT


def _pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> tuple[list[float], list[float]]:
    """Return swing-high and swing-low prices using a local pivot window."""
    highs: list[float] = []
    lows: list[float] = []
    n = len(high)
    for i in range(left, n - right):
        window_h = high[i - left : i + right + 1]
        window_l = low[i - left : i + right + 1]
        if high[i] >= np.max(window_h):
            highs.append(float(high[i]))
        if low[i] <= np.min(window_l):
            lows.append(float(low[i]))
    return highs, lows


def nearest_levels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    lookback: int = SR_LOOKBACK,
    sma20: Optional[float] = None,
    sma50: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Estimate nearby support / resistance for beginner-friendly display."""
    if len(close) < 10:
        return {
            "support": None,
            "resistance": None,
            "support_2": None,
            "resistance_2": None,
        }

    start = max(0, len(close) - lookback)
    h = high[start:]
    l = low[start:]
    price = float(close[-1])

    swing_h, swing_l = _pivots(h, l, SR_PIVOT_LEFT, SR_PIVOT_RIGHT)

    # Also treat recent period high/low as levels
    period_high = float(np.max(h))
    period_low = float(np.min(l))
    swing_h.append(period_high)
    swing_l.append(period_low)

    # Dynamic MA levels often act as soft S/R
    for level in (sma20, sma50):
        if level is None:
            continue
        if level < price:
            swing_l.append(float(level))
        elif level > price:
            swing_h.append(float(level))

    supports = sorted({round(x, 8) for x in swing_l if x < price}, reverse=True)
    resistances = sorted({round(x, 8) for x in swing_h if x > price})

    return {
        "support": supports[0] if supports else None,
        "support_2": supports[1] if len(supports) > 1 else None,
        "resistance": resistances[0] if resistances else None,
        "resistance_2": resistances[1] if len(resistances) > 1 else None,
    }


def format_level(value: Optional[float], digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.{digits}f}"
