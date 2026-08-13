"""Detect potential trading setups from indicator snapshots.

This module never places orders. It only emits alert dictionaries for humans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SMA_FAST,
    SMA_SLOW,
)
from indicators import compute_all
from models import CandleSeries


@dataclass
class SetupAlert:
    instrument: str
    name: str
    asset_class: str
    timeframe: str
    setup: str
    side: str  # "bullish" | "bearish" | "neutral"
    strength: str  # "low" | "medium" | "high"
    price: float
    message: str
    metrics: dict[str, Any]
    scanned_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def last_value(arr: np.ndarray) -> Optional[float]:
    if len(arr) == 0 or np.isnan(arr[-1]):
        return None
    return float(arr[-1])


# Back-compat alias used by scan module
_last = last_value


def _prev(arr: np.ndarray) -> Optional[float]:
    if len(arr) < 2 or np.isnan(arr[-2]):
        return None
    return float(arr[-2])


def _crossed_up(a_prev: Optional[float], a_now: Optional[float],
                b_prev: Optional[float], b_now: Optional[float]) -> bool:
    if None in (a_prev, a_now, b_prev, b_now):
        return False
    return a_prev <= b_prev and a_now > b_now


def _crossed_down(a_prev: Optional[float], a_now: Optional[float],
                  b_prev: Optional[float], b_now: Optional[float]) -> bool:
    if None in (a_prev, a_now, b_prev, b_now):
        return False
    return a_prev >= b_prev and a_now < b_now


def analyze_series(series: CandleSeries, display_name: str) -> list[SetupAlert]:
    """Return zero or more setup alerts for one instrument/timeframe."""
    if len(series) < max(SMA_SLOW, 35):
        return []

    ind = compute_all(series.close, series.high, series.low)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    price = series.last_close
    alerts: list[SetupAlert] = []

    rsi_now = _last(ind["rsi"])
    rsi_prev = _prev(ind["rsi"])
    sma_f = _last(ind["sma_fast"])
    sma_s = _last(ind["sma_slow"])
    sma_f_prev = _prev(ind["sma_fast"])
    sma_s_prev = _prev(ind["sma_slow"])
    macd_now = _last(ind["macd"])
    macd_sig = _last(ind["macd_signal"])
    macd_prev = _prev(ind["macd"])
    macd_sig_prev = _prev(ind["macd_signal"])
    bb_u = _last(ind["bb_upper"])
    bb_l = _last(ind["bb_lower"])
    bb_m = _last(ind["bb_mid"])
    atr_now = _last(ind["atr"])

    base_metrics = {
        "rsi": round(rsi_now, 2) if rsi_now is not None else None,
        "sma_fast": round(sma_f, 6) if sma_f is not None else None,
        "sma_slow": round(sma_s, 6) if sma_s is not None else None,
        "macd": round(macd_now, 6) if macd_now is not None else None,
        "macd_signal": round(macd_sig, 6) if macd_sig is not None else None,
        "bb_upper": round(bb_u, 6) if bb_u is not None else None,
        "bb_mid": round(bb_m, 6) if bb_m is not None else None,
        "bb_lower": round(bb_l, 6) if bb_l is not None else None,
        "atr": round(atr_now, 6) if atr_now is not None else None,
        "bars": len(series),
    }

    def emit(setup: str, side: str, strength: str, message: str) -> None:
        alerts.append(
            SetupAlert(
                instrument=series.instrument,
                name=display_name,
                asset_class=series.asset_class,
                timeframe=series.timeframe,
                setup=setup,
                side=side,
                strength=strength,
                price=round(price, 6),
                message=message,
                metrics=base_metrics,
                scanned_at=now,
            )
        )

    # RSI regimes / exits
    if rsi_now is not None:
        if rsi_now <= RSI_OVERSOLD:
            emit(
                "rsi_oversold",
                "bullish",
                "medium" if rsi_now > 20 else "high",
                f"RSI={rsi_now:.1f} ≤ {RSI_OVERSOLD} (oversold). Watch for bounce confirmation.",
            )
        elif rsi_now >= RSI_OVERBOUGHT:
            emit(
                "rsi_overbought",
                "bearish",
                "medium" if rsi_now < 80 else "high",
                f"RSI={rsi_now:.1f} ≥ {RSI_OVERBOUGHT} (overbought). Watch for pullback confirmation.",
            )
        elif rsi_prev is not None and rsi_prev <= RSI_OVERSOLD < rsi_now:
            emit(
                "rsi_exit_oversold",
                "bullish",
                "medium",
                f"RSI exited oversold ({rsi_prev:.1f} → {rsi_now:.1f}).",
            )
        elif rsi_prev is not None and rsi_prev >= RSI_OVERBOUGHT > rsi_now:
            emit(
                "rsi_exit_overbought",
                "bearish",
                "medium",
                f"RSI exited overbought ({rsi_prev:.1f} → {rsi_now:.1f}).",
            )

    # SMA crossover (20/50)
    if _crossed_up(sma_f_prev, sma_f, sma_s_prev, sma_s):
        emit(
            "sma_golden_cross",
            "bullish",
            "high",
            f"SMA{SMA_FAST} crossed above SMA{SMA_SLOW} (golden cross).",
        )
    elif _crossed_down(sma_f_prev, sma_f, sma_s_prev, sma_s):
        emit(
            "sma_death_cross",
            "bearish",
            "high",
            f"SMA{SMA_FAST} crossed below SMA{SMA_SLOW} (death cross).",
        )
    elif sma_f is not None and sma_s is not None:
        if sma_f > sma_s and price > sma_f:
            emit(
                "trend_aligned_bullish",
                "bullish",
                "low",
                f"Price above SMA{SMA_FAST} and SMA{SMA_FAST}>SMA{SMA_SLOW} (bullish structure).",
            )
        elif sma_f < sma_s and price < sma_f:
            emit(
                "trend_aligned_bearish",
                "bearish",
                "low",
                f"Price below SMA{SMA_FAST} and SMA{SMA_FAST}<SMA{SMA_SLOW} (bearish structure).",
            )

    # MACD signal cross
    if _crossed_up(macd_prev, macd_now, macd_sig_prev, macd_sig):
        emit(
            "macd_bullish_cross",
            "bullish",
            "medium",
            "MACD line crossed above signal line.",
        )
    elif _crossed_down(macd_prev, macd_now, macd_sig_prev, macd_sig):
        emit(
            "macd_bearish_cross",
            "bearish",
            "medium",
            "MACD line crossed below signal line.",
        )

    # Bollinger band touches / breaks
    if bb_l is not None and price <= bb_l:
        emit(
            "bb_lower_touch",
            "bullish",
            "medium",
            "Price at/below lower Bollinger Band — possible mean-reversion long watch.",
        )
    elif bb_u is not None and price >= bb_u:
        emit(
            "bb_upper_touch",
            "bearish",
            "medium",
            "Price at/above upper Bollinger Band — possible mean-reversion short watch.",
        )

    return alerts
