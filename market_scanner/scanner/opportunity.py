"""Ranked opportunity cards for the daily scanner (analysis only).

Each instrument/timeframe produces ONE ranked result with a confidence score.
Weak/conflicting conditions become "NO STRONG SETUP" instead of forced signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SCORE_HIGH,
    SCORE_LOW,
    SCORE_MEDIUM,
    SMA_FAST,
    SMA_SLOW,
)
from indicators import compute_all
from models import CandleSeries
from scanner.levels import format_level, nearest_levels
from scanner.setups import _crossed_down, _crossed_up, _prev, last_value


@dataclass
class Opportunity:
    instrument: str
    name: str
    asset_class: str
    timeframe: str
    price: float
    direction: str  # bullish | bearish | neutral
    confidence: str  # HIGH | MEDIUM | LOW | NO STRONG SETUP
    score: int  # 0–100
    reason: str
    rsi: Optional[float]
    sma20: Optional[float]
    sma50: Optional[float]
    sma_relation: str
    macd_condition: str
    atr: Optional[float]
    atr_pct: Optional[float]
    volatility_note: str
    support: Optional[float]
    resistance: Optional[float]
    support_2: Optional[float]
    resistance_2: Optional[float]
    factors: list[str] = field(default_factory=list)
    scanned_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_actionable(self) -> bool:
        return self.confidence in ("HIGH", "MEDIUM", "LOW") and self.direction in (
            "bullish",
            "bearish",
        )


def _sma_relation(price: float, sma20: Optional[float], sma50: Optional[float]) -> str:
    if sma20 is None or sma50 is None:
        return "Moving averages not ready yet"
    if sma20 > sma50 and price > sma20:
        return f"Bullish stack: price above SMA{SMA_FAST}, and SMA{SMA_FAST} above SMA{SMA_SLOW}"
    if sma20 < sma50 and price < sma20:
        return f"Bearish stack: price below SMA{SMA_FAST}, and SMA{SMA_FAST} below SMA{SMA_SLOW}"
    if sma20 > sma50:
        return f"SMA{SMA_FAST} is above SMA{SMA_SLOW}, but price is not cleanly above SMA{SMA_FAST}"
    if sma20 < sma50:
        return f"SMA{SMA_FAST} is below SMA{SMA_SLOW}, but price is not cleanly below SMA{SMA_FAST}"
    return "SMA20 and SMA50 are roughly flat / mixed"


def _macd_condition(
    macd_now: Optional[float],
    macd_sig: Optional[float],
    macd_prev: Optional[float],
    macd_sig_prev: Optional[float],
) -> tuple[str, Optional[str], int]:
    """Return (plain-English condition, optional side, score points)."""
    if None in (macd_now, macd_sig):
        return "MACD not ready yet", None, 0
    if _crossed_up(macd_prev, macd_now, macd_sig_prev, macd_sig):
        return "MACD just crossed above its signal line (bullish momentum shift)", "bullish", 25
    if _crossed_down(macd_prev, macd_now, macd_sig_prev, macd_sig):
        return "MACD just crossed below its signal line (bearish momentum shift)", "bearish", 25
    if macd_now > macd_sig and macd_now > 0:
        return "MACD is above signal and above zero (bullish momentum)", "bullish", 8
    if macd_now < macd_sig and macd_now < 0:
        return "MACD is below signal and below zero (bearish momentum)", "bearish", 8
    if macd_now > macd_sig:
        return "MACD is above its signal line (mild bullish bias)", "bullish", 5
    if macd_now < macd_sig:
        return "MACD is below its signal line (mild bearish bias)", "bearish", 5
    return "MACD is flat / mixed", None, 0


def _volatility_note(atr: Optional[float], price: float) -> tuple[Optional[float], str]:
    if atr is None or price <= 0:
        return None, "Volatility (ATR) not ready yet"
    pct = (atr / price) * 100.0
    if pct >= 3.0:
        note = f"High volatility — typical daily move ~{pct:.2f}% of price (ATR)"
    elif pct >= 1.0:
        note = f"Moderate volatility — typical daily move ~{pct:.2f}% of price (ATR)"
    else:
        note = f"Low volatility — typical daily move ~{pct:.2f}% of price (ATR)"
    return round(pct, 3), note


def _confidence_label(score: int) -> str:
    if score >= SCORE_HIGH:
        return "HIGH"
    if score >= SCORE_MEDIUM:
        return "MEDIUM"
    if score >= SCORE_LOW:
        return "LOW"
    return "NO STRONG SETUP"


def evaluate_opportunity(series: CandleSeries, display_name: str) -> Opportunity:
    """Build one ranked opportunity card for an instrument/timeframe."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    price = float(series.last_close)

    if len(series) < max(SMA_SLOW, 35):
        return Opportunity(
            instrument=series.instrument,
            name=display_name,
            asset_class=series.asset_class,
            timeframe=series.timeframe,
            price=round(price, 6),
            direction="neutral",
            confidence="NO STRONG SETUP",
            score=0,
            reason="Not enough price history yet to judge this market reliably.",
            rsi=None,
            sma20=None,
            sma50=None,
            sma_relation="Not enough bars for SMA20/SMA50",
            macd_condition="Not enough bars for MACD",
            atr=None,
            atr_pct=None,
            volatility_note="n/a",
            support=None,
            resistance=None,
            support_2=None,
            resistance_2=None,
            factors=[],
            scanned_at=now,
        )

    ind = compute_all(series.close, series.high, series.low)
    rsi_now = last_value(ind["rsi"])
    rsi_prev = _prev(ind["rsi"])
    sma20 = last_value(ind["sma_fast"])
    sma50 = last_value(ind["sma_slow"])
    sma20_prev = _prev(ind["sma_fast"])
    sma50_prev = _prev(ind["sma_slow"])
    macd_now = last_value(ind["macd"])
    macd_sig = last_value(ind["macd_signal"])
    macd_prev = _prev(ind["macd"])
    macd_sig_prev = _prev(ind["macd_signal"])
    bb_u = last_value(ind["bb_upper"])
    bb_l = last_value(ind["bb_lower"])
    atr_now = last_value(ind["atr"])

    bull_score = 0
    bear_score = 0
    bull_factors: list[str] = []
    bear_factors: list[str] = []

    # SMA structure / crosses
    if _crossed_up(sma20_prev, sma20, sma50_prev, sma50):
        bull_score += 35
        bull_factors.append("Golden cross: SMA20 just moved above SMA50")
    elif _crossed_down(sma20_prev, sma20, sma50_prev, sma50):
        bear_score += 35
        bear_factors.append("Death cross: SMA20 just moved below SMA50")
    elif sma20 is not None and sma50 is not None:
        if sma20 > sma50 and price > sma20:
            bull_score += 12
            bull_factors.append("Price and moving averages are stacked bullish")
        elif sma20 < sma50 and price < sma20:
            bear_score += 12
            bear_factors.append("Price and moving averages are stacked bearish")

    # RSI
    if rsi_now is not None:
        if rsi_now <= RSI_OVERSOLD:
            pts = 28 if rsi_now <= 20 else 20
            bull_score += pts
            bull_factors.append(f"RSI is oversold at {rsi_now:.1f} (bounce watch)")
        elif rsi_now >= RSI_OVERBOUGHT:
            pts = 28 if rsi_now >= 80 else 20
            bear_score += pts
            bear_factors.append(f"RSI is overbought at {rsi_now:.1f} (pullback watch)")
        elif rsi_prev is not None and rsi_prev <= RSI_OVERSOLD < rsi_now:
            bull_score += 18
            bull_factors.append(f"RSI left oversold ({rsi_prev:.1f} → {rsi_now:.1f})")
        elif rsi_prev is not None and rsi_prev >= RSI_OVERBOUGHT > rsi_now:
            bear_score += 18
            bear_factors.append(f"RSI left overbought ({rsi_prev:.1f} → {rsi_now:.1f})")
        elif rsi_now >= 55:
            bull_score += 4
            bull_factors.append(f"RSI is mildly strong at {rsi_now:.1f}")
        elif rsi_now <= 45:
            bear_score += 4
            bear_factors.append(f"RSI is mildly weak at {rsi_now:.1f}")

    # MACD
    macd_text, macd_side, macd_pts = _macd_condition(
        macd_now, macd_sig, macd_prev, macd_sig_prev
    )
    if macd_side == "bullish":
        bull_score += macd_pts
        bull_factors.append(macd_text)
    elif macd_side == "bearish":
        bear_score += macd_pts
        bear_factors.append(macd_text)

    # Bollinger
    if bb_l is not None and price <= bb_l:
        bull_score += 15
        bull_factors.append("Price is at/below the lower Bollinger Band")
    elif bb_u is not None and price >= bb_u:
        bear_score += 15
        bear_factors.append("Price is at/above the upper Bollinger Band")

    # Decide direction from net confluence
    net = bull_score - bear_score
    if bull_score >= bear_score and bull_score > 0 and net >= 8:
        direction = "bullish"
        raw_score = bull_score
        factors = bull_factors
        # Penalize if opposing evidence is strong
        if bear_score >= 15:
            raw_score = max(0, raw_score - min(bear_score, 20))
            factors = factors + [f"Note: some opposing pressure ({bear_score} bear points)"]
    elif bear_score > bull_score and bear_score > 0 and -net >= 8:
        direction = "bearish"
        raw_score = bear_score
        factors = bear_factors
        if bull_score >= 15:
            raw_score = max(0, raw_score - min(bull_score, 20))
            factors = factors + [f"Note: some opposing pressure ({bull_score} bull points)"]
    else:
        direction = "neutral"
        raw_score = max(bull_score, bear_score)
        factors = bull_factors + bear_factors
        # Mixed / unclear — force below actionable unless extremely one-sided (handled above)
        raw_score = min(raw_score, SCORE_LOW - 1)

    # Confluence bonus when multiple independent factors agree
    if direction in ("bullish", "bearish") and len([f for f in factors if not f.startswith("Note:")]) >= 3:
        raw_score += 10
        factors.append("Several indicators agree (confluence bonus)")

    score = int(max(0, min(100, raw_score)))
    confidence = _confidence_label(score)

    # Neutral or weak => NO STRONG SETUP wording
    if direction == "neutral" or confidence == "NO STRONG SETUP":
        confidence = "NO STRONG SETUP"
        direction = "neutral" if direction == "neutral" else direction
        if not factors:
            reason = (
                "No clear bullish or bearish setup right now. "
                "Indicators are mixed or quiet — better to wait."
            )
        else:
            reason = (
                "NO STRONG SETUP: conditions are not clear enough for a ranked opportunity. "
                + " | ".join(factors[:3])
            )
        # Keep direction neutral for non-actionable cards
        if score < SCORE_LOW:
            direction = "neutral"
    else:
        lead = factors[0] if factors else "Technical conditions lined up"
        reason = f"{lead}."
        if len(factors) > 1:
            reason += " Also: " + "; ".join(factors[1:3]) + "."

    atr_pct, vol_note = _volatility_note(atr_now, price)
    levels = nearest_levels(
        series.close, series.high, series.low, sma20=sma20, sma50=sma50
    )

    return Opportunity(
        instrument=series.instrument,
        name=display_name,
        asset_class=series.asset_class,
        timeframe=series.timeframe,
        price=round(price, 6),
        direction=direction,
        confidence=confidence,
        score=score,
        reason=reason,
        rsi=round(rsi_now, 2) if rsi_now is not None else None,
        sma20=round(sma20, 6) if sma20 is not None else None,
        sma50=round(sma50, 6) if sma50 is not None else None,
        sma_relation=_sma_relation(price, sma20, sma50),
        macd_condition=macd_text,
        atr=round(atr_now, 6) if atr_now is not None else None,
        atr_pct=atr_pct,
        volatility_note=vol_note,
        support=levels["support"],
        resistance=levels["resistance"],
        support_2=levels["support_2"],
        resistance_2=levels["resistance_2"],
        factors=factors,
        scanned_at=now,
    )


def rank_opportunities(opps: list[Opportunity]) -> list[Opportunity]:
    """Strongest first: HIGH > MEDIUM > LOW > NO STRONG SETUP, then by score."""
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NO STRONG SETUP": 3}
    return sorted(
        opps,
        key=lambda o: (
            order.get(o.confidence, 9),
            -o.score,
            o.asset_class,
            o.instrument,
            o.timeframe,
        ),
    )


def format_price(value: float) -> str:
    return format_level(value)
