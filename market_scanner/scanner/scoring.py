"""Pluggable scoring rules for opportunity confidence (alerts only).

ORIGINAL rules match the merged live scanner. REVISED candidates must be
proposed from TRAINING data only, then frozen for out-of-sample testing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ScoringRules:
    """Point weights and confidence thresholds.

    Thresholds: score >= high -> HIGH, >= medium -> MEDIUM, >= low -> LOW.
    """

    name: str = "original"
    # Confidence cutoffs
    score_high: int = 60
    score_medium: int = 40
    score_low: int = 25
    # Direction / confluence
    min_net_for_direction: int = 8
    opposing_penalty_trigger: int = 15
    opposing_penalty_cap: int = 20
    confluence_min_factors: int = 3
    confluence_bonus: int = 10
    # Feature point weights (bullish or bearish side)
    sma_cross: int = 35
    sma_stack: int = 12
    rsi_extreme_strong: int = 28  # RSI <=20 or >=80
    rsi_extreme: int = 20
    rsi_exit: int = 18
    rsi_mild: int = 4
    macd_cross: int = 25
    macd_strong: int = 8  # above/below signal and zero
    macd_mild: int = 5
    bb_touch: int = 15
    # Optional: require N agreeing factors for HIGH (0 = disabled)
    high_min_factors: int = 0
    notes: str = "Baseline merged scanner weights/thresholds."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def confidence_label(self, score: int, factor_count: int = 0) -> str:
        if score >= self.score_high and (
            self.high_min_factors <= 0 or factor_count >= self.high_min_factors
        ):
            return "HIGH"
        if score >= self.score_medium:
            # Met raw high score but failed factor gate → still MEDIUM
            if score >= self.score_high:
                return "MEDIUM"
            return "MEDIUM"
        if score >= self.score_low:
            return "LOW"
        return "NO STRONG SETUP"


# Frozen copy of the currently merged live scanner rules.
ORIGINAL_RULES = ScoringRules(name="original")


def explain_why_high_is_rare(rules: ScoringRules = ORIGINAL_RULES) -> str:
    """Beginner-friendly explanation of HIGH scarcity under original rules."""
    return f"""
WHY HIGH SIGNALS ARE RARE (under "{rules.name}" rules)
=====================================================
Confidence uses a 0–100 point score:
  HIGH   ≥ {rules.score_high}
  MEDIUM ≥ {rules.score_medium}
  LOW    ≥ {rules.score_low}
  else   NO STRONG SETUP

Typical point sources (one side only):
  • SMA20/50 golden or death cross ........ +{rules.sma_cross}  (rare event)
  • Bullish/bearish MA stack .............. +{rules.sma_stack}  (common)
  • RSI extreme (≤30 / ≥70) ............... +{rules.rsi_extreme} to +{rules.rsi_extreme_strong}
  • RSI leaving extreme ................... +{rules.rsi_exit}
  • Mild RSI tilt ......................... +{rules.rsi_mild}
  • MACD signal-line cross ................ +{rules.macd_cross}  (uncommon)
  • MACD strong / mild bias ............... +{rules.macd_strong} / +{rules.macd_mild}
  • Bollinger band touch .................. +{rules.bb_touch}
  • ≥{rules.confluence_min_factors} agreeing factors bonus ......... +{rules.confluence_bonus}
  • Opposing evidence can subtract up to .. -{rules.opposing_penalty_cap}

What this means in practice:
  A “normal” mild trend day is often only ~{rules.sma_stack + rules.rsi_mild + rules.macd_mild}
  to ~{rules.sma_stack + rules.rsi_mild + rules.macd_strong + rules.confluence_bonus} points
  → LOW (or NO STRONG SETUP), not HIGH.

  Reaching {rules.score_high}+ usually needs uncommon combinations, for example:
  • SMA cross ({rules.sma_cross}) + MACD cross ({rules.macd_cross}) = {rules.sma_cross + rules.macd_cross}
  • SMA cross + RSI extreme + confluence bonus
  • RSI extreme + Bollinger touch + MACD cross + confluence

  So HIGH is scarce by design: the threshold sits near rare multi-factor events,
  while everyday trend/momentum states land in LOW/MEDIUM. That is why the
  baseline backtest saw only a handful of HIGH trades versus hundreds of LOW/MEDIUM.
""".strip()
