"""Frozen V11_S_FVG_SWEEP specification for V12 replication (DO NOT TUNE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backtest.scanner_v11 import V11Spec


FROZEN_V11_S_FVG_SWEEP: dict[str, Any] = {
    "origin": "V11_S_FVG_SWEEP (stocks near-miss; NOT promoted)",
    "primary_timeframe": "4H",
    "component": "FVG_SWEEP",
    "liquidity_sweep": (
        "Causal wick beyond last confirmed swing high/low then close back inside; "
        "bullish sweep: low < swing_low AND close > swing_low AND close >= open."
    ),
    "fvg_definition": (
        "Strict 3-candle FVG at bar i: bullish if low[i] > high[i-2]; "
        "bearish if high[i] < low[i-2]. Mitigation tracked forward only."
    ),
    "fvg_after_sweep": (
        "Entry bar i when (fresh OR partial FVG active at i) AND liquidity sweep occurred "
        "within bars i-0..i-3 (same causal rules as market_context_v11)."
    ),
    "entry_timing": "Signal bar close; optional entry slippage stress only in robustness.",
    "stop_exit": "V11 adaptive ATR exit (_adaptive_exit): initial 1.5×ATR stop, structure-break exit, ATR trail after 1R.",
    "max_hold_bars": 24,
    "cost_model": "ROUND_TRIP_COST by asset_class × cost_mult (2× stress gate)",
    "macro_consensus": "UNKNOWN — not used",
    "tuning_policy": "Frozen before expanded-history evaluation; no parameter changes on new data.",
}


@dataclass(frozen=True)
class V12Experiment:
    key: str
    name: str
    market_class: str
    spec: V11Spec
    phase: str
    frozen: bool = True


def build_v12_experiments() -> list[V12Experiment]:
    """Pre-registered V12 experiments only (no open search)."""
    return [
        V12Experiment(
            "E1",
            "V12_FROZEN_FVG_SWEEP",
            "stocks",
            V11Spec(
                "E1",
                "V12_FROZEN_FVG_SWEEP",
                "stocks",
                "FVG_SWEEP",
                "Frozen V11_S_FVG_SWEEP replication on extended data.",
                baseline="fvg",
            ),
            "primary_replication",
        ),
        V12Experiment(
            "E2",
            "V12_FROZEN_FVG_SWEEP_AVOID1H",
            "stocks",
            V11Spec(
                "E2",
                "V12_FROZEN_FVG_SWEEP_AVOID1H",
                "stocks",
                "FVG_SWEEP",
                "Frozen FVG_SWEEP + avoid 1h HIGH-event window.",
                baseline="fvg",
                event_mode="avoid_1h",
            ),
            "follow_up",
        ),
        V12Experiment(
            "E3",
            "V12_FROZEN_FVG_SWEEP_HTF",
            "stocks",
            V11Spec(
                "E3",
                "V12_FROZEN_FVG_SWEEP_HTF",
                "stocks",
                "FVG_SWEEP_HTF",
                "Frozen FVG_SWEEP + completed Daily structure alignment.",
                baseline="fvg",
            ),
            "follow_up",
        ),
        V12Experiment(
            "E4",
            "V12_FROZEN_COMM_EXIT_R2",
            "commodities",
            V11Spec(
                "E4",
                "V12_FROZEN_COMM_EXIT_R2",
                "commodities",
                "LIQ_HTF",
                "Frozen V11_C_EXIT_R2 commodity exit test (LIQ_HTF entry, 2R exit).",
                baseline="liq",
                is_exit_variant=True,
                exit_mode="r2",
                entry_component="LIQ_HTF",
            ),
            "commodity_exit_replication",
        ),
        V12Experiment(
            "E5",
            "V12_FROZEN_COMM_EXIT_PARTIAL",
            "commodities",
            V11Spec(
                "E5",
                "V12_FROZEN_COMM_EXIT_PARTIAL",
                "commodities",
                "LIQ_HTF",
                "Frozen V11_C_EXIT_PARTIAL commodity exit test.",
                baseline="liq",
                is_exit_variant=True,
                exit_mode="partial_1r",
                entry_component="LIQ_HTF",
            ),
            "commodity_exit_replication",
        ),
    ]


def cross_asset_replication_specs() -> list[V11Spec]:
    """Same frozen FVG_SWEEP evaluated on commodities and forex for diagnostics."""
    return [
        V11Spec("X1", "V12_DIAG_FVG_SWEEP_COMM", "commodities", "FVG_SWEEP", "Diagnostic cross-asset FVG_SWEEP.", baseline="fvg"),
        V11Spec("X2", "V12_DIAG_FVG_SWEEP_FX", "forex", "FVG_SWEEP", "Diagnostic cross-asset FVG_SWEEP.", baseline="fvg"),
    ]
