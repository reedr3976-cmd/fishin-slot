"""Frozen V12 E3 (V12_FROZEN_FVG_SWEEP_HTF) specification for V13 confirmation.

DO NOT TUNE. DO NOT OPTIMISE against confirmation data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backtest.scanner_v11 import V11Spec


# Exact V12 E3 wiring — component FVG_SWEEP_HTF, stocks, adaptive ATR exit.
FROZEN_E3_SPEC = V11Spec(
    "E3",
    "V13_FROZEN_E3_FVG_SWEEP_HTF",
    "stocks",
    "FVG_SWEEP_HTF",
    "Frozen V12 E3 confirmation candidate. No parameter changes.",
    baseline="fvg",
)


FROZEN_E3_DOCUMENT: dict[str, Any] = {
    "candidate_id": "V12_FROZEN_FVG_SWEEP_HTF",
    "v13_name": "V13_FROZEN_E3_FVG_SWEEP_HTF",
    "origin": "V12 follow-up E3 (pre-registered); NOT V11_S_FVG_SWEEP",
    "primary_timeframe": "4H",
    "component": "FVG_SWEEP_HTF",
    "market_class": "stocks",
    "liquidity_sweep": {
        "definition": (
            "Causal wick beyond last confirmed swing high/low then close back inside. "
            "Bullish: low < swing_low AND close > swing_low AND close >= open. "
            "Bearish: high > swing_high AND close < swing_high AND close <= open."
        ),
        "pivot": 2,
        "source": "market_context.ContextArrays.liq_sweep_*",
    },
    "fvg": {
        "definition": (
            "Strict 3-candle FVG at bar i: bullish if low[i] > high[i-2]; "
            "bearish if high[i] < low[i-2]. Mitigation tracked forward only."
        ),
        "active": "fresh OR partial (partial touch of zone without full mitigation)",
        "max_age_bars_4h": 30,
        "proximity_filter_atr": 1.5,
    },
    "sweep_fvg_sequence": (
        "Entry bar i when (fresh OR partial FVG active at i) AND liquidity sweep "
        "occurred within bars i-0..i-3 (fvg_after_sweep_*)."
    ),
    "daily_htf_alignment": {
        "definition": (
            "Completed Daily structure class at 4H timestamp t via _htf_class: "
            "swing_structure_dir on daily bars with pivot=2 AND ADX(14) >= ADX_WEAK. "
            "Bullish E3 requires daily_class==1; bearish requires daily_class==-1."
        ),
        "adx_weak": 18.0,
        "adx_strong": 25.0,
        "lookup": "np.searchsorted(daily.timestamps, t, side='right') - 1",
        "note": "V13 causal audit verifies whether incomplete current-day OHLC leaks.",
    },
    "entry_timing": "Signal bar close (4H close). Adverse entry slippage only in stress tests.",
    "signal_timestamp": "series.timestamps[i] at the 4H signal bar",
    "stop": {
        "mode": "adaptive ATR (_adaptive_exit)",
        "initial_stop_atr_mult": 1.5,
        "structure_break_exit": True,
        "trail_after_1R": True,
        "trail_uses_confirmed_swing": True,
    },
    "exit_target": {
        "exit_mode": "atr (default adaptive — not r2/partial)",
        "max_hold_bars_4h": 24,
        "reasons": ["atr_stop", "structure_break", "max_hold"],
    },
    "transaction_costs": {
        "stock_round_trip": 0.0010,
        "applied": "once per trade (entry+exit proxy)",
        "stress_multiples": [1.0, 1.5, 2.0, 3.0],
    },
    "slippage": {
        "baseline_entry_slip_atr": 0.0,
        "stress_entry_slip_atr": 0.05,
    },
    "universe": {
        "dev": ["SPY", "QQQ", "AAPL", "MSFT", "XOM", "AMZN", "GOOGL", "META", "JPM"],
        "final_inst": ["NVDA", "JNJ", "WMT", "BA", "DIS"],
        "instrument_type": "Dukascopy US equity/ETF CFDs (bid), not cash equities",
    },
    "tuning_policy": "FROZEN for V13. Perturbations are diagnostic only; never replace candidate.",
    "code_refs": {
        "signal": "scanner_v11._signal_component FVG_SWEEP_HTF",
        "context": "market_context_v11.build_v11_context / fvg_after_sweep_* / daily_class",
        "exit": "scanner_v2._adaptive_exit via scanner_v11._exit_for_mode(mode='atr')",
    },
}


def frozen_e3_hash() -> str:
    """Stable content hash of the frozen specification document."""
    payload = json.dumps(FROZEN_E3_DOCUMENT, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FROZEN_E3_VERSION = f"E3-{frozen_e3_hash()[:12]}"
