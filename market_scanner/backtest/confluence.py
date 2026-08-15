"""Confluence filter study for MEDIUM/HIGH (analysis only).

Policies (ORIGINAL scoring thresholds unchanged; LOW never gated):
  A — Original baseline
  B — Require macd_strong for MEDIUM/HIGH
  C — Require directional S/R: near_support (bullish) / near_resistance (bearish)
  D — Require both B and C (reported even if sample is small)

Live scanner is not modified by this module.
"""

from __future__ import annotations

from typing import Any, Optional

from config import MIN_SIGNALS_FOR_CONCLUSION, VALIDATION_TRAIN_FRACTION
from backtest.engine import load_series_map, run_backtest_on_map
from backtest.metrics import TradeResult, summarize_trades
from scanner.scoring import ORIGINAL_RULES


def _flag(trade: TradeResult, name: str) -> bool:
    flags = trade.feature_flags or {}
    return int(flags.get(name, 0) or 0) == 1


def has_macd_strong(trade: TradeResult) -> bool:
    return _flag(trade, "macd_strong")


def has_directional_sr(trade: TradeResult) -> bool:
    """Bullish needs near_support; bearish needs near_resistance."""
    if trade.direction == "bullish":
        return _flag(trade, "near_support")
    if trade.direction == "bearish":
        return _flag(trade, "near_resistance")
    return False


def passes_policy(trade: TradeResult, policy: str) -> bool:
    """Return True if trade is kept under the named policy."""
    if trade.confidence == "LOW" or trade.confidence not in ("HIGH", "MEDIUM", "LOW"):
        return True
    if trade.confidence not in ("HIGH", "MEDIUM"):
        return True
    if policy == "A":
        return True
    if policy == "B":
        return has_macd_strong(trade)
    if policy == "C":
        return has_directional_sr(trade)
    if policy == "D":
        return has_macd_strong(trade) and has_directional_sr(trade)
    raise ValueError(f"Unknown confluence policy: {policy}")


def apply_policy(trades: list[TradeResult], policy: str) -> list[TradeResult]:
    return [t for t in trades if passes_policy(t, policy)]


def _bag_block(trades: list[TradeResult], name: str) -> dict[str, Any]:
    classes = sorted({t.asset_class for t in trades})
    instruments = sorted({t.instrument for t in trades})
    return {
        "overall": summarize_trades(f"{name}/ALL", trades),
        "by_confidence": {
            conf: summarize_trades(
                f"{name}/{conf}", [t for t in trades if t.confidence == conf]
            )
            for conf in ("HIGH", "MEDIUM", "LOW")
        },
        "medium_high": summarize_trades(
            f"{name}/MH",
            [t for t in trades if t.confidence in ("HIGH", "MEDIUM")],
        ),
        "by_asset_class": {
            cls: summarize_trades(
                f"{name}/{cls}", [t for t in trades if t.asset_class == cls]
            )
            for cls in classes
        },
        "by_asset_class_mh": {
            cls: summarize_trades(
                f"{name}/{cls}/MH",
                [
                    t
                    for t in trades
                    if t.asset_class == cls and t.confidence in ("HIGH", "MEDIUM")
                ],
            )
            for cls in classes
        },
        "by_instrument_mh": {
            inst: summarize_trades(
                f"{name}/{inst}/MH",
                [
                    t
                    for t in trades
                    if t.instrument == inst and t.confidence in ("HIGH", "MEDIUM")
                ],
            )
            for inst in instruments
        },
    }


def _removed_mh(original: list[TradeResult], policy: str) -> list[TradeResult]:
    return [
        t
        for t in original
        if t.confidence in ("HIGH", "MEDIUM") and not passes_policy(t, policy)
    ]


def _breadth(
    original: list[TradeResult],
    policy: str,
    kept: list[TradeResult],
) -> dict[str, Any]:
    removed = _removed_mh(original, policy)
    orig_mh = [t for t in original if t.confidence in ("HIGH", "MEDIUM")]
    kept_mh = [t for t in kept if t.confidence in ("HIGH", "MEDIUM")]

    rem_sum = float(sum(t.net_return for t in removed)) if removed else 0.0
    by_class: dict[str, Any] = {}
    for cls in sorted({t.asset_class for t in orig_mh} | {"forex", "commodity"}):
        subset = [t for t in removed if t.asset_class == cls]
        nets = [t.net_return for t in subset]
        by_class[cls] = {
            "n": len(subset),
            "sum_net": float(sum(nets)) if nets else 0.0,
            "avg_net": float(sum(nets) / len(nets)) if nets else None,
            "share_of_removed_sum": (
                float(sum(nets) / rem_sum) if nets and rem_sum != 0 else None
            ),
        }

    by_inst: list[dict[str, Any]] = []
    for inst in sorted({t.instrument for t in removed}):
        subset = [t for t in removed if t.instrument == inst]
        nets = [t.net_return for t in subset]
        by_inst.append(
            {
                "instrument": inst,
                "asset_class": subset[0].asset_class,
                "n": len(subset),
                "sum_net": float(sum(nets)) if nets else 0.0,
                "avg_net": float(sum(nets) / len(nets)) if nets else None,
            }
        )
    by_inst.sort(key=lambda r: abs(r["sum_net"]), reverse=True)

    # Leave-one-instrument-out: does MH avg lift vs A survive without top instrument?
    a_mh_avg = (
        float(sum(t.net_return for t in orig_mh) / len(orig_mh)) if orig_mh else None
    )
    loo: list[dict[str, Any]] = []
    for inst in sorted({t.instrument for t in kept_mh}):
        subset = [t for t in kept_mh if t.instrument != inst]
        if not subset or a_mh_avg is None:
            continue
        avg = float(sum(t.net_return for t in subset) / len(subset))
        loo.append(
            {
                "dropped_instrument": inst,
                "n": len(subset),
                "avg": avg,
                "delta_vs_A_mh_avg": avg - a_mh_avg,
            }
        )
    loo.sort(key=lambda r: r["delta_vs_A_mh_avg"], reverse=True)

    ranked = sorted(removed, key=lambda t: t.net_return)

    def share(k: int, worst: bool) -> Optional[float]:
        if not removed or rem_sum == 0:
            return None
        picks = ranked[:k] if worst else ranked[-k:]
        return float(sum(t.net_return for t in picks) / rem_sum)

    return {
        "removed_n": len(removed),
        "kept_mh_n": len(kept_mh),
        "removed_summary": summarize_trades(f"{policy}_removed_MH", removed),
        "kept_summary": summarize_trades(f"{policy}_kept_MH", kept_mh),
        "removed_sum_net": rem_sum,
        "removed_by_asset_class": by_class,
        "removed_by_instrument": by_inst,
        "leave_one_instrument_out_kept_mh": loo[:12],
        "concentration": {
            "worst_1_share_of_removed_sum": share(1, True),
            "worst_3_share_of_removed_sum": share(3, True),
            "best_1_share_of_removed_sum": share(1, False),
            "best_3_share_of_removed_sum": share(3, False),
        },
        "sample_permits_d": len(kept_mh) >= MIN_SIGNALS_FOR_CONCLUSION,
    }


def run_confluence_study(
    *,
    demo: bool = False,
    instruments=None,
    timeframes=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    tfs = list(timeframes) if timeframes is not None else ["1d", "1wk"]
    series_map, errors, bars = load_series_map(instruments, tfs, demo=demo)
    mode = "demo" if demo else "public_historical"

    train_run = run_backtest_on_map(
        series_map,
        ORIGINAL_RULES,
        start_frac=0.0,
        end_frac=train_fraction,
        mode=mode,
        errors=errors,
    )
    test_run = run_backtest_on_map(
        series_map,
        ORIGINAL_RULES,
        start_frac=train_fraction,
        end_frac=1.0,
        mode=mode,
        errors=errors,
    )
    test = test_run.trades
    orig_mh = [t for t in test if t.confidence in ("HIGH", "MEDIUM")]

    policies = {
        "A_original": "A",
        "B_require_macd_strong": "B",
        "C_require_directional_sr": "C",
        "D_macd_strong_and_sr": "D",
    }
    policy_blocks: dict[str, Any] = {}
    breadth: dict[str, Any] = {}
    removal_counts: dict[str, Any] = {}

    for name, code in policies.items():
        kept = apply_policy(test, code)
        policy_blocks[name] = _bag_block(kept, code)
        if code == "A":
            removal_counts[name] = {
                "removed_mh": 0,
                "kept_mh": len(orig_mh),
                "kept_total": len(kept),
            }
            breadth[name] = None
        else:
            rem = _removed_mh(test, code)
            removal_counts[name] = {
                "removed_mh": len(rem),
                "kept_mh": len(orig_mh) - len(rem),
                "kept_total": len(kept),
            }
            breadth[name] = _breadth(test, code, kept)

    # Feature hit rates on TEST MH (context)
    feature_rates = {
        "macd_strong": sum(1 for t in orig_mh if has_macd_strong(t)),
        "directional_sr": sum(1 for t in orig_mh if has_directional_sr(t)),
        "both": sum(
            1 for t in orig_mh if has_macd_strong(t) and has_directional_sr(t)
        ),
        "mh_total": len(orig_mh),
    }

    return {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": sorted({k for k, _ in series_map}),
        "timeframes": sorted({t for _, t in series_map}),
        "errors": errors,
        "rules": "original",
        "note": (
            "Post-hoc confluence filters on the same chronological TEST signals. "
            "LOW never gated. Scoring thresholds unchanged. Live scanner not modified."
        ),
        "feature_rates_test_mh": feature_rates,
        "removal_counts": removal_counts,
        "policies_test": policy_blocks,
        "breadth_test": breadth,
        "train_mh_count": sum(
            1 for t in train_run.trades if t.confidence in ("HIGH", "MEDIUM")
        ),
        "min_signals_for_conclusion": MIN_SIGNALS_FOR_CONCLUSION,
    }
