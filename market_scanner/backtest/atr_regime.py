"""ATR / volatility regime filter study (analysis only).

Policies (ORIGINAL scoring thresholds unchanged; LOW never gated):
  A — Original: keep all HIGH / MEDIUM / LOW
  B — Suppress MEDIUM/HIGH when feature flag high_atr is active

Live scanner is not modified by this module.
"""

from __future__ import annotations

from typing import Any, Optional

from config import VALIDATION_TRAIN_FRACTION
from backtest.engine import load_series_map, run_backtest_on_map
from backtest.metrics import TradeResult, group_metrics, summarize_trades
from scanner.scoring import ORIGINAL_RULES


def _is_high_atr(trade: TradeResult) -> bool:
    flags = trade.feature_flags or {}
    return int(flags.get("high_atr", 0) or 0) == 1


def apply_atr_policy(trades: list[TradeResult], policy: str) -> list[TradeResult]:
    """Filter trades by ATR policy. LOW always kept."""
    if policy == "A":
        return list(trades)
    if policy != "B":
        raise ValueError(f"Unknown ATR policy: {policy}")
    kept: list[TradeResult] = []
    for t in trades:
        if t.confidence == "LOW":
            kept.append(t)
            continue
        if t.confidence in ("HIGH", "MEDIUM") and _is_high_atr(t):
            continue
        kept.append(t)
    return kept


def _bag_block(trades: list[TradeResult], name: str) -> dict[str, Any]:
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
            for cls in sorted({t.asset_class for t in trades})
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
            for cls in sorted({t.asset_class for t in trades})
        },
    }


def _breadth_analysis(
    original: list[TradeResult], removed: list[TradeResult]
) -> dict[str, Any]:
    """Is any A→B lift concentrated in a few trades or one asset class?"""
    removed_mh = [
        t for t in removed if t.confidence in ("HIGH", "MEDIUM")
    ]
    removed_nets = [t.net_return for t in removed_mh]
    total_removed = float(sum(removed_nets)) if removed_nets else 0.0

    by_class: dict[str, Any] = {}
    for cls in sorted({t.asset_class for t in removed_mh} | {"forex", "commodity"}):
        subset = [t for t in removed_mh if t.asset_class == cls]
        nets = [t.net_return for t in subset]
        by_class[cls] = {
            "n": len(subset),
            "sum_net": float(sum(nets)) if nets else 0.0,
            "avg_net": float(sum(nets) / len(nets)) if nets else None,
            "share_of_removed_sum": (
                float(sum(nets) / total_removed) if nets and total_removed != 0 else None
            ),
        }

    by_instrument: list[dict[str, Any]] = []
    for inst in sorted({t.instrument for t in removed_mh}):
        subset = [t for t in removed_mh if t.instrument == inst]
        nets = [t.net_return for t in subset]
        by_instrument.append(
            {
                "instrument": inst,
                "asset_class": subset[0].asset_class if subset else "",
                "n": len(subset),
                "sum_net": float(sum(nets)) if nets else 0.0,
                "avg_net": float(sum(nets) / len(nets)) if nets else None,
            }
        )
    by_instrument.sort(key=lambda x: abs(x["sum_net"]), reverse=True)

    # Concentration: share of |sum| from worst / best few removed trades
    ranked = sorted(removed_mh, key=lambda t: t.net_return)
    def top_share(k: int, worst: bool) -> Optional[float]:
        if not removed_mh or total_removed == 0:
            return None
        picks = ranked[:k] if worst else ranked[-k:]
        s = float(sum(t.net_return for t in picks))
        return s / total_removed

    # Leave-one-class: MH metrics on original without that class's removed trades
    # (i.e. apply B only within one class) — already covered by by_asset_class_mh
    orig_mh = [t for t in original if t.confidence in ("HIGH", "MEDIUM")]
    kept_b_mh = [t for t in orig_mh if not _is_high_atr(t)]

    return {
        "removed_mh_count": len(removed_mh),
        "removed_mh_summary": summarize_trades("removed_MH", removed_mh),
        "removed_sum_net": total_removed,
        "removed_by_asset_class": by_class,
        "removed_by_instrument": by_instrument,
        "concentration": {
            "worst_1_share_of_removed_sum": top_share(1, True),
            "worst_3_share_of_removed_sum": top_share(3, True),
            "best_1_share_of_removed_sum": top_share(1, False),
            "best_3_share_of_removed_sum": top_share(3, False),
        },
        "kept_vs_removed_mh": {
            "kept_n": len(kept_b_mh),
            "removed_n": len(removed_mh),
            "kept_avg": summarize_trades("kept_MH", kept_b_mh).avg_return,
            "removed_avg": summarize_trades("removed_MH", removed_mh).avg_return,
        },
    }


def run_atr_regime_study(
    *,
    demo: bool = False,
    instruments=None,
    timeframes=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    """Chronological TEST comparison of original vs suppress-high_atr on MEDIUM/HIGH."""
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

    test_trades = test_run.trades
    removed = [
        t
        for t in test_trades
        if t.confidence in ("HIGH", "MEDIUM") and _is_high_atr(t)
    ]
    policy_a = apply_atr_policy(test_trades, "A")
    policy_b = apply_atr_policy(test_trades, "B")

    train_high_atr_mh = sum(
        1
        for t in train_run.trades
        if t.confidence in ("HIGH", "MEDIUM") and _is_high_atr(t)
    )
    test_high_atr_mh = len(removed)

    return {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": sorted({k for k, _ in series_map}),
        "timeframes": sorted({t for _, t in series_map}),
        "errors": errors,
        "rules": "original",
        "note": (
            "Post-hoc filter on the same chronological TEST signals as the original "
            "scanner walk. LOW never gated. Scoring thresholds unchanged. "
            "Live scanner not modified."
        ),
        "counts": {
            "train_mh_high_atr": train_high_atr_mh,
            "test_mh_high_atr_removed": test_high_atr_mh,
            "test_total": len(test_trades),
            "test_a_kept": len(policy_a),
            "test_b_kept": len(policy_b),
        },
        "policies_test": {
            "A_original": _bag_block(policy_a, "A"),
            "B_suppress_high_atr_mh": _bag_block(policy_b, "B"),
        },
        "breadth_test": _breadth_analysis(test_trades, removed),
        "metrics_original_test": group_metrics(test_trades),
    }
