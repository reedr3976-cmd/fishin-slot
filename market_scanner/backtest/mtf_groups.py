"""Classify MEDIUM/HIGH MTF relationships and compare filter policies A/B/C.

Groups (for clear 1d signals vs weekly confirmation):
  AGREE           — 1wk direction clear and matches 1d
  DISAGREE        — 1wk direction clear and opposite to 1d
  WEEKLY_UNKNOWN  — 1wk direction missing/unclear (neutral / not ready)

Policies (score thresholds unchanged; LOW never MTF-gated):
  A — Original: keep all MEDIUM/HIGH
  B — Suppress only genuine DISAGREE
  C — Require explicit AGREE (suppress DISAGREE + WEEKLY_UNKNOWN)
"""

from __future__ import annotations

from typing import Any, Optional

from config import BACKTEST_WARMUP_BARS, FORWARD_BARS, ROUND_TRIP_COST, SMA_SLOW, VALIDATION_TRAIN_FRACTION
from models import CandleSeries
from scanner.mtf_filter import directions_agree
from scanner.opportunity import evaluate_opportunity
from scanner.scoring import ORIGINAL_RULES, ScoringRules
from backtest.engine import _slice_series, load_series_map, BacktestRun
from backtest.metrics import TradeResult, group_metrics, summarize_trades
from backtest.mtf_backtest import _direction_at_ts


def _make_trade(
    series: CandleSeries,
    i: int,
    horizon: int,
    opp,
    *,
    mtf_group: str,
    rules_name: str,
) -> TradeResult:
    cost = ROUND_TRIP_COST.get(series.asset_class, 0.001)
    entry = float(series.close[i])
    exit_px = float(series.close[i + horizon])
    if opp.direction == "bullish":
        gross = (exit_px - entry) / entry
    else:
        gross = (entry - exit_px) / entry
    net = gross - cost
    return TradeResult(
        instrument=series.instrument,
        asset_class=series.asset_class,
        timeframe=series.timeframe,
        confidence=opp.confidence,
        direction=opp.direction,
        score=opp.score,
        entry_idx=i,
        exit_idx=i + horizon,
        entry_ts=int(series.timestamps[i]),
        exit_ts=int(series.timestamps[i + horizon]),
        entry_price=entry,
        exit_price=exit_px,
        gross_return=gross,
        cost=cost,
        net_return=net,
        win=net > 0,
        feature_flags=dict(opp.feature_flags),
        rules_name=rules_name,
        mtf_status=mtf_group,
    )


def classify_daily_mh_signals(
    daily: CandleSeries,
    weekly: CandleSeries,
    rules: ScoringRules,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
) -> tuple[list[TradeResult], dict[str, int]]:
    """Walk 1d series; tag each MEDIUM/HIGH with AGREE/DISAGREE/WEEKLY_UNKNOWN.

    Also records LOW trades (mtf_status=n/a) for policy overall stats.
    Non-overlapping within this series.
    """
    horizon = FORWARD_BARS["1d"]
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5)
    trades: list[TradeResult] = []
    counts = {"AGREE": 0, "DISAGREE": 0, "WEEKLY_UNKNOWN": 0, "LOW": 0}

    i = max(warmup, start_idx or warmup)
    n = len(daily)
    last_start = (
        n - horizon if end_idx_exclusive is None else min(n - horizon, end_idx_exclusive)
    )

    while i < last_start:
        hist = _slice_series(daily, i)
        opp = evaluate_opportunity(hist, daily.instrument, rules=rules)
        if not (
            opp.confidence in ("HIGH", "MEDIUM", "LOW")
            and opp.direction in ("bullish", "bearish")
        ):
            i += 1
            continue

        if opp.confidence == "LOW":
            group = "LOW"
            counts["LOW"] += 1
        else:
            wk_dir, _, _ = _direction_at_ts(weekly, int(daily.timestamps[i]), rules)
            if wk_dir is None:
                group = "WEEKLY_UNKNOWN"
            elif directions_agree(opp.direction, wk_dir):
                group = "AGREE"
            else:
                group = "DISAGREE"
            counts[group] += 1

        trades.append(
            _make_trade(
                daily, i, horizon, opp, mtf_group=group, rules_name=rules.name
            )
        )
        i += horizon

    return trades, counts


def apply_policy(trades: list[TradeResult], policy: str) -> list[TradeResult]:
    """Filter trades by policy A/B/C (LOW always kept)."""
    policy = policy.upper()
    out: list[TradeResult] = []
    for t in trades:
        if t.confidence == "LOW" or t.mtf_status == "LOW":
            out.append(t)
            continue
        if t.confidence not in ("HIGH", "MEDIUM"):
            continue
        if policy == "A":
            out.append(t)
        elif policy == "B":
            # Suppress only genuine DISAGREE
            if t.mtf_status != "DISAGREE":
                out.append(t)
        elif policy == "C":
            # Require explicit AGREE
            if t.mtf_status == "AGREE":
                out.append(t)
        else:
            raise ValueError(f"Unknown policy {policy}")
    return out


def run_mtf_group_study(
    *,
    demo: bool = False,
    instruments=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    """Chronological TEST analysis of MTF groups and policies A/B/C.

    Focuses on 1d signals vs 1wk confirmation to isolate WEEKLY_UNKNOWN.
    """
    series_map, errors, bars = load_series_map(
        instruments, ["1d", "1wk"], demo=demo
    )
    mode = "demo" if demo else "public_historical"
    rules = ORIGINAL_RULES

    paired = sorted(
        {k for k, _ in series_map if (k, "1d") in series_map and (k, "1wk") in series_map}
    )

    train_trades: list[TradeResult] = []
    test_trades: list[TradeResult] = []
    train_counts = {"AGREE": 0, "DISAGREE": 0, "WEEKLY_UNKNOWN": 0, "LOW": 0}
    test_counts = {"AGREE": 0, "DISAGREE": 0, "WEEKLY_UNKNOWN": 0, "LOW": 0}

    for key in paired:
        daily = series_map[(key, "1d")]
        weekly = series_map[(key, "1wk")]
        n = len(daily)
        split = int(n * train_fraction)

        tr, c_tr = classify_daily_mh_signals(
            daily, weekly, rules, start_idx=0, end_idx_exclusive=split
        )
        te, c_te = classify_daily_mh_signals(
            daily, weekly, rules, start_idx=split, end_idx_exclusive=n
        )
        train_trades.extend(tr)
        test_trades.extend(te)
        for k in train_counts:
            train_counts[k] += c_tr[k]
            test_counts[k] += c_te[k]

    # Group performance on TEST for MEDIUM/HIGH only
    mh_test = [t for t in test_trades if t.confidence in ("HIGH", "MEDIUM")]
    groups = {}
    for g in ("AGREE", "DISAGREE", "WEEKLY_UNKNOWN"):
        subset = [t for t in mh_test if t.mtf_status == g]
        groups[g] = {
            "all_mh": summarize_trades(g, subset),
            "HIGH": summarize_trades(f"{g}/HIGH", [t for t in subset if t.confidence == "HIGH"]),
            "MEDIUM": summarize_trades(
                f"{g}/MEDIUM", [t for t in subset if t.confidence == "MEDIUM"]
            ),
            "by_asset_class": {
                cls: summarize_trades(
                    f"{g}/{cls}", [t for t in subset if t.asset_class == cls]
                )
                for cls in sorted({t.asset_class for t in subset})
            },
        }

    policies = {}
    for name, code in (("A_original", "A"), ("B_suppress_disagree_only", "B"), ("C_require_agree", "C")):
        kept = apply_policy(test_trades, code)
        policies[name] = {
            "trades": kept,
            "overall": summarize_trades(name, kept),
            "by_confidence": {
                conf: summarize_trades(
                    f"{name}/{conf}", [t for t in kept if t.confidence == conf]
                )
                for conf in ("HIGH", "MEDIUM", "LOW")
            },
            "medium_high": summarize_trades(
                f"{name}/MH", [t for t in kept if t.confidence in ("HIGH", "MEDIUM")]
            ),
            "by_asset_class": {
                cls: summarize_trades(
                    f"{name}/{cls}", [t for t in kept if t.asset_class == cls]
                )
                for cls in sorted({t.asset_class for t in kept})
            },
            "kept_mh_groups": {
                g: sum(
                    1
                    for t in kept
                    if t.confidence in ("HIGH", "MEDIUM") and t.mtf_status == g
                )
                for g in ("AGREE", "DISAGREE", "WEEKLY_UNKNOWN")
            },
        }

    return {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": paired,
        "errors": errors,
        "focus": "1d MEDIUM/HIGH signals classified vs 1wk confirmation",
        "train_counts": train_counts,
        "test_counts": test_counts,
        "groups_test": groups,
        "policies_test": policies,
    }
