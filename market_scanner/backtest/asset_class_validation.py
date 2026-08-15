"""Focused asset-class / symbol validation (analysis only).

Control = ORIGINAL scanner entries on 4H, trending MH (sma_stack), fixed hold.
Live scanner is not modified.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config import (
    MIN_SIGNALS_FOR_CONCLUSION,
    ROUND_TRIP_COST,
    STUDY_ASSET_CLASSES,
    VALIDATION_TRAIN_FRACTION,
    study_instruments,
)
from backtest.engine import load_series_map
from backtest.fourh_exits import (
    ENTRY_STOP_POLICIES,
    FIXED_HOLD,
    ExitPolicy,
    collect_entries,
    realize_entries,
)
from backtest.metrics import MetricBag, TradeResult, summarize_trades
from scanner.scoring import ORIGINAL_RULES


def _mh_trending(trades: list[TradeResult]) -> list[TradeResult]:
    return [
        t
        for t in trades
        if t.confidence in ("HIGH", "MEDIUM")
        and int((t.feature_flags or {}).get("sma_stack", 0) or 0) == 1
    ]


def _bag_dict(bag: MetricBag) -> dict[str, Any]:
    return bag.to_dict()


def _metrics(trades: list[TradeResult], label: str) -> dict[str, Any]:
    bag = summarize_trades(label, trades)
    d = _bag_dict(bag)
    d["avg_winner"] = bag.avg_winner
    d["avg_loser"] = bag.avg_loser
    d["profit_factor"] = bag.profit_factor
    return d


def _by_symbol(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for inst in sorted({t.instrument for t in trades}):
        subset = [t for t in trades if t.instrument == inst]
        out[inst] = {
            "asset_class": subset[0].asset_class,
            **_metrics(subset, inst),
        }
    return out


def _by_class(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cls in ("commodity", "forex", "stocks"):
        subset = [t for t in trades if t.asset_class == cls]
        out[cls] = _metrics(subset, cls)
    return out


def _collect_window(series_map, start_frac: float, end_frac: float):
    collected = {}
    for key, series in series_map.items():
        n = len(series)
        entries = collect_entries(
            series,
            ORIGINAL_RULES,
            start_idx=int(n * start_frac),
            end_idx_exclusive=int(n * end_frac),
            require_trending=False,
        )
        # Keep only MH + sma_stack (trending focus; original scores)
        entries = [
            e
            for e in entries
            if e.confidence in ("HIGH", "MEDIUM")
            and int((e.feature_flags or {}).get("sma_stack", 0) or 0) == 1
        ]
        collected[key] = (series, entries)
    return collected


def _realize(
    collected,
    policy: ExitPolicy = FIXED_HOLD,
    *,
    cost_overrides: Optional[dict[str, float]] = None,
) -> list[TradeResult]:
    trades: list[TradeResult] = []
    for _key, (series, entries) in collected.items():
        override = None
        if cost_overrides and series.asset_class in cost_overrides:
            override = cost_overrides[series.asset_class]
        trades.extend(
            realize_entries(series, entries, policy, cost_override=override)
        )
    return trades


def _split_trades_by_time(
    trades: list[TradeResult], n_periods: int = 3
) -> list[list[TradeResult]]:
    if not trades:
        return [[] for _ in range(n_periods)]
    ordered = sorted(trades, key=lambda t: t.entry_ts)
    ts = [t.entry_ts for t in ordered]
    lo, hi = ts[0], ts[-1]
    if hi <= lo:
        return [ordered] + [[] for _ in range(n_periods - 1)]
    edges = [lo + (hi - lo) * i / n_periods for i in range(n_periods + 1)]
    buckets: list[list[TradeResult]] = [[] for _ in range(n_periods)]
    for t in ordered:
        # last edge inclusive
        idx = n_periods - 1
        for i in range(n_periods):
            if t.entry_ts < edges[i + 1] or i == n_periods - 1:
                idx = i
                if t.entry_ts < edges[i + 1] or i == n_periods - 1:
                    if i < n_periods - 1 and t.entry_ts >= edges[i + 1]:
                        continue
                break
        # simpler assignment
        for i in range(n_periods):
            left = edges[i]
            right = edges[i + 1]
            if i == n_periods - 1:
                if left <= t.entry_ts <= right:
                    buckets[i].append(t)
                    break
            elif left <= t.entry_ts < right:
                buckets[i].append(t)
                break
    return buckets


def _contrast_symbols(
    trades: list[TradeResult], strong: list[str], weak: list[str]
) -> dict[str, Any]:
    def profile(names: list[str]) -> dict[str, Any]:
        subset = [t for t in trades if t.instrument in names]
        if not subset:
            return {"n": 0}
        atr_pcts = [
            t.atr_at_entry / t.entry_price
            for t in subset
            if t.atr_at_entry and t.entry_price > 0
        ]
        wins = [t for t in subset if t.win]
        losses = [t for t in subset if not t.win]
        return {
            "n": len(subset),
            "symbols": names,
            "metrics": _metrics(subset, "+".join(names)),
            "avg_score": float(np.mean([t.score for t in subset])),
            "avg_atr_pct": float(np.mean(atr_pcts)) if atr_pcts else None,
            "bullish_share": sum(1 for t in subset if t.direction == "bullish")
            / len(subset),
            "avg_winner": float(np.mean([t.net_return for t in wins])) if wins else None,
            "avg_loser": float(np.mean([t.net_return for t in losses])) if losses else None,
            "feature_rates": {
                f: sum(1 for t in subset if int((t.feature_flags or {}).get(f, 0)) == 1)
                / len(subset)
                for f in (
                    "sma_cross",
                    "rsi_mild",
                    "rsi_extreme",
                    "rsi_exit",
                    "macd_strong",
                    "macd_cross",
                    "bb_touch",
                    "high_atr",
                )
            },
        }

    return {"strong": profile(strong), "weak": profile(weak)}


def run_asset_class_validation(
    *,
    demo: bool = False,
    instruments=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    keys = (
        list(instruments)
        if instruments is not None
        else list(study_instruments().keys())
    )
    keys = [k for k in keys if k in study_instruments()]

    print("  loading 4H series...", flush=True)
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=demo)

    print("  collecting ORIGINAL trending MH entries...", flush=True)
    # Full series chronological: train / test
    train_c = _collect_window(series_map, 0.0, train_fraction)
    test_c = _collect_window(series_map, train_fraction, 1.0)

    # Also three equal chronological thirds of the FULL sample for period robustness
    # (each third is a separate OOS-style window; not used to fit parameters)
    p1 = _collect_window(series_map, 0.0, 1 / 3)
    p2 = _collect_window(series_map, 1 / 3, 2 / 3)
    p3 = _collect_window(series_map, 2 / 3, 1.0)

    print("  realizing control (fixed hold)...", flush=True)
    test_fixed = _realize(test_c, FIXED_HOLD)
    train_fixed = _realize(train_c, FIXED_HOLD)

    # Period robustness — commodities (and all classes) on each third
    period_results = {}
    for name, coll in (("P1_early", p1), ("P2_mid", p2), ("P3_late", p3)):
        trades = _realize(coll, FIXED_HOLD)
        period_results[name] = {
            "all": _metrics(trades, name),
            "by_asset_class": _by_class(trades),
            "by_symbol": _by_symbol(trades),
            "commodity_symbols": {
                s: m
                for s, m in _by_symbol(trades).items()
                if m["asset_class"] == "commodity"
            },
        }

    # Within TEST: early vs late half (stability of commodity edge inside OOS)
    test_ordered = sorted(test_fixed, key=lambda t: t.entry_ts)
    mid = len(test_ordered) // 2
    test_early = test_ordered[:mid]
    test_late = test_ordered[mid:]

    print("  realizing ATR stops on same TEST entries...", flush=True)
    stop_by_policy: dict[str, Any] = {}
    for pol in ENTRY_STOP_POLICIES:
        trades = _realize(test_c, pol)
        stop_by_policy[pol.name] = {
            "all": _metrics(trades, pol.name),
            "by_asset_class": _by_class(trades),
            "by_symbol": _by_symbol(trades),
        }

    # Cost stress for commodities only (same entries, fixed hold)
    print("  commodity cost stress...", flush=True)
    base_comm_cost = ROUND_TRIP_COST["commodity"]
    cost_stress = {}
    for mult, label in (
        (1.0, "base_10bps"),
        (2.0, "stress_20bps"),
        (3.0, "stress_30bps"),
        (5.0, "stress_50bps"),
    ):
        cost = base_comm_cost * mult
        trades = _realize(test_c, FIXED_HOLD, cost_overrides={"commodity": cost})
        comm = [t for t in trades if t.asset_class == "commodity"]
        cost_stress[label] = {
            "round_trip_cost": cost,
            "commodity": _metrics(comm, f"comm_{label}"),
            "USOIL": _metrics([t for t in comm if t.instrument == "USOIL"], "USOIL"),
            "XAGUSD": _metrics([t for t in comm if t.instrument == "XAGUSD"], "XAGUSD"),
            "XAUUSD": _metrics([t for t in comm if t.instrument == "XAUUSD"], "XAUUSD"),
        }

    # Symbol exclusion proposal: rank on TRAIN, validate on TEST (unseen for ranking)
    # Exclude symbols with TRAIN avg_return < 0 and n >= 15 (pre-specified, not optimized)
    train_sym = _by_symbol(train_fixed)
    exclude_candidates = sorted(
        [
            s
            for s, m in train_sym.items()
            if (m.get("avg_return") is not None)
            and m["avg_return"] < 0
            and m["signals"] >= 15
        ]
    )
    # Also a stricter set: worst TRAIN avg among those with n>=20
    worst_train = sorted(
        [
            (s, m["avg_return"], m["signals"])
            for s, m in train_sym.items()
            if m.get("avg_return") is not None and m["signals"] >= 20
        ],
        key=lambda x: x[1],
    )
    exclude_worst3 = [s for s, _, _ in worst_train[:3]]

    def apply_exclude(trades: list[TradeResult], banned: list[str]) -> list[TradeResult]:
        return [t for t in trades if t.instrument not in banned]

    exclusion = {
        "rule": (
            "Pre-specified on TRAIN only: exclude symbols with TRAIN avg_return < 0 "
            "and n>=15. Also report worst-3 by TRAIN avg among n>=20. "
            "Validated on untouched TEST. Symbols NOT removed from live catalog."
        ),
        "exclude_negative_train": {
            "symbols": exclude_candidates,
            "train_kept": _metrics(
                apply_exclude(train_fixed, exclude_candidates), "train_excl_neg"
            ),
            "test_baseline": _metrics(test_fixed, "test_base"),
            "test_kept": _metrics(
                apply_exclude(test_fixed, exclude_candidates), "test_excl_neg"
            ),
            "test_removed": _metrics(
                [t for t in test_fixed if t.instrument in exclude_candidates],
                "test_removed_neg",
            ),
        },
        "exclude_worst3_train": {
            "symbols": exclude_worst3,
            "train_ranks": [
                {"symbol": s, "avg": a, "n": n} for s, a, n in worst_train[:5]
            ],
            "test_baseline": _metrics(test_fixed, "test_base"),
            "test_kept": _metrics(
                apply_exclude(test_fixed, exclude_worst3), "test_excl_w3"
            ),
            "test_removed": _metrics(
                [t for t in test_fixed if t.instrument in exclude_worst3],
                "test_removed_w3",
            ),
        },
    }

    contrast = _contrast_symbols(
        test_fixed,
        strong=["USOIL", "XAGUSD"],
        weak=["EURUSD", "QQQ", "AUDUSD", "GBPUSD"],
    )

    return {
        "mode": "demo" if demo else "public_historical",
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": sorted({k for k, _ in series_map}),
        "asset_classes": list(STUDY_ASSET_CLASSES),
        "timeframe": "4h",
        "errors": errors,
        "control": "ORIGINAL scoring · trending MH (sma_stack) · fixed 4-bar hold",
        "live_unchanged": True,
        "no_take_profit": True,
        "baseline_test": {
            "all": _metrics(test_fixed, "test_all"),
            "by_asset_class": _by_class(test_fixed),
            "by_symbol": _by_symbol(test_fixed),
        },
        "baseline_train": {
            "all": _metrics(train_fixed, "train_all"),
            "by_asset_class": _by_class(train_fixed),
            "by_symbol": _by_symbol(train_fixed),
        },
        "test_early_vs_late": {
            "early": {
                "all": _metrics(test_early, "test_early"),
                "by_asset_class": _by_class(test_early),
            },
            "late": {
                "all": _metrics(test_late, "test_late"),
                "by_asset_class": _by_class(test_late),
            },
        },
        "period_thirds": period_results,
        "stops_test": stop_by_policy,
        "commodity_cost_stress": cost_stress,
        "symbol_exclusion_study": exclusion,
        "why_strong_vs_weak": contrast,
        "min_signals": MIN_SIGNALS_FOR_CONCLUSION,
    }
