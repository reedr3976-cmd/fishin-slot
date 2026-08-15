"""Regime + walk-forward study for commodity fragility (analysis only).

Control = ORIGINAL 4H MEDIUM/HIGH with sma_stack, fixed 4-bar hold.
Thresholds for regime filters are frozen from TRAIN only (no per-symbol fit).
Live scanner is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    FIXED_HOLD,
    EntrySignal,
    ExitPolicy,
    collect_entries,
    realize_entries,
)
from backtest.metrics import TradeResult, summarize_trades
from scanner.scoring import ORIGINAL_RULES


def _is_mh_trend(e: EntrySignal) -> bool:
    return (
        e.confidence in ("HIGH", "MEDIUM")
        and int((e.feature_flags or {}).get("sma_stack", 0) or 0) == 1
    )


def _metrics(trades: list[TradeResult], label: str) -> dict[str, Any]:
    bag = summarize_trades(label, trades)
    return bag.to_dict()


def _by_class(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    return {
        cls: _metrics([t for t in trades if t.asset_class == cls], cls)
        for cls in ("commodity", "forex", "stocks")
    }


def _by_symbol(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    out = {}
    for inst in sorted({t.instrument for t in trades}):
        subset = [t for t in trades if t.instrument == inst]
        d = _metrics(subset, inst)
        d["asset_class"] = subset[0].asset_class
        out[inst] = d
    return out


def _collect_full(series_map):
    collected = {}
    for key, series in series_map.items():
        entries = collect_entries(series, ORIGINAL_RULES)
        # Attach rolling ATR% median rank (lookback 50) using only past atr_pcts
        atr_hist: list[float] = []
        enriched: list[EntrySignal] = []
        for e in entries:
            vol_ratio = None
            if e.atr_pct is not None:
                if len(atr_hist) >= 20:
                    med = float(np.median(atr_hist[-50:]))
                    if med > 0:
                        vol_ratio = e.atr_pct / med
                atr_hist.append(e.atr_pct)
            # store vol_ratio in feature_flags-like via monkey field — use feature_flags copy
            flags = dict(e.feature_flags or {})
            # encode as milliscale int for portability: vol_ratio * 1000
            if vol_ratio is not None:
                flags["_vol_ratio_x1000"] = int(round(vol_ratio * 1000))
            if e.trend_sep_atr is not None:
                flags["_trend_sep_x100"] = int(round(e.trend_sep_atr * 100))
            if e.atr_pct is not None:
                flags["_atr_pct_x10000"] = int(round(e.atr_pct * 10000))
            enriched.append(
                EntrySignal(
                    instrument=e.instrument,
                    asset_class=e.asset_class,
                    timeframe=e.timeframe,
                    confidence=e.confidence,
                    direction=e.direction,
                    score=e.score,
                    entry_idx=e.entry_idx,
                    entry_ts=e.entry_ts,
                    entry_price=e.entry_price,
                    atr_at_entry=e.atr_at_entry,
                    feature_flags=flags,
                    rules_name=e.rules_name,
                    atr_pct=e.atr_pct,
                    trend_sep_atr=e.trend_sep_atr,
                    sma20=e.sma20,
                    sma50=e.sma50,
                )
            )
        collected[key] = (series, enriched)
    return collected


def _slice_entries(collected, start_frac: float, end_frac: float):
    out = {}
    for key, (series, entries) in collected.items():
        n = len(series)
        lo, hi = int(n * start_frac), int(n * end_frac)
        out[key] = (series, [e for e in entries if lo <= e.entry_idx < hi])
    return out


def _realize(
    collected,
    entries_filter: Callable[[EntrySignal], bool],
    policy: ExitPolicy = FIXED_HOLD,
    *,
    policy_by_class: Optional[dict[str, ExitPolicy]] = None,
    cost_overrides: Optional[dict[str, float]] = None,
) -> list[TradeResult]:
    trades: list[TradeResult] = []
    for _key, (series, entries) in collected.items():
        kept = [e for e in entries if entries_filter(e)]
        pol = FIXED_HOLD
        if policy_by_class and series.asset_class in policy_by_class:
            pol = policy_by_class[series.asset_class]
        else:
            pol = policy
        override = None
        if cost_overrides and series.asset_class in cost_overrides:
            override = cost_overrides[series.asset_class]
        trades.extend(
            realize_entries(series, kept, pol, cost_override=override)
        )
    return trades


def _vol_ratio(e: EntrySignal) -> Optional[float]:
    v = (e.feature_flags or {}).get("_vol_ratio_x1000")
    return None if v is None else v / 1000.0


def _train_medians(train_entries: list[EntrySignal], asset_class: str) -> dict[str, float]:
    subset = [
        e
        for e in train_entries
        if e.asset_class == asset_class and _is_mh_trend(e)
    ]
    atrs = [e.atr_pct for e in subset if e.atr_pct is not None]
    seps = [e.trend_sep_atr for e in subset if e.trend_sep_atr is not None]
    vols = [v for e in subset if (v := _vol_ratio(e)) is not None]
    return {
        "atr_pct_median": float(np.median(atrs)) if atrs else 0.0,
        "trend_sep_median": float(np.median(seps)) if seps else 0.0,
        "vol_ratio_median": float(np.median(vols)) if vols else 1.0,
        "n": len(subset),
    }


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    # filter applied to entries; policy may vary by asset class via wrapper
    filter_fn: Callable[[EntrySignal], bool]
    policy: ExitPolicy = FIXED_HOLD
    cost_mult_commodity: float = 1.0


def _flatten_entries(collected) -> list[EntrySignal]:
    out: list[EntrySignal] = []
    for _k, (_s, entries) in collected.items():
        out.extend(entries)
    return out


def _regime_label(e: EntrySignal, med: dict[str, float]) -> str:
    """Assign a simple regime tag for diagnostics (not a fitted model)."""
    atr = e.atr_pct
    sep = e.trend_sep_atr
    high_vol = atr is not None and atr >= med["atr_pct_median"]
    strong_trend = sep is not None and sep >= med["trend_sep_median"]
    if strong_trend and high_vol:
        return "trend_highvol"
    if strong_trend and not high_vol:
        return "trend_lowvol"
    if (not strong_trend) and high_vol:
        return "range_highvol"
    return "range_lowvol"


def _pack(trades: list[TradeResult], label: str) -> dict[str, Any]:
    return {
        "all": _metrics(trades, label),
        "by_asset_class": _by_class(trades),
        "by_symbol": _by_symbol(trades),
        "oil_silver": _metrics(
            [t for t in trades if t.instrument in ("USOIL", "XAGUSD")],
            f"{label}/oil_silver",
        ),
        "xau": _metrics([t for t in trades if t.instrument == "XAUUSD"], f"{label}/xau"),
    }


def run_regime_walkforward(
    *,
    demo: bool = False,
    instruments=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    keys = list(instruments) if instruments else list(study_instruments().keys())
    keys = [k for k in keys if k in study_instruments()]

    print("  loading 4H...", flush=True)
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=demo)
    print("  collecting entries + regime features...", flush=True)
    full = _collect_full(series_map)
    train_c = _slice_entries(full, 0.0, train_fraction)
    test_c = _slice_entries(full, train_fraction, 1.0)

    train_flat = _flatten_entries(train_c)
    comm_med = _train_medians(train_flat, "commodity")
    fx_med = _train_medians(train_flat, "forex")
    stk_med = _train_medians(train_flat, "stocks")

    # --- Diagnose commodity regimes on each chronological third ---
    folds = [
        ("F1", 0.0, 0.25),
        ("F2", 0.25, 0.50),
        ("F3", 0.50, 0.75),
        ("F4", 0.75, 1.0),
    ]

    def baseline_filter(e: EntrySignal) -> bool:
        return _is_mh_trend(e)

    print("  diagnosing regimes by fold...", flush=True)
    fold_diag = {}
    for name, a, b in folds:
        coll = _slice_entries(full, a, b)
        trades = _realize(coll, baseline_filter, FIXED_HOLD)
        # regime breakdown for commodities using commodity train medians
        entries = [e for e in _flatten_entries(coll) if _is_mh_trend(e)]
        regime_trades: dict[str, list[TradeResult]] = {
            "trend_highvol": [],
            "trend_lowvol": [],
            "range_highvol": [],
            "range_lowvol": [],
        }
        # map entry_idx+instrument -> trade
        tmap = {(t.instrument, t.entry_idx): t for t in trades}
        for e in entries:
            if e.asset_class != "commodity":
                continue
            lab = _regime_label(e, comm_med)
            tr = tmap.get((e.instrument, e.entry_idx))
            if tr:
                regime_trades[lab].append(tr)
        fold_diag[name] = {
            "window": [a, b],
            "all": _pack(trades, name),
            "commodity_regimes": {
                lab: _metrics(ts, f"{name}/{lab}") for lab, ts in regime_trades.items()
            },
            "oil_silver_regimes": {
                lab: _metrics(
                    [t for t in ts if t.instrument in ("USOIL", "XAGUSD")],
                    f"{name}/{lab}/os",
                )
                for lab, ts in regime_trades.items()
            },
        }

    # --- Candidates (thresholds frozen from TRAIN medians; no per-symbol tuning) ---
    atr_c = comm_med["atr_pct_median"]
    sep_c = comm_med["trend_sep_median"]
    vol_c = comm_med["vol_ratio_median"]
    atr_f = fx_med["atr_pct_median"]
    sep_f = fx_med["trend_sep_median"]
    atr_s = stk_med["atr_pct_median"]
    sep_s = stk_med["trend_sep_median"]

    def orig(e: EntrySignal) -> bool:
        return _is_mh_trend(e)

    def comm_high_vol(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "commodity":
            return True  # leave FX/stocks as original
        return e.atr_pct is not None and e.atr_pct >= atr_c

    def comm_strong_trend(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "commodity":
            return True
        return e.trend_sep_atr is not None and e.trend_sep_atr >= sep_c

    def comm_highvol_strongtrend(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "commodity":
            return True
        return (
            e.atr_pct is not None
            and e.atr_pct >= atr_c
            and e.trend_sep_atr is not None
            and e.trend_sep_atr >= sep_c
        )

    def comm_vol_ratio(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "commodity":
            return True
        vr = _vol_ratio(e)
        return vr is not None and vr >= max(1.0, vol_c)

    def fx_strong_trend_only(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "forex":
            return True
        return e.trend_sep_atr is not None and e.trend_sep_atr >= sep_f

    def fx_score_45(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "forex":
            return True
        return e.score >= 45

    def stocks_strong_trend(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "stocks":
            return True
        return e.trend_sep_atr is not None and e.trend_sep_atr >= sep_s

    def stocks_score_45(e: EntrySignal) -> bool:
        if not _is_mh_trend(e):
            return False
        if e.asset_class != "stocks":
            return True
        return e.score >= 45

    hold2 = ExitPolicy(name="hold_2", mode="fixed", horizon_bars=2)
    hold6 = ExitPolicy(name="hold_6", mode="fixed", horizon_bars=6)
    hold8 = ExitPolicy(name="hold_8", mode="fixed", horizon_bars=8)

    @dataclass
    class CandSpec:
        name: str
        description: str
        filter_fn: Callable[[EntrySignal], bool]
        policy: ExitPolicy = FIXED_HOLD
        policy_by_class: Optional[dict[str, ExitPolicy]] = None

    specs: list[CandSpec] = [
        CandSpec("ORIGINAL", "MH trending + fixed 4-bar hold", orig),
        CandSpec(
            "COMM_high_ATR",
            f"Commodity atr_pct≥train median ({atr_c:.4f}); FX/stocks unchanged",
            comm_high_vol,
        ),
        CandSpec(
            "COMM_strong_trend",
            f"Commodity |SMA20-SMA50|/ATR≥train median ({sep_c:.3f})",
            comm_strong_trend,
        ),
        CandSpec(
            "COMM_highvol_and_strongtrend",
            "Commodity needs both high ATR% and strong SMA separation",
            comm_highvol_strongtrend,
        ),
        CandSpec(
            "COMM_elevated_vol_ratio",
            f"Commodity ATR%/recent-median ≥ max(1, {vol_c:.2f})",
            comm_vol_ratio,
        ),
        CandSpec("FX_strong_trend", "Forex stronger SMA separation only", fx_strong_trend_only),
        CandSpec("FX_score_ge_45", "Forex score ≥ 45 only", fx_score_45),
        CandSpec(
            "FX_hold_2",
            "Forex hold 2 bars; commodity/stocks stay 4",
            orig,
            policy_by_class={"forex": hold2, "commodity": FIXED_HOLD, "stocks": FIXED_HOLD},
        ),
        CandSpec(
            "FX_hold_6",
            "Forex hold 6 bars; others stay 4",
            orig,
            policy_by_class={"forex": hold6, "commodity": FIXED_HOLD, "stocks": FIXED_HOLD},
        ),
        CandSpec("STK_strong_trend", "Stocks stronger SMA separation only", stocks_strong_trend),
        CandSpec("STK_score_ge_45", "Stocks score ≥ 45 only", stocks_score_45),
        CandSpec(
            "STK_hold_6",
            "Stocks hold 6 bars; others stay 4",
            orig,
            policy_by_class={"stocks": hold6, "commodity": FIXED_HOLD, "forex": FIXED_HOLD},
        ),
        CandSpec("ALL_hold_8", "All classes hold 8 bars", orig, policy=hold8),
    ]

    print("  evaluating candidates on TEST + folds...", flush=True)
    candidate_results = {}
    for spec in specs:
        test_trades = _realize(
            test_c,
            spec.filter_fn,
            spec.policy,
            policy_by_class=spec.policy_by_class,
        )
        fold_packs = {}
        for fname, a, b in folds:
            coll = _slice_entries(full, a, b)
            fold_packs[fname] = _pack(
                _realize(
                    coll,
                    spec.filter_fn,
                    spec.policy,
                    policy_by_class=spec.policy_by_class,
                ),
                f"{spec.name}/{fname}",
            )

        cost_stress = {}
        for mult, label in ((1.0, "1x"), (2.0, "2x"), (3.0, "3x")):
            overrides = {
                "commodity": ROUND_TRIP_COST["commodity"] * mult,
                "forex": ROUND_TRIP_COST["forex"] * mult,
                "stocks": ROUND_TRIP_COST["stocks"] * mult,
            }
            tr = _realize(
                test_c,
                spec.filter_fn,
                spec.policy,
                policy_by_class=spec.policy_by_class,
                cost_overrides=overrides,
            )
            cost_stress[label] = _pack(tr, f"{spec.name}/cost{label}")

        comm_pos = sum(
            1
            for pack in fold_packs.values()
            if (pack["by_asset_class"]["commodity"].get("avg_return") or -1) > 0
        )
        all_pos = sum(
            1
            for pack in fold_packs.values()
            if (pack["all"].get("avg_return") or -1) > 0
        )
        oil_pos = sum(
            1
            for pack in fold_packs.values()
            if (pack["oil_silver"].get("avg_return") or -1) > 0
        )

        candidate_results[spec.name] = {
            "description": spec.description,
            "policy": spec.policy.name,
            "test": _pack(test_trades, spec.name),
            "folds": fold_packs,
            "cost_stress_test": cost_stress,
            "folds_commodity_positive": comm_pos,
            "folds_oil_silver_positive": oil_pos,
            "folds_all_positive": all_pos,
            "removed_vs_original_test": None,
        }

    orig_test = _realize(test_c, orig, FIXED_HOLD)
    orig_n = len(orig_test)
    for name, block in candidate_results.items():
        block["removed_vs_original_test"] = orig_n - block["test"]["all"]["signals"]

    return {
        "mode": "demo" if demo else "public_historical",
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": sorted({k for k, _ in series_map}),
        "asset_classes": list(STUDY_ASSET_CLASSES),
        "timeframe": "4h",
        "errors": errors,
        "live_unchanged": True,
        "train_medians": {
            "commodity": comm_med,
            "forex": fx_med,
            "stocks": stk_med,
        },
        "fold_regime_diagnostics": fold_diag,
        "candidates": candidate_results,
        "original_test_n": orig_n,
        "min_signals": MIN_SIGNALS_FOR_CONCLUSION,
        "note": (
            "Regime thresholds frozen from TRAIN class medians only. "
            "No per-symbol optimization. No live changes."
        ),
    }
