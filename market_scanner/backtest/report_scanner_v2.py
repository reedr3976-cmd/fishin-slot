"""Reporting helpers for Scanner V2 research (analysis only)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np

from config import MIN_SIGNALS_FOR_CONCLUSION, V2_N_FOLDS, V2_TRAIN_FRACTION
from backtest.metrics import MetricBag, summarize_trades
from backtest.scanner_v2 import ALL_STAGES, V2Stage, V2Trade, chronological_folds, run_stage_on_map


def _expectancy(trades: list[V2Trade]) -> Optional[float]:
    if not trades:
        return None
    return float(np.mean([t.net_return for t in trades]))


def _avg_r(trades: list[V2Trade]) -> Optional[float]:
    if not trades:
        return None
    return float(np.mean([t.net_r for t in trades]))


def enrich_bag(bag: MetricBag, trades: list[V2Trade]) -> dict[str, Any]:
    d = bag.to_dict()
    d["expectancy"] = _expectancy(trades)
    d["avg_r"] = _avg_r(trades)
    d["avg_return_pct"] = (bag.avg_return * 100.0) if bag.avg_return is not None else None
    longs = [t for t in trades if t.direction == "bullish"]
    shorts = [t for t in trades if t.direction == "bearish"]
    d["long"] = summarize_trades("long", longs).to_dict()
    d["long"]["expectancy"] = _expectancy(longs)
    d["short"] = summarize_trades("short", shorts).to_dict()
    d["short"]["expectancy"] = _expectancy(shorts)
    return d


def slice_metrics(label: str, trades: list[V2Trade]) -> dict[str, Any]:
    bag = summarize_trades(label, trades)
    return enrich_bag(bag, trades)


def by_key(trades: list[V2Trade], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        groups[str(key_fn(t))].append(t)
    return {k: slice_metrics(k, v) for k, v in sorted(groups.items())}


def evaluate_robustness(
    *,
    test_trades: list[V2Trade],
    fold_expectancies: list[float],
    test_2x: list[V2Trade],
    by_symbol: dict[str, dict[str, Any]],
    by_regime: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Pre-specified success bar (not fitted on TEST)."""
    test_exp = _expectancy(test_trades)
    exp_2x = _expectancy(test_2x)
    n = len(test_trades)
    bag = summarize_trades("test", test_trades)
    pos_folds = sum(1 for e in fold_expectancies if e is not None and e > 0)
    sym_pos = [
        s
        for s, m in by_symbol.items()
        if (m.get("expectancy") or 0) > 0 and (m.get("signals") or 0) >= 5
    ]
    regime_trend = by_regime.get("trending", {})
    checks = {
        "positive_test_expectancy": bool(test_exp is not None and test_exp > 0),
        "folds_positive_ge_3_of_4": pos_folds >= 3,
        "positive_after_2x_costs": bool(exp_2x is not None and exp_2x > 0),
        "adequate_trade_count": n >= MIN_SIGNALS_FOR_CONCLUSION,
        "acceptable_max_drawdown": bool(
            bag.max_drawdown is not None and bag.max_drawdown <= 0.35
        ),
        "not_single_symbol_dependent": len(sym_pos) >= 2,
        "trending_regime_not_only_empty": (regime_trend.get("signals") or 0) > 0,
    }
    checks["folds_positive_count"] = pos_folds
    checks["symbols_positive_ge5"] = sym_pos
    checks["test_expectancy"] = test_exp
    checks["test_2x_expectancy"] = exp_2x
    checks["test_n"] = n
    checks["test_max_dd"] = bag.max_drawdown
    checks["all_pass"] = all(
        checks[k]
        for k in (
            "positive_test_expectancy",
            "folds_positive_ge_3_of_4",
            "positive_after_2x_costs",
            "adequate_trade_count",
            "acceptable_max_drawdown",
            "not_single_symbol_dependent",
        )
    )
    return checks


def select_candidate_on_train(stage_train: dict[str, list[V2Trade]]) -> str:
    """Pick stage by TRAIN expectancy among V2 stages with n>=20; never uses TEST."""
    best_name = "ORIGINAL"
    best_exp = float("-inf")
    for name, trades in stage_train.items():
        if name == "ORIGINAL":
            continue
        if len(trades) < 20:
            continue
        exp = _expectancy(trades) or float("-inf")
        if exp > best_exp:
            best_exp = exp
            best_name = name
    if best_exp == float("-inf"):
        # Fall back to densest V2 stage on train
        v2 = {k: v for k, v in stage_train.items() if k != "ORIGINAL"}
        if v2:
            best_name = max(v2.items(), key=lambda kv: len(kv[1]))[0]
    return best_name


def verdict_from_checks(
    original_checks: dict[str, Any],
    candidate_name: str,
    candidate_checks: dict[str, Any],
    stage_summaries: dict[str, Any],
) -> tuple[str, str]:
    """Return (verdict_code, explanation)."""
    cand_pass = candidate_checks.get("all_pass", False)
    orig_exp = original_checks.get("test_expectancy")
    cand_exp = candidate_checks.get("test_expectancy")

    if cand_pass and cand_exp is not None and (orig_exp is None or cand_exp > orig_exp):
        # Still require broader evidence before paper-trading
        folds = candidate_checks.get("folds_positive_count", 0)
        syms = candidate_checks.get("symbols_positive_ge5", [])
        if folds >= 4 and len(syms) >= 3 and candidate_checks.get("positive_after_2x_costs"):
            return (
                "V2_ROBUST_PAPER",
                "V2 ROBUST ENOUGH FOR PAPER-TRADING CANDIDATE",
            )
        return (
            "V2_PROMISING",
            "V2 PROMISING — CONTINUE RESEARCH",
        )

    if cand_pass:
        return (
            "V2_PROMISING",
            "V2 PROMISING — CONTINUE RESEARCH (passes bar but not clearly better than ORIGINAL)",
        )

    # Near-miss: ≥2 hard checks and positive TEST
    hard = [
        "positive_test_expectancy",
        "folds_positive_ge_3_of_4",
        "positive_after_2x_costs",
        "adequate_trade_count",
        "acceptable_max_drawdown",
        "not_single_symbol_dependent",
    ]
    passed = sum(1 for k in hard if candidate_checks.get(k))
    if candidate_checks.get("positive_test_expectancy") and passed >= 3:
        return (
            "V2_PROMISING",
            "V2 PROMISING — CONTINUE RESEARCH (partial robustness)",
        )

    if orig_exp is not None and orig_exp > 0 and (cand_exp is None or cand_exp <= orig_exp):
        return ("ORIGINAL_BETTER", "ORIGINAL BETTER")

    return ("V2_FAILED", "V2 FAILED — REDESIGN AGAIN")


def build_study_payload(
    series_map: dict,
    *,
    stages: tuple[V2Stage, ...] = ALL_STAGES,
    train_frac: float = V2_TRAIN_FRACTION,
    n_folds: int = V2_N_FOLDS,
) -> dict[str, Any]:
    stage_train: dict[str, list[V2Trade]] = {}
    stage_test: dict[str, list[V2Trade]] = {}
    stage_test_2x: dict[str, list[V2Trade]] = {}
    stage_folds: dict[str, list[dict]] = {}
    summaries: dict[str, Any] = {}

    for stage in stages:
        train = run_stage_on_map(series_map, stage, start_frac=0.0, end_frac=train_frac, cost_mult=1.0)
        test = run_stage_on_map(series_map, stage, start_frac=train_frac, end_frac=1.0, cost_mult=1.0)
        test_2x = run_stage_on_map(series_map, stage, start_frac=train_frac, end_frac=1.0, cost_mult=2.0)
        folds = chronological_folds(series_map, stage, n_folds=n_folds, cost_mult=1.0)
        stage_train[stage.name] = train
        stage_test[stage.name] = test
        stage_test_2x[stage.name] = test_2x
        stage_folds[stage.name] = folds

        fold_exps = []
        fold_rows = []
        for fr in folds:
            exp = _expectancy(fr["trades"])
            fold_exps.append(exp if exp is not None else 0.0)
            fold_rows.append(
                {
                    "fold": fr["fold"],
                    "start_frac": fr["start_frac"],
                    "end_frac": fr["end_frac"],
                    "metrics": slice_metrics(f"fold{fr['fold']}", fr["trades"]),
                }
            )

        by_sym = by_key(test, lambda t: t.instrument)
        by_cls = by_key(test, lambda t: t.asset_class)
        by_reg = by_key(test, lambda t: t.regime)
        checks = evaluate_robustness(
            test_trades=test,
            fold_expectancies=fold_exps,
            test_2x=test_2x,
            by_symbol=by_sym,
            by_regime=by_reg,
        )
        summaries[stage.name] = {
            "notes": stage.notes,
            "train": slice_metrics("train", train),
            "test": slice_metrics("test", test),
            "test_2x_costs": slice_metrics("test_2x", test_2x),
            "folds": fold_rows,
            "by_symbol": by_sym,
            "by_asset_class": by_cls,
            "by_regime": by_reg,
            "by_trigger": by_key(test, lambda t: t.trigger),
            "by_exit_reason": by_key(test, lambda t: t.exit_reason),
            "robustness": checks,
        }

    candidate = select_candidate_on_train(stage_train)
    orig_checks = summaries["ORIGINAL"]["robustness"]
    cand_checks = summaries[candidate]["robustness"]
    code, label = verdict_from_checks(orig_checks, candidate, cand_checks, summaries)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Scanner V2 research/backtest only — live ORIGINAL unchanged",
        "timeframe": "4h",
        "train_fraction": train_frac,
        "n_folds": n_folds,
        "risk_fraction": 0.01,
        "cost_model": "ROUND_TRIP_COST × cost_mult, converted to R then to equity %",
        "architecture": {
            "regime": "HH/HL or LH/LL + SMA stack & slope + ADX≥20 + DI align + ATR%≥0.6×med50",
            "triggers": "pullback-to-SMA20 resume; optional 20-bar breakout continuation (S3)",
            "exits": "S1 fixed 4-bar; S2/S3 ATR1.5 stop + chandelier trail + structure-break + max 24 bars",
            "risk": "1% equity per trade using ATR stop distance as 1R (all stages incl. ORIGINAL)",
            "stages": {s.name: s.notes for s in stages},
        },
        "selection_rule": "Best V2 TRAIN expectancy with n≥20 (TEST untouched for selection)",
        "selected_candidate": candidate,
        "stages": summaries,
        "verdict_code": code,
        "verdict": label,
        "instruments": sorted({k for k, _ in series_map}),
        "series_bars": {f"{k}:{tf}": len(s) for (k, tf), s in series_map.items()},
    }


def format_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V2 — RESEARCH / BACKTEST ONLY (live ORIGINAL unchanged)")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Timeframe: {payload.get('timeframe')} | Train frac: {payload.get('train_fraction')}")
    a(f"Instruments: {', '.join(payload.get('instruments') or [])}")
    a("")
    a("ARCHITECTURE")
    a("-" * 72)
    arch = payload.get("architecture") or {}
    for k, v in arch.items():
        if k == "stages":
            a("  stages:")
            for sn, note in v.items():
                a(f"    - {sn}: {note}")
        else:
            a(f"  {k}: {v}")
    a("")
    a(f"Selection rule: {payload.get('selection_rule')}")
    a(f"Selected candidate (TRAIN only): {payload.get('selected_candidate')}")
    a("")

    for name, s in (payload.get("stages") or {}).items():
        a("=" * 72)
        a(f"STAGE: {name}")
        a(f"  {s.get('notes')}")
        a("-" * 72)
        for period in ("train", "test", "test_2x_costs"):
            m = s.get(period) or {}
            a(
                f"  {period:14s} n={m.get('signals', 0):4d}  "
                f"win={_pct(m.get('win_rate'))}  "
                f"avg={_bps(m.get('avg_return'))}  "
                f"exp={_bps(m.get('expectancy'))}  "
                f"PF={_pf(m.get('profit_factor'))}  "
                f"DD={_pct(m.get('max_drawdown'))}  "
                f"avgR={_num(m.get('avg_r'))}"
            )
            lng = m.get("long") or {}
            sh = m.get("short") or {}
            a(
                f"    long n={lng.get('signals', 0)} exp={_bps(lng.get('expectancy'))} | "
                f"short n={sh.get('signals', 0)} exp={_bps(sh.get('expectancy'))}"
            )
        a("  folds:")
        for fr in s.get("folds") or []:
            m = fr.get("metrics") or {}
            a(
                f"    fold {fr['fold']} [{fr['start_frac']:.2f}-{fr['end_frac']:.2f}] "
                f"n={m.get('signals', 0)} exp={_bps(m.get('expectancy'))} "
                f"avg={_bps(m.get('avg_return'))} DD={_pct(m.get('max_drawdown'))}"
            )
        a("  by asset class (TEST):")
        for cls, m in (s.get("by_asset_class") or {}).items():
            a(
                f"    {cls:10s} n={m.get('signals', 0):4d} "
                f"exp={_bps(m.get('expectancy'))} win={_pct(m.get('win_rate'))}"
            )
        a("  by symbol (TEST):")
        for sym, m in (s.get("by_symbol") or {}).items():
            a(
                f"    {sym:8s} n={m.get('signals', 0):4d} "
                f"exp={_bps(m.get('expectancy'))} win={_pct(m.get('win_rate'))}"
            )
        a("  by regime (TEST):")
        for reg, m in (s.get("by_regime") or {}).items():
            a(
                f"    {reg:10s} n={m.get('signals', 0):4d} exp={_bps(m.get('expectancy'))}"
            )
        rob = s.get("robustness") or {}
        a("  robustness checks:")
        for k, v in rob.items():
            if k == "all_pass":
                continue
            a(f"    {k}: {v}")
        a(f"  ALL_PASS: {rob.get('all_pass')}")
        a("")

    a("=" * 72)
    a(f"VERDICT: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    a("=" * 72)
    a("Live scanner: NOT modified. ORIGINAL remains the control/benchmark.")
    return "\n".join(lines) + "\n"


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:6.2f}%"


def _bps(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{10000.0 * x:+7.2f}bp"


def _num(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.3f}"


def _pf(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:5.2f}"


def write_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
