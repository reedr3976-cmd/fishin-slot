"""Reporting for Scanner V3 research (analysis only)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    MIN_SIGNALS_FOR_CONCLUSION,
    V3_MAX_DD_ACCEPT,
    V3_MIN_FOLDS_POSITIVE,
    V3_MIN_SYMBOLS_POSITIVE,
    V3_MIN_TRADES,
    V3_N_FOLDS,
    V3_TRAIN_FRACTION,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v3 import (
    ENTRY_STAGES,
    EXIT_STAGES,
    V3Stage,
    V3_S2_STRUCT_MA,
    build_asset_class_stages,
    chronological_folds_v3,
    collect_v3_stage,
    run_v3_window,
)


def _exp(trades: list[V2Trade]) -> Optional[float]:
    if not trades:
        return None
    return float(np.mean([t.net_return for t in trades]))


def _total_return(trades: list[V2Trade]) -> float:
    eq = 1.0
    for t in trades:
        eq *= 1.0 + t.net_return
    return eq - 1.0


def slice_metrics(label: str, trades: list[V2Trade]) -> dict[str, Any]:
    bag = summarize_trades(label, trades)
    d = bag.to_dict()
    d["expectancy"] = _exp(trades)
    d["total_return"] = _total_return(trades) if trades else 0.0
    d["avg_r"] = float(np.mean([t.net_r for t in trades])) if trades else None
    longs = [t for t in trades if t.direction == "bullish"]
    shorts = [t for t in trades if t.direction == "bearish"]
    d["long_n"] = len(longs)
    d["short_n"] = len(shorts)
    d["long_expectancy"] = _exp(longs)
    d["short_expectancy"] = _exp(shorts)
    return d


def by_key(trades: list[V2Trade], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        groups[str(key_fn(t))].append(t)
    return {k: slice_metrics(k, v) for k, v in sorted(groups.items())}


def oos_gate(
    *,
    test: list[V2Trade],
    test_2x: list[V2Trade],
    fold_exps: list[float],
    by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    test_m = slice_metrics("test", test)
    exp = test_m.get("expectancy")
    exp2 = _exp(test_2x)
    n = len(test)
    pos_folds = sum(1 for e in fold_exps if e > 0)
    sym_pos = [
        s
        for s, m in by_symbol.items()
        if (m.get("expectancy") or 0) > 0 and (m.get("signals") or 0) >= 5
    ]
    checks = {
        "positive_oos_expectancy": bool(exp is not None and exp > 0),
        "folds_positive_ge_3": pos_folds >= V3_MIN_FOLDS_POSITIVE,
        "positive_after_2x_costs": bool(exp2 is not None and exp2 > 0),
        "adequate_trade_count": n >= V3_MIN_TRADES,
        "acceptable_max_drawdown": bool(
            test_m.get("max_drawdown") is not None
            and test_m["max_drawdown"] <= V3_MAX_DD_ACCEPT
        ),
        "not_single_symbol": len(sym_pos) >= V3_MIN_SYMBOLS_POSITIVE,
    }
    checks["folds_positive_count"] = pos_folds
    checks["symbols_positive_ge5"] = sym_pos
    checks["test_expectancy"] = exp
    checks["test_2x_expectancy"] = exp2
    checks["test_n"] = n
    checks["test_max_dd"] = test_m.get("max_drawdown")
    checks["test_total_return"] = test_m.get("total_return")
    checks["all_pass"] = all(checks[k] for k in (
        "positive_oos_expectancy",
        "folds_positive_ge_3",
        "positive_after_2x_costs",
        "adequate_trade_count",
        "acceptable_max_drawdown",
        "not_single_symbol",
    ))
    return checks


def evaluate_stage(
    series_map: dict,
    stage: V3Stage,
    *,
    train_frac: float = V3_TRAIN_FRACTION,
    n_folds: int = V3_N_FOLDS,
) -> dict[str, Any]:
    print(f"  collecting {stage.name}...", flush=True)
    cached = collect_v3_stage(series_map, stage, cost_mult=1.0)
    train = run_v3_window(
        series_map, stage, start_frac=0.0, end_frac=train_frac, cost_mult=1.0, cached=cached
    )
    test = run_v3_window(
        series_map, stage, start_frac=train_frac, end_frac=1.0, cost_mult=1.0, cached=cached
    )
    test_2x = run_v3_window(
        series_map, stage, start_frac=train_frac, end_frac=1.0, cost_mult=2.0, cached=cached
    )
    folds = chronological_folds_v3(
        series_map, stage, n_folds=n_folds, cost_mult=1.0, cached=cached
    )
    fold_rows = []
    fold_exps: list[float] = []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append(
            {
                "fold": fr["fold"],
                "start_frac": fr["start_frac"],
                "end_frac": fr["end_frac"],
                "metrics": m,
            }
        )
    by_sym = by_key(test, lambda t: t.instrument)
    by_cls = by_key(test, lambda t: t.asset_class)
    gate = oos_gate(test=test, test_2x=test_2x, fold_exps=fold_exps, by_symbol=by_sym)
    return {
        "notes": stage.notes,
        "kind": stage.kind,
        "train": slice_metrics("train", train),
        "test": slice_metrics("test", test),
        "test_2x_costs": slice_metrics("test_2x", test_2x),
        "folds": fold_rows,
        "by_symbol": by_sym,
        "by_asset_class": by_cls,
        "by_exit_reason": by_key(test, lambda t: t.exit_reason),
        "oos_gate": gate,
    }


def _beats(a: Optional[float], b: Optional[float]) -> bool:
    if a is None:
        return False
    if b is None:
        return a > 0
    return a > b


def justify_asset_class(entry_results: dict[str, dict]) -> tuple[bool, str, Optional[str]]:
    """Stage 4 only if a V3 entry candidate clearly beats benchmarks on OOS."""
    orig = entry_results.get("ORIGINAL", {}).get("oos_gate", {})
    v2 = entry_results.get("V2_S3_DUAL_TRIG", {}).get("oos_gate", {})
    candidates = []
    for name, res in entry_results.items():
        if not name.startswith("V3_S"):
            continue
        g = res.get("oos_gate", {})
        if not g.get("positive_oos_expectancy"):
            continue
        # Prefer candidates that beat both benchmarks on OOS expectancy
        if _beats(g.get("test_expectancy"), orig.get("test_expectancy")) and _beats(
            g.get("test_expectancy"), v2.get("test_expectancy")
        ):
            candidates.append((name, g.get("test_expectancy") or 0.0, g.get("all_pass")))
    if not candidates:
        return False, "No V3 entry stage with positive OOS expectancy beating ORIGINAL and V2_S3.", None
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    best = candidates[0][0]
    # Prefer simplified regime without ADX if both positive
    prefer = "V3_S2_STRUCT_MA" if any(c[0] == "V3_S2_STRUCT_MA" for c in candidates) else best
    return True, f"Justified by OOS-positive {prefer} beating ORIGINAL & V2_S3.", prefer


def verdict_v3(
    entry_results: dict[str, dict],
    exit_results: dict[str, dict],
    s4_results: dict[str, dict],
) -> tuple[str, str, Optional[str]]:
    """OOS-first verdict. TRAIN is never the promotion criterion."""
    all_res = {**entry_results, **exit_results, **s4_results}
    passers = [n for n, r in all_res.items() if r.get("oos_gate", {}).get("all_pass")]
    # Prefer V3 names
    v3_pass = [n for n in passers if n.startswith("V3_")]
    if v3_pass:
        # Rank by OOS expectancy
        v3_pass.sort(
            key=lambda n: all_res[n]["oos_gate"].get("test_expectancy") or 0.0,
            reverse=True,
        )
        best = v3_pass[0]
        return (
            "PASS",
            f"PASS — {best} is strong enough for further paper-trading validation "
            "(positive OOS expectancy, folds, 2× costs, n, DD, multi-symbol). "
            "Do NOT merge into live scanner automatically.",
            best,
        )

    # Near-miss diagnostics
    near = []
    for n, r in all_res.items():
        if not n.startswith("V3_"):
            continue
        g = r.get("oos_gate", {})
        hard = [
            "positive_oos_expectancy",
            "folds_positive_ge_3",
            "positive_after_2x_costs",
            "adequate_trade_count",
            "acceptable_max_drawdown",
            "not_single_symbol",
        ]
        score = sum(1 for k in hard if g.get(k))
        if g.get("positive_oos_expectancy") and score >= 3:
            near.append((n, score, g.get("test_expectancy")))
    if near:
        near.sort(key=lambda x: (x[1], x[2] or 0), reverse=True)
        return (
            "FAIL",
            f"FAIL — redesign again. Nearest miss: {near[0][0]} "
            f"(passed {near[0][1]}/6 OOS gates). Live ORIGINAL untouched.",
            near[0][0],
        )
    return (
        "FAIL",
        "FAIL — redesign again. No V3 candidate achieved positive OOS expectancy "
        "with fold/cost/DD robustness. Live ORIGINAL untouched.",
        None,
    )


def build_v3_payload(series_map: dict) -> dict[str, Any]:
    entry_results: dict[str, dict] = {}
    for stage in ENTRY_STAGES:
        entry_results[stage.name] = evaluate_stage(series_map, stage)

    exit_results: dict[str, dict] = {}
    for stage in EXIT_STAGES:
        exit_results[stage.name] = evaluate_stage(series_map, stage)

    # Diagnose entry vs exit weakness
    orig_exp = entry_results["ORIGINAL"]["oos_gate"].get("test_expectancy")
    best_exit = max(
        exit_results.items(),
        key=lambda kv: kv[1]["oos_gate"].get("test_expectancy") or float("-inf"),
    )
    best_entry_v3 = max(
        ((n, r) for n, r in entry_results.items() if n.startswith("V3_S")),
        key=lambda kv: kv[1]["oos_gate"].get("test_expectancy") or float("-inf"),
        default=(None, None),
    )
    diagnosis = {
        "original_oos_expectancy": orig_exp,
        "best_exit_only": {
            "name": best_exit[0],
            "oos_expectancy": best_exit[1]["oos_gate"].get("test_expectancy"),
            "all_pass": best_exit[1]["oos_gate"].get("all_pass"),
        },
        "best_v3_entry": {
            "name": best_entry_v3[0],
            "oos_expectancy": (best_entry_v3[1] or {}).get("oos_gate", {}).get("test_expectancy")
            if best_entry_v3[1]
            else None,
            "all_pass": (best_entry_v3[1] or {}).get("oos_gate", {}).get("all_pass")
            if best_entry_v3[1]
            else False,
        },
        "weakness": None,
    }
    be = diagnosis["best_exit_only"]["oos_expectancy"]
    bv = diagnosis["best_v3_entry"]["oos_expectancy"]
    if be is not None and orig_exp is not None and be > orig_exp and (bv is None or be >= (bv or -1)):
        diagnosis["weakness"] = (
            "Exit logic is a material weakness of ORIGINAL: improved exits lift OOS "
            "more than (or as much as) V3 entry redesigns on this sample."
        )
    elif bv is not None and orig_exp is not None and bv > orig_exp:
        diagnosis["weakness"] = (
            "Entry logic is the larger weakness: V3 breakout/regime redesign beats "
            "ORIGINAL OOS more than exit-only patches."
        )
    else:
        diagnosis["weakness"] = (
            "Neither exit-only patches nor V3 entry redesigns restore robust OOS edge "
            "versus costs/folds on this sample."
        )

    s4_results: dict[str, dict] = {}
    justified, reason, base_name = justify_asset_class(entry_results)
    s4_meta = {"justified": justified, "reason": reason, "base": base_name}
    if justified and base_name:
        base = next((s for s in ENTRY_STAGES if s.name == base_name), V3_S2_STRUCT_MA)
        for stage in build_asset_class_stages(base):
            s4_results[stage.name] = evaluate_stage(series_map, stage)
    else:
        s4_meta["skipped"] = True

    code, label, promoted = verdict_v3(entry_results, exit_results, s4_results)

    # Comparisons vs ORIGINAL and V2_S3
    comparisons = {}
    for name, res in {**entry_results, **exit_results, **s4_results}.items():
        if name in ("ORIGINAL", "V2_S3_DUAL_TRIG"):
            continue
        comparisons[name] = {
            "oos_vs_ORIGINAL_bp": _bp_delta(
                res["oos_gate"].get("test_expectancy"),
                entry_results["ORIGINAL"]["oos_gate"].get("test_expectancy"),
            ),
            "oos_vs_V2_S3_bp": _bp_delta(
                res["oos_gate"].get("test_expectancy"),
                entry_results["V2_S3_DUAL_TRIG"]["oos_gate"].get("test_expectancy"),
            ),
            "oos_gate_pass": res["oos_gate"].get("all_pass"),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Scanner V3 research only — live ORIGINAL untouched; V2 not merged live",
        "timeframe": "4h",
        "universe": "forex + stocks + commodities",
        "train_fraction": V3_TRAIN_FRACTION,
        "n_folds": V3_N_FOLDS,
        "risk_fraction": 0.01,
        "selection_rule": "OOS gates only (TRAIN diagnostic; never promote on TRAIN alone)",
        "architecture": {
            "stage1": "Breakout-first (20-bar high/low), no heavy indicator gate",
            "stage2": "Simplified regime: structure HH/HL·LH/LL + MA stack/slope; ADX probe only",
            "stage3": "Exit-only on frozen ORIGINAL MH entries",
            "stage4": "Asset-class regimes only if Stage1/2 OOS-justified",
        },
        "entry_stages": entry_results,
        "exit_only_stages": exit_results,
        "asset_class_stage4": s4_results,
        "stage4_meta": s4_meta,
        "diagnosis_entry_vs_exit": diagnosis,
        "comparisons_vs_benchmarks": comparisons,
        "verdict_code": code,
        "verdict": label,
        "promoted_candidate": promoted,
        "instruments": sorted({k for k, _ in series_map}),
        "series_bars": {f"{k}:{tf}": len(s) for (k, tf), s in series_map.items()},
    }


def _bp_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (a - b) * 10000.0


def format_v3_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V3 — RESEARCH / BACKTEST ONLY")
    a("Live ORIGINAL untouched | V2 not merged into live scanner")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"TF={payload.get('timeframe')}  universe={payload.get('universe')}")
    a(f"Instruments: {', '.join(payload.get('instruments') or [])}")
    a(f"Selection: {payload.get('selection_rule')}")
    a("")
    arch = payload.get("architecture") or {}
    a("ARCHITECTURE")
    for k, v in arch.items():
        a(f"  {k}: {v}")
    a("")

    def _dump_stage(name: str, s: dict) -> None:
        a("=" * 72)
        a(f"STAGE: {name}")
        a(f"  {s.get('notes')}")
        a("-" * 72)
        for period in ("train", "test", "test_2x_costs"):
            m = s.get(period) or {}
            a(
                f"  {period:14s} n={m.get('signals', 0):4d}  "
                f"win={_pct(m.get('win_rate'))}  "
                f"avgW={_bps(m.get('avg_winner'))}  avgL={_bps(m.get('avg_loser'))}  "
                f"exp={_bps(m.get('expectancy'))}  PF={_pf(m.get('profit_factor'))}  "
                f"DD={_pct(m.get('max_drawdown'))}  tot={_pct(m.get('total_return'))}"
            )
        a("  folds:")
        for fr in s.get("folds") or []:
            m = fr.get("metrics") or {}
            a(
                f"    fold {fr['fold']} [{fr['start_frac']:.2f}-{fr['end_frac']:.2f}] "
                f"n={m.get('signals', 0)} exp={_bps(m.get('expectancy'))} "
                f"DD={_pct(m.get('max_drawdown'))} tot={_pct(m.get('total_return'))}"
            )
        a("  by asset class (TEST):")
        for cls, m in (s.get("by_asset_class") or {}).items():
            a(
                f"    {cls:10s} n={m.get('signals', 0):4d} "
                f"exp={_bps(m.get('expectancy'))} win={_pct(m.get('win_rate'))}"
            )
        g = s.get("oos_gate") or {}
        a("  OOS gate:")
        for k, v in g.items():
            a(f"    {k}: {v}")
        a(f"  ALL_PASS: {g.get('all_pass')}")
        a("")

    a("# ENTRY STAGES (incl. ORIGINAL & V2_S3 benchmarks)")
    for name, s in (payload.get("entry_stages") or {}).items():
        _dump_stage(name, s)

    a("# EXIT-ONLY STAGES (ORIGINAL entries)")
    for name, s in (payload.get("exit_only_stages") or {}).items():
        _dump_stage(name, s)

    s4 = payload.get("asset_class_stage4") or {}
    meta = payload.get("stage4_meta") or {}
    a("# STAGE 4 ASSET-CLASS")
    a(f"  justified={meta.get('justified')}  reason={meta.get('reason')}")
    if not s4:
        a("  (skipped)")
        a("")
    for name, s in s4.items():
        _dump_stage(name, s)

    a("# DIAGNOSIS: ENTRY vs EXIT")
    diag = payload.get("diagnosis_entry_vs_exit") or {}
    for k, v in diag.items():
        a(f"  {k}: {v}")
    a("")
    a("# COMPARISONS vs ORIGINAL / V2_S3 (OOS expectancy delta, bp)")
    for name, c in (payload.get("comparisons_vs_benchmarks") or {}).items():
        a(
            f"  {name:28s} vsORIG={_num(c.get('oos_vs_ORIGINAL_bp'))}bp  "
            f"vsV2S3={_num(c.get('oos_vs_V2_S3_bp'))}bp  pass={c.get('oos_gate_pass')}"
        )
    a("")
    a("=" * 72)
    a(f"VERDICT: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    a(f"PROMOTED (paper only if PASS): {payload.get('promoted_candidate')}")
    a("=" * 72)
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
    return f"{x:+.2f}"


def _pf(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "  inf"
    return f"{x:5.2f}"


def write_v3_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v3_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
