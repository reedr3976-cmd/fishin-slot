"""V7 robustness research reporting (research only).

Investigates V6 near-misses, evaluates new V7 families under frozen TRAIN
selection, and compares against V6/V5. Does not loosen PASS gates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import (
    V7_ATR_STOP_MULT,
    V7_COMMODITY_DISCOVERY,
    V7_COMMODITY_HELDOUT,
    V7_ENTRY_SLIP_ATR,
    V7_FX_DISCOVERY,
    V7_FX_HELDOUT,
    V7_MAX_DD_ACCEPT,
    V7_MAX_HOLD,
    V7_MC_RUNS,
    V7_MC_SEED,
    V7_MIN_FOLDS_POSITIVE,
    V7_MIN_HELDOUT_TRADES,
    V7_MIN_SYMBOLS_POSITIVE,
    V7_MIN_TRADES,
    V7_N_FOLDS,
    V7_STOCK_DISCOVERY,
    V7_STOCK_HELDOUT,
    V7_TRAIN_FRACTION,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import leave_out_symbols, monte_carlo
from backtest.scanner_v7 import FAMILIES, V7Family, folds_for_family, run_family_on_map


def _exp(trades: list[V2Trade]) -> Optional[float]:
    if not trades:
        return None
    return float(np.mean([t.net_return for t in trades]))


def _tot(trades: list[V2Trade]) -> float:
    eq = 1.0
    for t in trades:
        eq *= 1.0 + t.net_return
    return eq - 1.0


def slice_metrics(label: str, trades: list[V2Trade]) -> dict[str, Any]:
    bag = summarize_trades(label, trades)
    d = bag.to_dict()
    d["expectancy"] = _exp(trades)
    d["total_return"] = _tot(trades) if trades else 0.0
    return d


def by_symbol(trades: list[V2Trade]) -> dict[str, Any]:
    groups: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        groups[t.instrument].append(t)
    per = {s: slice_metrics(s, ts) for s, ts in sorted(groups.items())}
    contrib = {s: float(sum(t.net_return for t in ts)) for s, ts in groups.items()}
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    pos = sum(v for _, v in ranked if v > 0)

    def share(n: int) -> Optional[float]:
        if pos <= 0:
            return None
        return float(sum(v for _, v in ranked[:n] if v > 0) / pos)

    return {
        "per_symbol": per,
        "pnl_contribution": {k: float(v) for k, v in ranked},
        "best1_share": share(1),
        "best3_share": share(3),
        "best_symbols": [k for k, _ in ranked[:5]],
        "symbols_positive_ge3": [
            s
            for s, mm in per.items()
            if (mm.get("expectancy") or 0) > 0 and (mm.get("signals") or 0) >= 3
        ],
    }


def period_concentration(fold_rows: list[dict], test_exp: Optional[float]) -> dict[str, Any]:
    """Detect fold lottery: only the last fold positive while TEST looks strong."""
    fold_exps = [(fr["fold"], (fr.get("metrics") or {}).get("expectancy") or 0.0) for fr in fold_rows]
    pos = [f for f, e in fold_exps if e > 0]
    only_last = pos == [V7_N_FOLDS]
    return {
        "positive_folds": pos,
        "only_final_fold_positive": only_last,
        "test_positive_while_only_final_fold": bool(
            only_last and test_exp is not None and test_exp > 0
        ),
        "fold_expectancies": {str(f): e for f, e in fold_exps},
    }


def gate_from(
    test: list[V2Trade],
    test_2x: list[V2Trade],
    fold_exps: list[float],
    held_out_test: list[V2Trade],
    sens_positive: int,
    sens_total: int,
) -> dict[str, Any]:
    m = slice_metrics("test", test)
    exp = m.get("expectancy")
    exp2 = _exp(test_2x)
    held_exp = _exp(held_out_test)
    sym = by_symbol(test)
    pos_folds = sum(1 for e in fold_exps if e > 0)
    sym_pos = sym["symbols_positive_ge3"]
    resilient_2x = bool(
        (exp2 is not None and exp2 > 0)
        or (exp is not None and exp > 0 and exp2 is not None and exp2 >= 0.5 * exp)
    )
    checks = {
        "positive_discovery_oos": bool(exp is not None and exp > 0),
        "positive_heldout_oos": bool(held_exp is not None and held_exp > 0),
        "resilient_2x": resilient_2x,
        "folds_ge_3": pos_folds >= V7_MIN_FOLDS_POSITIVE,
        "folds_positive_count": pos_folds,
        "adequate_n": len(test) >= V7_MIN_TRADES,
        "adequate_heldout_n": len(held_out_test) >= V7_MIN_HELDOUT_TRADES,
        "acceptable_dd": bool((m.get("max_drawdown") or 1) <= V7_MAX_DD_ACCEPT),
        "multi_symbol": len(sym_pos) >= min(V7_MIN_SYMBOLS_POSITIVE, max(1, len(sym["per_symbol"]))),
        "not_one_symbol": not ((sym.get("best1_share") or 0) >= 0.70),
        "not_fragile": sens_positive >= max(1, sens_total // 2),
        "heldout_n": len(held_out_test),
        "heldout_expectancy": held_exp,
        "test_expectancy": exp,
        "test_2x_expectancy": exp2,
        "symbols_positive": sym_pos,
        "best1_share": sym.get("best1_share"),
    }
    checks["all_pass"] = all(
        checks[k]
        for k in (
            "positive_discovery_oos",
            "positive_heldout_oos",
            "resilient_2x",
            "folds_ge_3",
            "adequate_n",
            "adequate_heldout_n",
            "acceptable_dd",
            "multi_symbol",
            "not_one_symbol",
            "not_fragile",
        )
    )
    return checks


def decision_from_gate(gate: dict[str, Any], *, test_n: int) -> dict[str, Any]:
    reasons_fail: list[str] = []
    label_map = {
        "positive_discovery_oos": "discovery OOS expectancy ≤ 0",
        "positive_heldout_oos": "held-out OOS expectancy ≤ 0",
        "resilient_2x": "fails 2× cost resilience",
        "folds_ge_3": f"positive folds {gate.get('folds_positive_count')}/4 (<3)",
        "adequate_n": f"inadequate trade count n={test_n} (need ≥{V7_MIN_TRADES})",
        "adequate_heldout_n": (
            f"inadequate held-out trade count n={gate.get('heldout_n')} "
            f"(need ≥{V7_MIN_HELDOUT_TRADES})"
        ),
        "acceptable_dd": "maximum drawdown above acceptance threshold",
        "multi_symbol": "edge not distributed across enough symbols",
        "not_one_symbol": "profit concentrated in one symbol (≥70% of positive PnL)",
        "not_fragile": "fragile to nearby parameter changes",
    }
    for k, msg in label_map.items():
        if not gate.get(k):
            reasons_fail.append(msg)
    if test_n < V7_MIN_TRADES:
        reasons_fail.append("reject: too few OOS trades (overfit / noise risk)")
    passed = bool(gate.get("all_pass"))
    return {
        "decision": "PASS" if passed else "FAIL",
        "reason": (
            "Meets discovery OOS, held-out OOS (with adequate n), folds, cost stress, "
            "sensitivity, and multi-symbol robustness gates."
            if passed
            else "; ".join(reasons_fail) if reasons_fail else "failed robustness gates"
        ),
    }


def evaluate_family(
    series_4h: dict,
    daily_map: dict,
    family: V7Family,
    discovery: tuple[str, ...],
    heldout: tuple[str, ...],
) -> dict[str, Any]:
    print(f"  family {family.name} discovery={discovery}...", flush=True)
    train = run_family_on_map(
        series_4h, family, discovery, daily_map=daily_map, start_frac=0.0, end_frac=V7_TRAIN_FRACTION
    )
    test = run_family_on_map(
        series_4h, family, discovery, daily_map=daily_map, start_frac=V7_TRAIN_FRACTION, end_frac=1.0
    )
    test_15 = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V7_TRAIN_FRACTION,
        end_frac=1.0,
        cost_mult=1.5,
    )
    test_2x = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V7_TRAIN_FRACTION,
        end_frac=1.0,
        cost_mult=2.0,
    )
    test_slip = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V7_TRAIN_FRACTION,
        end_frac=1.0,
        entry_slip_atr=V7_ENTRY_SLIP_ATR,
    )
    held_test = run_family_on_map(
        series_4h, family, heldout, daily_map=daily_map, start_frac=V7_TRAIN_FRACTION, end_frac=1.0
    )
    held_full = run_family_on_map(
        series_4h, family, heldout, daily_map=daily_map, start_frac=0.0, end_frac=1.0
    )
    folds = folds_for_family(
        series_4h, family, discovery, daily_map=daily_map, n_folds=V7_N_FOLDS
    )
    fold_rows = []
    fold_exps = []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append(
            {
                "fold": fr["fold"],
                "metrics": m,
                "start_frac": fr["start_frac"],
                "end_frac": fr["end_frac"],
            }
        )

    sens = []
    for ov in (
        {"atr_stop_mult": 1.25},
        {"atr_stop_mult": 1.75},
        {"max_hold": 20},
        {"max_hold": 28},
    ):
        t = run_family_on_map(
            series_4h,
            family,
            discovery,
            daily_map=daily_map,
            start_frac=V7_TRAIN_FRACTION,
            end_frac=1.0,
            **ov,
        )
        e = _exp(t)
        sens.append(
            {
                "override": ov,
                "n": len(t),
                "expectancy": e,
                "positive": bool(e is not None and e > 0),
            }
        )
    sens_pos = sum(1 for s in sens if s["positive"])

    train_sym = by_symbol(train)
    test_sym = by_symbol(test)
    held_sym = by_symbol(held_test)
    best = test_sym["best_symbols"]
    drop1 = leave_out_symbols(test, {best[0]} if best else set())
    g = gate_from(test, test_2x, fold_exps, held_test, sens_pos, len(sens))
    decision = decision_from_gate(g, test_n=len(test))
    period = period_concentration(fold_rows, _exp(test))

    frozen_rules = {
        "family": family.name,
        "key": family.key,
        "notes": family.notes,
        "addresses": family.addresses,
        "timeframe": "4h",
        "risk": "1% equity via ATR stop = 1R",
        "atr_stop_mult": V7_ATR_STOP_MULT,
        "max_hold": V7_MAX_HOLD,
        "exit": "adaptive ATR trail (V2 adaptive)",
        "selection": "TRAIN-only; configuration frozen before OOS/held-out",
    }

    return {
        "family_key": family.key,
        "name": family.name,
        "notes": family.notes,
        "addresses": family.addresses,
        "frozen_rules": frozen_rules,
        "discovery": list(discovery),
        "heldout": list(heldout),
        "train": slice_metrics("train", train),
        "train_n": len(train),
        "train_expectancy": _exp(train),
        "train_by_symbol": train_sym,
        "train_diversified": len(train_sym["symbols_positive_ge3"]) >= V7_MIN_SYMBOLS_POSITIVE
        and not ((train_sym.get("best1_share") or 0) >= 0.70),
        "test": slice_metrics("test", test),
        "test_1_5x": slice_metrics("1.5x", test_15),
        "test_2x": slice_metrics("2x", test_2x),
        "test_slip": slice_metrics("slip", test_slip),
        "heldout_test": slice_metrics("held_test", held_test),
        "heldout_full": slice_metrics("held_full", held_full),
        "heldout_by_symbol": held_sym,
        "folds": fold_rows,
        "period_concentration": period,
        "by_symbol": test_sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "sensitivity": sens,
        "monte_carlo": monte_carlo(test, n_runs=V7_MC_RUNS, seed=V7_MC_SEED),
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
    }


ASSET_CLASSES = (
    {
        "name": "stocks",
        "discovery": V7_STOCK_DISCOVERY,
        "heldout": V7_STOCK_HELDOUT,
    },
    {
        "name": "commodities",
        "discovery": V7_COMMODITY_DISCOVERY,
        "heldout": V7_COMMODITY_HELDOUT,
    },
    {
        "name": "forex",
        "discovery": V7_FX_DISCOVERY,
        "heldout": V7_FX_HELDOUT,
    },
)


def select_on_train(results: list[dict]) -> Optional[dict]:
    """TRAIN-only selection. Prefer diversified TRAIN; never uses TEST/held-out.

    Preference order:
      1) train_n >= 20, train_expectancy > 0, train_diversified
      2) train_n >= 20, max train_expectancy
      3) any with trades, max train_expectancy
    """
    preferred = [
        r
        for r in results
        if (r.get("train_n") or 0) >= 20
        and (r.get("train_expectancy") or 0) > 0
        and r.get("train_diversified")
    ]
    if preferred:
        return max(preferred, key=lambda r: r.get("train_expectancy") or float("-inf"))
    elig = [r for r in results if (r.get("train_n") or 0) >= 20]
    if not elig:
        elig = [r for r in results if (r.get("train_n") or 0) > 0]
    if not elig:
        return None
    return max(elig, key=lambda r: r.get("train_expectancy") or float("-inf"))


def diagnose_v6_near_misses() -> dict[str, Any]:
    """Explain why V6 Stocks C and Commodities C failed (from frozen V6 report)."""
    path = Path(__file__).resolve().parent.parent / "output" / "scanner_v6_report.json"
    if not path.exists():
        return {
            "available": False,
            "note": "V6 report not found; diagnosis limited to design rationale.",
            "stocks_c": {
                "why_failed": (
                    "V6 Stocks C VOL_EXPANSION: strong discovery TEST but TRAIN negative, "
                    "n=20 < min trades, only 1/4 folds positive (final fold / OOS window), "
                    "held-out stocks negative — classic thin-sample period lottery, not a "
                    "generalizable edge."
                )
            },
            "commodities_c": {
                "why_failed": (
                    "V6 Commodities C VOL_EXPANSION: discovery TEST profits 100% from gold "
                    "(silver negative); held-out oil n=9 too small despite flashy expectancy — "
                    "insufficient symbol diversity to claim cross-commodity vol-expansion edge."
                )
            },
        }
    v6 = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"available": True, "v6_verdict": v6.get("verdict")}

    def pack(cls: str, fam_key: str) -> dict[str, Any]:
        fams = (v6.get("by_asset_class") or {}).get(cls, {}).get("families") or {}
        row = None
        for _name, r in fams.items():
            if r.get("family_key") == fam_key:
                row = r
                break
        if row is None:
            # Exact name fallback only (avoid substring traps like "C" in "CONFIRM")
            target = f"V6_{fam_key}_"
            for name, r in fams.items():
                if name.startswith(target) or name == f"V6_{fam_key}":
                    row = r
                    break
        if not row:
            return {"found": False}
        te = row.get("test") or {}
        tr = row.get("train") or {}
        ho = row.get("heldout_test") or {}
        gate = row.get("gate") or {}
        sym = row.get("by_symbol") or {}
        folds = [
            {
                "fold": fr.get("fold"),
                "n": (fr.get("metrics") or {}).get("signals"),
                "exp": (fr.get("metrics") or {}).get("expectancy"),
            }
            for fr in (row.get("folds") or [])
        ]
        return {
            "found": True,
            "name": row.get("name"),
            "decision": row.get("decision"),
            "decision_reason": row.get("decision_reason"),
            "train_n": tr.get("signals"),
            "train_expectancy": tr.get("expectancy"),
            "test_n": te.get("signals"),
            "test_expectancy": te.get("expectancy"),
            "test_pf": te.get("profit_factor"),
            "test_win_rate": te.get("win_rate"),
            "heldout_n": ho.get("signals"),
            "heldout_expectancy": ho.get("expectancy"),
            "folds": folds,
            "folds_positive": gate.get("folds_positive_count"),
            "best1_share": sym.get("best1_share"),
            "pnl_contribution": sym.get("pnl_contribution"),
            "gate_failures": [k for k, v in gate.items() if k != "all_pass" and v is False],
        }

    stocks_c = pack("stocks", "C")
    commodities_c = pack("commodities", "C")

    stocks_c["why_failed"] = (
        "Stocks C did not generalise because the discovery TEST edge was a thin-sample "
        f"artifact (test n={stocks_c.get('test_n')}, need ≥25): TRAIN expectancy was "
        f"{stocks_c.get('train_expectancy')} (not a strong in-sample foundation), only "
        f"{stocks_c.get('folds_positive')}/4 folds were positive and the sole positive fold "
        "aligns with the late/OOS window (period lottery), while held-out stocks were negative "
        f"(exp={stocks_c.get('heldout_expectancy')}, n={stocks_c.get('heldout_n')}). "
        "High PF/win-rate on ~20 OOS trades is insufficient evidence of a real edge."
    )
    commodities_c["why_failed"] = (
        "Commodities C failed diversification, not headline expectancy: discovery TEST "
        f"best1_share={commodities_c.get('best1_share')} with PnL "
        f"{commodities_c.get('pnl_contribution')} (single-symbol dominated positive "
        "contribution); held-out was oil-only with "
        f"n={commodities_c.get('heldout_n')} (exp={commodities_c.get('heldout_expectancy')}) "
        "— too small and single-instrument to prove vol-expansion works across commodities. "
        "V7 expands discovery to include oil and holds out non-oil sectors "
        "(gas/copper/corn) under research_only symbols."
    )
    out["stocks_c"] = stocks_c
    out["commodities_c"] = commodities_c
    out["design_response"] = {
        "do_not": "Retune V6-C parameters on the same OOS/held-out windows.",
        "instead": (
            "New V7 families add confirmation, pullback-after-expansion, retest, MTF, "
            "and regime+momentum structure; TRAIN selection prefers multi-symbol "
            "TRAIN diversification; held-out min trade count is a hard gate."
        ),
    }
    return out


def _v6_comparison() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "output" / "scanner_v6_report.json"
    if not path.exists():
        return {"available": False, "note": "V6 report not found."}
    v6 = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "v6_verdict": v6.get("verdict"),
        "v6_verdict_code": v6.get("verdict_code"),
        "v6_genuine_improvement_vs_v5": v6.get("genuine_improvement_vs_v5"),
        "v6_selected": {
            cls: (block or {}).get("train_selected_family")
            for cls, block in (v6.get("by_asset_class") or {}).items()
        },
        "note": (
            "Genuine V7 improvement requires a TRAIN-selected candidate that PASSes "
            "independent gates (including held-out n and multi-symbol). Better discovery "
            "TEST numbers alone are not an improvement over V6 FAIL."
        ),
    }


def _v5_comparison() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "output" / "scanner_v5_report.json"
    if not path.exists():
        return {"available": False, "note": "V5 report not found."}
    v5 = json.loads(path.read_text(encoding="utf-8"))
    held = (v5.get("held_out_stocks") or {}).get("test") or {}
    return {
        "available": True,
        "v5_verdict": v5.get("verdict"),
        "v5_heldout_stocks_oos_exp": held.get("expectancy"),
        "v5_heldout_stocks_n": held.get("signals"),
        "note": "V5 falsified V4_S1_STOCK on held-out stocks; V7 must not revive that claim without new gated evidence.",
    }


def build_final_summary(by_class: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cls, block in by_class.items():
        for name, r in (block.get("families") or {}).items():
            te = r.get("test") or {}
            ho = r.get("heldout_test") or {}
            t2 = r.get("test_2x") or {}
            mc = r.get("monte_carlo") or {}
            rows.append(
                {
                    "candidate": name,
                    "family_key": r.get("family_key"),
                    "market_class": cls,
                    "train_selected": name == block.get("train_selected_family"),
                    "frozen_rules": r.get("frozen_rules"),
                    "trades_oos": te.get("signals"),
                    "win_rate": te.get("win_rate"),
                    "profit_factor": te.get("profit_factor"),
                    "expectancy": te.get("expectancy"),
                    "max_drawdown": te.get("max_drawdown"),
                    "oos_expectancy": te.get("expectancy"),
                    "heldout_expectancy": ho.get("expectancy"),
                    "heldout_trades": ho.get("signals"),
                    "stress_2x_expectancy": t2.get("expectancy"),
                    "stress_slip_expectancy": (r.get("test_slip") or {}).get("expectancy"),
                    "parameter_sensitivity": r.get("sensitivity"),
                    "monte_carlo_dd_median": ((mc.get("max_drawdown") or {}).get("median")),
                    "monte_carlo_dd_p95": ((mc.get("max_drawdown") or {}).get("p95")),
                    "monte_carlo_total_return_median": ((mc.get("total_return") or {}).get("median")),
                    "folds_positive": (r.get("gate") or {}).get("folds_positive_count"),
                    "symbol_diversification": {
                        "best1_share": (r.get("by_symbol") or {}).get("best1_share"),
                        "symbols_positive": (r.get("gate") or {}).get("symbols_positive"),
                        "pnl_contribution": (r.get("by_symbol") or {}).get("pnl_contribution"),
                    },
                    "period_concentration": r.get("period_concentration"),
                    "decision": r.get("decision"),
                    "reason": r.get("decision_reason"),
                }
            )
    return rows


def identify_near_misses(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Near-miss: FAIL but positive discovery OOS and at least 2 of {heldout+, folds≥2, n ok}."""
    near = []
    for row in summary:
        if row.get("decision") == "PASS":
            continue
        oos = row.get("oos_expectancy")
        if oos is None or oos <= 0:
            continue
        score = 0
        if (row.get("heldout_expectancy") or 0) > 0:
            score += 1
        if (row.get("folds_positive") or 0) >= 2:
            score += 1
        if (row.get("trades_oos") or 0) >= V7_MIN_TRADES:
            score += 1
        if (row.get("stress_2x_expectancy") or 0) > 0:
            score += 1
        if score >= 2:
            near.append(
                {
                    "candidate": row.get("candidate"),
                    "market_class": row.get("market_class"),
                    "train_selected": row.get("train_selected"),
                    "near_miss_score": score,
                    "what_blocked_pass": row.get("reason"),
                    "oos_expectancy": oos,
                    "heldout_expectancy": row.get("heldout_expectancy"),
                    "folds_positive": row.get("folds_positive"),
                    "trades_oos": row.get("trades_oos"),
                }
            )
    near.sort(key=lambda x: x.get("near_miss_score") or 0, reverse=True)
    return near


def build_v7_payload(series_4h: dict, daily_map: dict) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []
    diagnosis = diagnose_v6_near_misses()

    for ac in ASSET_CLASSES:
        print(f"ASSET CLASS: {ac['name']}", flush=True)
        fam_results = []
        for fam in FAMILIES:
            fam_results.append(
                evaluate_family(
                    series_4h,
                    daily_map,
                    fam,
                    ac["discovery"],
                    ac["heldout"],
                )
            )
        train_pick = select_on_train(fam_results)
        pick_name = train_pick["name"] if train_pick else None
        candidate = next((r for r in fam_results if r["name"] == pick_name), None)
        class_verdict = "FAIL"
        if candidate and candidate.get("decision") == "PASS":
            class_verdict = "PASS"
            promoted.append(
                {
                    "asset_class": ac["name"],
                    "family": candidate["name"],
                    "notes": candidate["notes"],
                    "gate": candidate["gate"],
                    "decision_reason": candidate.get("decision_reason"),
                }
            )
        other_pass = [
            r["name"] for r in fam_results if r.get("decision") == "PASS" and r["name"] != pick_name
        ]
        by_class[ac["name"]] = {
            "discovery_symbols": list(ac["discovery"]),
            "heldout_symbols": list(ac["heldout"]),
            "families": {r["name"]: r for r in fam_results},
            "train_selected_family": pick_name,
            "train_selection_rule": (
                "Prefer TRAIN n≥20, positive TRAIN expectancy, and TRAIN multi-symbol "
                "diversification (best1_share<70%); else max TRAIN expectancy with n≥20. "
                "OOS/held-out unused for selection."
            ),
            "selected_gate": (candidate or {}).get("gate"),
            "selected_decision": (candidate or {}).get("decision"),
            "selected_decision_reason": (candidate or {}).get("decision_reason"),
            "other_families_passing_gates": other_pass,
            "class_verdict": class_verdict,
        }

    summary = build_final_summary(by_class)
    near = identify_near_misses(summary)
    any_pass = any(r["decision"] == "PASS" for r in summary)
    if promoted:
        overall = (
            "V7 PASS — TRAIN-selected candidate(s) passed independent robustness gates "
            "(next validation stage only; NOT enabled for paper/live)"
        )
        code = "V7_PASS"
    elif any_pass:
        overall = (
            "V7 FAIL — some non-selected families meet gates, but TRAIN-frozen selection "
            "did not; do not cherry-pick OOS winners"
        )
        code = "V7_FAIL"
    else:
        overall = "V7 FAIL — no candidate has sufficient evidence to proceed"
        code = "V7_FAIL"

    v6cmp = _v6_comparison()
    v5cmp = _v5_comparison()
    genuine = bool(promoted)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "V7 robustness research after V6 FAIL; diagnose Stocks/Commodities C "
            "near-misses; new families; live ORIGINAL untouched"
        ),
        "live_scanner_protected": True,
        "auto_promote": False,
        "families_tested": [
            {
                "key": f.key,
                "name": f.name,
                "notes": f.notes,
                "addresses": f.addresses,
            }
            for f in FAMILIES
        ],
        "selection_rule": (
            "Per asset class: choose family on TRAIN only (prefer diversified positive "
            "TRAIN), freeze, then evaluate TEST/held-out/costs/sensitivity/MC"
        ),
        "gate_policy": {
            "note": "PASS criteria not loosened vs V6; held-out min trades added as hard gate",
            "min_trades": V7_MIN_TRADES,
            "min_heldout_trades": V7_MIN_HELDOUT_TRADES,
            "min_folds_positive": V7_MIN_FOLDS_POSITIVE,
            "min_symbols_positive": V7_MIN_SYMBOLS_POSITIVE,
            "max_dd": V7_MAX_DD_ACCEPT,
            "max_best1_share": 0.70,
        },
        "v6_near_miss_diagnosis": diagnosis,
        "by_asset_class": by_class,
        "final_candidate_summary": summary,
        "strongest_near_misses": near,
        "v6_comparison": v6cmp,
        "v5_comparison": v5cmp,
        "genuine_robustness_improvement": genuine,
        "improvement_note": (
            "Yes — at least one TRAIN-selected V7 candidate cleared independent gates "
            "that V6 could not; this is a robustness improvement (still not live)."
            if genuine
            else (
                "No — V7 does not provide a genuine robustness improvement over V6 FAIL. "
                "No TRAIN-selected candidate cleared held-out/OOS/cost/sensitivity/"
                "diversification gates."
            )
        ),
        "promoted_candidates": promoted,
        "verdict_code": code,
        "verdict": overall,
        "next_stage": (
            [
                {
                    "action": "Independent validation protocol before any paper trading",
                    "candidate": p,
                }
                for p in promoted
            ]
            if promoted
            else ["No candidate strong enough for next validation stage"]
        ),
        "stop_note": "Stop after this report; wait for human approval before any further changes.",
    }


def format_v7_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V7 — ROBUSTNESS RESEARCH (after V6 FAIL)")
    a("Live ORIGINAL untouched | no paper/live enablement | no auto-promote")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Selection: {payload.get('selection_rule')}")
    a(f"Gate policy: {payload.get('gate_policy')}")
    a("Families:")
    for f in payload.get("families_tested") or []:
        a(f"  {f['key']}: {f['name']} — {f['notes']}")
        a(f"      addresses: {f.get('addresses')}")
    a("")

    a("=" * 72)
    a("V6 NEAR-MISS DIAGNOSIS")
    a("=" * 72)
    diag = payload.get("v6_near_miss_diagnosis") or {}
    a(f"  available={diag.get('available')} v6_verdict={diag.get('v6_verdict')}")
    for key in ("stocks_c", "commodities_c"):
        block = diag.get(key) or {}
        a(f"  [{key}]")
        a(f"    why: {block.get('why_failed')}")
        a(
            f"    train_n={block.get('train_n')} train_exp={_bps(block.get('train_expectancy'))} "
            f"test_n={block.get('test_n')} test_exp={_bps(block.get('test_expectancy'))} "
            f"held_n={block.get('heldout_n')} held_exp={_bps(block.get('heldout_expectancy'))} "
            f"folds+={block.get('folds_positive')} best1={block.get('best1_share')}"
        )
        a(f"    decision={block.get('decision')} reason={block.get('decision_reason')}")
    a(f"  design_response: {diag.get('design_response')}")
    a("")

    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}")
        a(f"  discovery={block.get('discovery_symbols')}  heldout={block.get('heldout_symbols')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}")
        a(f"  class_verdict: {block.get('class_verdict')}")
        a(f"  selected_decision: {block.get('selected_decision')} — {block.get('selected_decision_reason')}")
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(f"  {name}  [{r.get('family_key')}] {r.get('notes')}")
            a(f"    addresses: {r.get('addresses')}")
            a(f"    frozen_rules: {r.get('frozen_rules')}")
            a(f"    train_diversified={r.get('train_diversified')}")
            for key in ("train", "test", "test_1_5x", "test_2x", "test_slip", "heldout_test"):
                m = r.get(key) or {}
                a(
                    f"    {key:12s} n={m.get('signals', 0):4d} win={_pct(m.get('win_rate'))} "
                    f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} "
                    f"DD={_pct(m.get('max_drawdown'))} tot={_pct(m.get('total_return'))}"
                )
            a("    folds:")
            for fr in r.get("folds") or []:
                m = fr.get("metrics") or {}
                a(f"      fold {fr['fold']} n={m.get('signals', 0)} exp={_bps(m.get('expectancy'))}")
            a(f"    period_concentration: {r.get('period_concentration')}")
            sym = r.get("by_symbol") or {}
            a(f"    best1_share={sym.get('best1_share')} best3={sym.get('best3_share')}")
            for s, m in (sym.get("per_symbol") or {}).items():
                a(f"      {s:8s} n={m.get('signals', 0):3d} exp={_bps(m.get('expectancy'))}")
            a(f"    heldout_by_symbol: {(r.get('heldout_by_symbol') or {}).get('pnl_contribution')}")
            a(f"    sensitivity: {r.get('sensitivity')}")
            a(f"    monte_carlo: {r.get('monte_carlo')}")
            a(f"    DECISION: {r.get('decision')} — {r.get('decision_reason')}")
            a(f"    GATE all_pass={((r.get('gate') or {}).get('all_pass'))} {r.get('gate')}")
        a("")

    a("=" * 72)
    a("FINAL CANDIDATE SUMMARY (all families × market classes)")
    a("=" * 72)
    for row in payload.get("final_candidate_summary") or []:
        a(
            f"  [{row.get('decision')}] {row.get('market_class'):11s} {row.get('candidate')} "
            f"{'(TRAIN-selected) ' if row.get('train_selected') else ''}"
            f"n={row.get('trades_oos')} win={_pct(row.get('win_rate'))} "
            f"PF={_pf(row.get('profit_factor'))} exp={_bps(row.get('expectancy'))} "
            f"DD={_pct(row.get('max_drawdown'))} "
            f"OOS={_bps(row.get('oos_expectancy'))} held={_bps(row.get('heldout_expectancy'))} "
            f"held_n={row.get('heldout_trades')} 2x={_bps(row.get('stress_2x_expectancy'))} "
            f"MC_dd_med={_pct(row.get('monte_carlo_dd_median'))} "
            f"folds+={row.get('folds_positive')}"
        )
        a(f"      reason: {row.get('reason')}")
        a(f"      diversification: {row.get('symbol_diversification')}")

    a("")
    a("# STRONGEST NEAR-MISSES")
    for nm in payload.get("strongest_near_misses") or []:
        a(f"  {nm}")
    if not payload.get("strongest_near_misses"):
        a("  (none)")

    a("")
    a("# V6 / V5 COMPARISON")
    a(f"  v6: {payload.get('v6_comparison')}")
    a(f"  v5: {payload.get('v5_comparison')}")
    a(f"  genuine_robustness_improvement={payload.get('genuine_robustness_improvement')}")
    a(f"  {payload.get('improvement_note')}")
    a("")
    a("=" * 72)
    a(f"FINAL DECISION: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    a(f"PROMOTED (to next validation stage only): {payload.get('promoted_candidates')}")
    a(f"NEXT: {payload.get('next_stage')}")
    a("Live scanner: NOT modified. Research/testing only. No deployment.")
    a(payload.get("stop_note"))
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


def _pf(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:5.2f}"


def write_v7_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v7_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
