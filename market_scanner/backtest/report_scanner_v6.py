"""V6 clean strategy-family reset reporting (research only)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    V6_ATR_STOP_MULT,
    V6_COMMODITY_DISCOVERY,
    V6_COMMODITY_HELDOUT,
    V6_ENTRY_SLIP_ATR,
    V6_FX_DISCOVERY,
    V6_FX_HELDOUT,
    V6_MAX_DD_ACCEPT,
    V6_MAX_HOLD,
    V6_MC_RUNS,
    V6_MC_SEED,
    V6_MIN_FOLDS_POSITIVE,
    V6_MIN_SYMBOLS_POSITIVE,
    V6_MIN_TRADES,
    V6_N_FOLDS,
    V6_STOCK_DISCOVERY,
    V6_STOCK_HELDOUT,
    V6_TRAIN_FRACTION,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import leave_out_symbols, monte_carlo
from backtest.scanner_v6 import FAMILIES, V6Family, folds_for_family, run_family_on_map


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
    sym_pos = [
        s
        for s, mm in sym["per_symbol"].items()
        if (mm.get("expectancy") or 0) > 0 and (mm.get("signals") or 0) >= 3
    ]
    resilient_2x = bool(
        (exp2 is not None and exp2 > 0)
        or (exp is not None and exp > 0 and exp2 is not None and exp2 >= 0.5 * exp)
    )
    checks = {
        "positive_discovery_oos": bool(exp is not None and exp > 0),
        "positive_heldout_oos": bool(held_exp is not None and held_exp > 0),
        "resilient_2x": resilient_2x,
        "folds_ge_3": pos_folds >= V6_MIN_FOLDS_POSITIVE,
        "folds_positive_count": pos_folds,
        "adequate_n": len(test) >= V6_MIN_TRADES,
        "acceptable_dd": bool((m.get("max_drawdown") or 1) <= V6_MAX_DD_ACCEPT),
        "multi_symbol": len(sym_pos) >= min(V6_MIN_SYMBOLS_POSITIVE, max(1, len(sym["per_symbol"]))),
        "not_one_symbol": not ((sym.get("best1_share") or 0) >= 0.70),
        "not_fragile": sens_positive >= max(1, sens_total // 2),
        "heldout_n": len(held_out_test),
        "heldout_expectancy": held_exp,
        "test_expectancy": exp,
        "test_2x_expectancy": exp2,
        "symbols_positive": sym_pos,
    }
    checks["all_pass"] = all(
        checks[k]
        for k in (
            "positive_discovery_oos",
            "positive_heldout_oos",
            "resilient_2x",
            "folds_ge_3",
            "adequate_n",
            "acceptable_dd",
            "multi_symbol",
            "not_one_symbol",
            "not_fragile",
        )
    )
    return checks


def evaluate_family(
    series_4h: dict,
    daily_map: dict,
    family: V6Family,
    discovery: tuple[str, ...],
    heldout: tuple[str, ...],
) -> dict[str, Any]:
    print(f"  family {family.name} discovery={discovery}...", flush=True)
    train = run_family_on_map(
        series_4h, family, discovery, daily_map=daily_map, start_frac=0.0, end_frac=V6_TRAIN_FRACTION
    )
    test = run_family_on_map(
        series_4h, family, discovery, daily_map=daily_map, start_frac=V6_TRAIN_FRACTION, end_frac=1.0
    )
    test_15 = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V6_TRAIN_FRACTION,
        end_frac=1.0,
        cost_mult=1.5,
    )
    test_2x = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V6_TRAIN_FRACTION,
        end_frac=1.0,
        cost_mult=2.0,
    )
    test_slip = run_family_on_map(
        series_4h,
        family,
        discovery,
        daily_map=daily_map,
        start_frac=V6_TRAIN_FRACTION,
        end_frac=1.0,
        entry_slip_atr=V6_ENTRY_SLIP_ATR,
    )
    # Held-out: chronological TEST window on held-out symbols (same freeze)
    held_test = run_family_on_map(
        series_4h, family, heldout, daily_map=daily_map, start_frac=V6_TRAIN_FRACTION, end_frac=1.0
    )
    held_full = run_family_on_map(
        series_4h, family, heldout, daily_map=daily_map, start_frac=0.0, end_frac=1.0
    )
    folds = folds_for_family(
        series_4h, family, discovery, daily_map=daily_map, n_folds=V6_N_FOLDS
    )
    fold_rows = []
    fold_exps = []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append({"fold": fr["fold"], "metrics": m, "start_frac": fr["start_frac"], "end_frac": fr["end_frac"]})

    # Sensitivity around ATR stop / max hold (report only; do not pick)
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
            start_frac=V6_TRAIN_FRACTION,
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

    sym = by_symbol(test)
    best = sym["best_symbols"]
    drop1 = leave_out_symbols(test, {best[0]} if best else set())
    g = gate_from(test, test_2x, fold_exps, held_test, sens_pos, len(sens))

    return {
        "family_key": family.key,
        "name": family.name,
        "notes": family.notes,
        "discovery": list(discovery),
        "heldout": list(heldout),
        "train": slice_metrics("train", train),
        "train_n": len(train),
        "train_expectancy": _exp(train),
        "test": slice_metrics("test", test),
        "test_1_5x": slice_metrics("1.5x", test_15),
        "test_2x": slice_metrics("2x", test_2x),
        "test_slip": slice_metrics("slip", test_slip),
        "heldout_test": slice_metrics("held_test", held_test),
        "heldout_full": slice_metrics("held_full", held_full),
        "folds": fold_rows,
        "by_symbol": sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "sensitivity": sens,
        "monte_carlo": monte_carlo(test, n_runs=V6_MC_RUNS, seed=V6_MC_SEED),
        "gate": g,
    }


ASSET_CLASSES = (
    {
        "name": "stocks",
        "discovery": V6_STOCK_DISCOVERY,
        "heldout": V6_STOCK_HELDOUT,
    },
    {
        "name": "commodities",
        "discovery": V6_COMMODITY_DISCOVERY,
        "heldout": V6_COMMODITY_HELDOUT,
    },
    {
        "name": "forex",
        "discovery": V6_FX_DISCOVERY,
        "heldout": V6_FX_HELDOUT,
    },
)


def select_on_train(results: list[dict]) -> Optional[dict]:
    """TRAIN-only selection among families with n>=20; never uses TEST/held-out."""
    elig = [r for r in results if (r.get("train_n") or 0) >= 20]
    if not elig:
        elig = [r for r in results if (r.get("train_n") or 0) > 0]
    if not elig:
        return None
    return max(elig, key=lambda r: r.get("train_expectancy") or float("-inf"))


def build_v6_payload(series_4h: dict, daily_map: dict) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []

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
        # Promotion uses OOS/held-out gates on TRAIN-selected family only
        candidate = next((r for r in fam_results if r["name"] == pick_name), None)
        verdict = "FAIL"
        if candidate and candidate["gate"].get("all_pass"):
            verdict = "PASS"
            promoted.append(
                {
                    "asset_class": ac["name"],
                    "family": candidate["name"],
                    "notes": candidate["notes"],
                    "gate": candidate["gate"],
                }
            )
        elif candidate:
            # Near-miss note
            verdict = "FAIL"
        by_class[ac["name"]] = {
            "discovery_symbols": list(ac["discovery"]),
            "heldout_symbols": list(ac["heldout"]),
            "families": {r["name"]: r for r in fam_results},
            "train_selected_family": pick_name,
            "train_selection_rule": "max TRAIN expectancy among families with n≥20 (OOS unused)",
            "selected_gate": (candidate or {}).get("gate"),
            "class_verdict": verdict,
        }

    if promoted:
        overall = "V6 PARTIAL PASS — see per-class PASS candidates (not enabled for paper/live)"
        code = "V6_PARTIAL_PASS"
    else:
        overall = "V6 FAIL — no candidate passed independent robustness standards"
        code = "V6_FAIL"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V6 clean strategy-family reset; V4/V5 falsified; live ORIGINAL untouched",
        "families_tested": [{"key": f.key, "name": f.name, "notes": f.notes} for f in FAMILIES],
        "selection_rule": "Per asset class: choose family on TRAIN only, then freeze; evaluate TEST/held-out/costs/sensitivity/MC",
        "by_asset_class": by_class,
        "promoted_candidates": promoted,
        "verdict_code": code,
        "verdict": overall,
        "next_stage": (
            [
                {
                    "action": "Independent validation protocol (similar to V5) before any paper trading",
                    "candidate": p,
                }
                for p in promoted
            ]
            if promoted
            else ["No candidate strong enough for next validation stage"]
        ),
    }


def format_v6_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V6 — CLEAN STRATEGY-FAMILY RESET")
    a("V4/V5 falsified | live ORIGINAL untouched | no paper/live enablement")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Selection: {payload.get('selection_rule')}")
    a("Families:")
    for f in payload.get("families_tested") or []:
        a(f"  {f['key']}: {f['name']} — {f['notes']}")
    a("")

    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}")
        a(f"  discovery={block.get('discovery_symbols')}  heldout={block.get('heldout_symbols')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}")
        a(f"  class_verdict: {block.get('class_verdict')}")
        a(f"  selected_gate: {block.get('selected_gate')}")
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(f"  {name}  [{r.get('family_key')}] {r.get('notes')}")
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
            sym = r.get("by_symbol") or {}
            a(f"    best1_share={sym.get('best1_share')} best3={sym.get('best3_share')}")
            for s, m in (sym.get("per_symbol") or {}).items():
                a(f"      {s:8s} n={m.get('signals', 0):3d} exp={_bps(m.get('expectancy'))}")
            a(f"    sensitivity: {r.get('sensitivity')}")
            a(f"    monte_carlo: {r.get('monte_carlo')}")
            a(f"    GATE all_pass={((r.get('gate') or {}).get('all_pass'))} {r.get('gate')}")
        a("")

    a("=" * 72)
    a(f"VERDICT: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    a(f"PROMOTED: {payload.get('promoted_candidates')}")
    a(f"NEXT: {payload.get('next_stage')}")
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


def write_v6_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v6_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
