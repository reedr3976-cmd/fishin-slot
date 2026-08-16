"""Reporting / orchestration for Scanner V4 research (analysis only)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    V4_MAX_DD_ACCEPT,
    V4_MIN_FOLDS_POSITIVE,
    V4_MIN_SYMBOLS_POSITIVE,
    V4_MIN_TRADES,
    V4_N_FOLDS,
    V4_TRAIN_FRACTION,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v4 import (
    FILTER_NAMES,
    STAGE2_VARIANTS,
    V4_S1_BASELINE,
    V4Candidate,
    chronological_folds_v4,
    collect_v4,
    fx_clone,
    run_v4_window,
    with_filter,
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
    return d


def by_key(trades: list[V2Trade], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        groups[str(key_fn(t))].append(t)
    return {k: slice_metrics(k, v) for k, v in sorted(groups.items())}


def concentration(trades: list[V2Trade]) -> dict[str, Any]:
    """Is edge dependent on a few symbols?"""
    if not trades:
        return {"n_symbols": 0, "top1_pnl_share": None, "top2_pnl_share": None, "note": "no trades"}
    by_sym: dict[str, float] = defaultdict(float)
    for t in trades:
        by_sym[t.instrument] += t.net_return
    total_pos = sum(v for v in by_sym.values() if v > 0)
    ranked = sorted(by_sym.items(), key=lambda kv: kv[1], reverse=True)
    top1 = ranked[0][1] if ranked else 0.0
    top2 = sum(v for _, v in ranked[:2])
    share1 = (top1 / total_pos) if total_pos > 0 else None
    share2 = (top2 / total_pos) if total_pos > 0 else None
    return {
        "n_symbols": len(by_sym),
        "symbol_pnl": {k: float(v) for k, v in ranked},
        "top1_symbol": ranked[0][0] if ranked else None,
        "top1_pnl_share_of_winners": share1,
        "top2_pnl_share_of_winners": share2,
        "dependent_on_one": bool(share1 is not None and share1 >= 0.70),
        "note": "top shares are of sum of positive symbol PnL sums",
    }


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
    # Reasonable 2× resilience: positive, or retains ≥50% of positive 1× expectancy
    resilient_2x = False
    if exp2 is not None and exp2 > 0:
        resilient_2x = True
    elif exp is not None and exp > 0 and exp2 is not None and exp2 >= 0.5 * exp:
        resilient_2x = True

    conc = concentration(test)
    checks = {
        "positive_oos_expectancy": bool(exp is not None and exp > 0),
        "positive_after_normal_costs": bool(exp is not None and exp > 0),  # nets include costs
        "resilient_2x_costs": resilient_2x,
        "folds_positive_ge_3": pos_folds >= V4_MIN_FOLDS_POSITIVE,
        "adequate_trade_count": n >= V4_MIN_TRADES,
        "acceptable_max_drawdown": bool(
            test_m.get("max_drawdown") is not None and test_m["max_drawdown"] <= V4_MAX_DD_ACCEPT
        ),
        "not_single_symbol": len(sym_pos) >= V4_MIN_SYMBOLS_POSITIVE
        and not conc.get("dependent_on_one", False),
    }
    checks["folds_positive_count"] = pos_folds
    checks["symbols_positive_ge5"] = sym_pos
    checks["test_expectancy"] = exp
    checks["test_2x_expectancy"] = exp2
    checks["test_n"] = n
    checks["test_max_dd"] = test_m.get("max_drawdown")
    checks["test_total_return"] = test_m.get("total_return")
    checks["concentration"] = conc
    checks["all_pass"] = all(
        checks[k]
        for k in (
            "positive_oos_expectancy",
            "positive_after_normal_costs",
            "resilient_2x_costs",
            "folds_positive_ge_3",
            "adequate_trade_count",
            "acceptable_max_drawdown",
            "not_single_symbol",
        )
    )
    return checks


def evaluate_candidate(
    series_map: dict,
    cand: V4Candidate,
    *,
    train_frac: float = V4_TRAIN_FRACTION,
    n_folds: int = V4_N_FOLDS,
) -> dict[str, Any]:
    print(f"  collecting {cand.name} classes={cand.asset_classes}...", flush=True)
    cached = collect_v4(series_map, cand, cost_mult=1.0)
    train = run_v4_window(
        series_map, cand, start_frac=0.0, end_frac=train_frac, cost_mult=1.0, cached=cached
    )
    test = run_v4_window(
        series_map, cand, start_frac=train_frac, end_frac=1.0, cost_mult=1.0, cached=cached
    )
    test_2x = run_v4_window(
        series_map, cand, start_frac=train_frac, end_frac=1.0, cost_mult=2.0, cached=cached
    )
    folds = chronological_folds_v4(
        series_map, cand, n_folds=n_folds, cost_mult=1.0, cached=cached
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
        "name": cand.name,
        "notes": cand.notes,
        "break_mode": cand.break_mode,
        "filter_name": cand.filter_name,
        "asset_classes": list(cand.asset_classes),
        "train": slice_metrics("train", train),
        "test": slice_metrics("test", test),
        "test_2x_costs": slice_metrics("test_2x", test_2x),
        "folds": fold_rows,
        "by_symbol": by_sym,
        "by_asset_class": by_cls,
        "oos_gate": gate,
        "train_n": len(train),
        "train_expectancy": _exp(train),
    }


def evaluate_class_attribution(
    series_map: dict, *, train_frac: float = V4_TRAIN_FRACTION, n_folds: int = V4_N_FOLDS
) -> dict[str, Any]:
    """Stage 1: V3_S2 baseline split by asset class (diagnostic; may inspect OOS)."""
    out: dict[str, Any] = {}
    for cls in ("stock", "commodity", "forex"):
        cand = V4Candidate(
            name=f"V4_S1_{cls.upper()}",
            notes=f"V3_S2 STRUCTURE+MA + 20-bar breakout on {cls} only (diagnostic).",
            break_mode="donchian20",
            filter_name="none",
            asset_classes=(cls,),
        )
        out[cls] = evaluate_candidate(series_map, cand, train_frac=train_frac, n_folds=n_folds)

    # Combined stock+commodity diagnostic
    sc = V4Candidate(
        name="V4_S1_STOCK_COMMODITY",
        notes="V3_S2 on stocks+commodities combined (diagnostic).",
        break_mode="donchian20",
        filter_name="none",
        asset_classes=("stock", "commodity"),
    )
    out["stock_commodity"] = evaluate_candidate(
        series_map, sc, train_frac=train_frac, n_folds=n_folds
    )
    return out


def stage1_justifies_stage2(attr: dict[str, Any]) -> tuple[bool, str]:
    """Use Stage-1 OOS diagnostics (explicitly allowed) to decide whether to build Stage 2."""
    fx = attr["forex"]["oos_gate"].get("test_expectancy")
    st = attr["stock"]["oos_gate"].get("test_expectancy")
    co = attr["commodity"]["oos_gate"].get("test_expectancy")
    sc = attr["stock_commodity"]["oos_gate"].get("test_expectancy")
    fx_folds = attr["forex"]["oos_gate"].get("folds_positive_count", 0)
    sc_folds = attr["stock_commodity"]["oos_gate"].get("folds_positive_count", 0)

    parts = []
    better = False
    # Material: SC or stock or commodity OOS exp > FX, and not a one-symbol artefact
    for label, exp, block in (
        ("stock", st, attr["stock"]),
        ("commodity", co, attr["commodity"]),
        ("stock_commodity", sc, attr["stock_commodity"]),
    ):
        if exp is None or fx is None:
            continue
        if exp > fx:
            conc = block["oos_gate"].get("concentration", {})
            n = block["oos_gate"].get("test_n", 0)
            parts.append(
                f"{label} OOS exp={exp:.6f} > FX={fx:.6f} (n={n}, "
                f"folds+={block['oos_gate'].get('folds_positive_count')}, "
                f"one_sym={conc.get('dependent_on_one')})"
            )
            if n >= 15 and not conc.get("dependent_on_one", False):
                better = True

    if better and (sc is not None and fx is not None and sc > fx):
        return True, "Stage 2 justified: " + "; ".join(parts)
    if better:
        return True, "Stage 2 justified (class-level): " + "; ".join(parts)
    return (
        False,
        f"Stage 2 not justified. FX OOS={fx} folds+={fx_folds}; "
        f"SC OOS={sc} folds+={sc_folds}; stock={st}; commodity={co}. "
        + ("; ".join(parts) if parts else "No material SC>FX OOS edge."),
    )


def select_on_train(results: list[dict[str, Any]], *, min_train_n: int = 20) -> Optional[dict]:
    """Pick best TRAIN expectancy with adequate n. Never uses TEST."""
    eligible = [r for r in results if (r.get("train_n") or 0) >= min_train_n]
    if not eligible:
        eligible = [r for r in results if (r.get("train_n") or 0) > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.get("train_expectancy") or float("-inf"))


def build_v4_payload(series_map: dict) -> dict[str, Any]:
    variants_tested = 0

    # ----- Stage 1 -----
    print("STAGE 1 — asset-class attribution (V3_S2 unchanged)...", flush=True)
    # Also full-universe baseline for reference
    baseline_all = evaluate_candidate(series_map, V4_S1_BASELINE)
    variants_tested += 1
    attribution = evaluate_class_attribution(series_map)
    variants_tested += 4  # stock, commodity, forex, stock_commodity

    justified, justify_reason = stage1_justifies_stage2(attribution)

    stage2_results: dict[str, Any] = {}
    stage3_results: dict[str, Any] = {}
    stage4_results: dict[str, Any] = {}
    selected_s2 = None
    selected_s3 = None
    frozen_candidate: Optional[V4Candidate] = None
    selection_log: list[str] = []

    if justified:
        # ----- Stage 2: TRAIN-only selection among structural variants -----
        print("STAGE 2 — structural swing-break variants (TRAIN select)...", flush=True)
        s2_list = []
        for cand in STAGE2_VARIANTS:
            r = evaluate_candidate(series_map, cand)
            stage2_results[cand.name] = r
            s2_list.append(r)
            variants_tested += 1
        selected_s2 = select_on_train(s2_list)
        if selected_s2:
            selection_log.append(
                f"Stage2 TRAIN pick: {selected_s2['name']} "
                f"train_exp={selected_s2.get('train_expectancy')} n={selected_s2.get('train_n')}"
            )
            base = next(c for c in STAGE2_VARIANTS if c.name == selected_s2["name"])

            # ----- Stage 3: filters individually; TRAIN select -----
            print("STAGE 3 — false-breakout filters individually (TRAIN select)...", flush=True)
            s3_list = []
            for fname in FILTER_NAMES:
                cand = with_filter(base, fname)
                # Avoid duplicate eval of identical "none" already in stage2
                if fname == "none":
                    stage3_results[cand.name] = selected_s2
                    s3_list.append(selected_s2)
                    continue
                r = evaluate_candidate(series_map, cand)
                stage3_results[cand.name] = r
                s3_list.append(r)
                variants_tested += 1
            selected_s3 = select_on_train(s3_list)
            if selected_s3:
                selection_log.append(
                    f"Stage3 TRAIN pick: {selected_s3['name']} "
                    f"train_exp={selected_s3.get('train_expectancy')} n={selected_s3.get('train_n')}"
                )
                # Reconstruct frozen candidate
                fname = "none"
                for fn in FILTER_NAMES:
                    if selected_s3["name"].endswith(f"__{fn}"):
                        fname = fn
                        break
                if selected_s3["name"] == base.name:
                    frozen_candidate = base
                else:
                    frozen_candidate = with_filter(base, fname)

            # ----- Stage 4: FX control with frozen params -----
            if frozen_candidate is not None:
                print("STAGE 4 — FX control with frozen params...", flush=True)
                fx_cand = fx_clone(frozen_candidate)
                stage4_results[fx_cand.name] = evaluate_candidate(series_map, fx_cand)
                variants_tested += 1
                # Re-state stock+commodity frozen result (already evaluated)
                sc_name = frozen_candidate.name
                if sc_name in stage3_results:
                    stage4_results[f"{sc_name}__STOCK_COMMODITY"] = stage3_results[sc_name]
                elif sc_name in stage2_results:
                    stage4_results[f"{sc_name}__STOCK_COMMODITY"] = stage2_results[sc_name]
    else:
        selection_log.append("Stage 2–4 skipped: " + justify_reason)

    verdict_code, verdict_label, promoted = final_verdict(
        attribution=attribution,
        justified=justified,
        stage2=stage2_results,
        stage3=stage3_results,
        stage4=stage4_results,
        frozen_name=frozen_candidate.name if frozen_candidate else None,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Scanner V4 research only — live ORIGINAL untouched; V3 not merged",
        "hypothesis": "V3_S2 STRUCTURE+MA; test whether stocks/commodities hold a robust edge",
        "timeframe": "4h",
        "risk_fraction": 0.01,
        "train_fraction": V4_TRAIN_FRACTION,
        "n_folds": V4_N_FOLDS,
        "selection_rule": "Variant/filter chosen on TRAIN expectancy (n≥20) then frozen; OOS never used for selection",
        "variants_tested_count": variants_tested,
        "stage1_baseline_all": baseline_all,
        "stage1_attribution": attribution,
        "stage1_justifies_stage2": justified,
        "stage1_justify_reason": justify_reason,
        "stage2_structural_variants": stage2_results,
        "stage3_filters": stage3_results,
        "stage4_fx_control": stage4_results,
        "selection_log": selection_log,
        "frozen_candidate": frozen_candidate.name if frozen_candidate else None,
        "frozen_spec": {
            "break_mode": frozen_candidate.break_mode,
            "filter_name": frozen_candidate.filter_name,
            "asset_classes": list(frozen_candidate.asset_classes),
        }
        if frozen_candidate
        else None,
        "verdict_code": verdict_code,
        "verdict": verdict_label,
        "promoted_candidate": promoted,
        "instruments": sorted({k for k, _ in series_map}),
        "series_bars": {f"{k}:{tf}": len(s) for (k, tf), s in series_map.items()},
    }


def final_verdict(
    *,
    attribution: dict,
    justified: bool,
    stage2: dict,
    stage3: dict,
    stage4: dict,
    frozen_name: Optional[str],
) -> tuple[str, str, Optional[str]]:
    sc_gate = None
    fx_gate = None
    sc_name = None

    if frozen_name:
        sc_block = stage3.get(frozen_name) or stage2.get(frozen_name)
        if sc_block:
            sc_gate = sc_block.get("oos_gate")
            sc_name = frozen_name
        fx_key = f"{frozen_name}__FX"
        if fx_key in stage4:
            fx_gate = stage4[fx_key].get("oos_gate")

    # If Stage 2 not run, still allow PASS only on diagnostic SC if it somehow passes all gates
    if sc_gate is None and justified is False:
        sc_block = attribution.get("stock_commodity")
        if sc_block and sc_block.get("oos_gate", {}).get("all_pass"):
            sc_gate = sc_block["oos_gate"]
            sc_name = "V4_S1_STOCK_COMMODITY"
            fx_gate = attribution.get("forex", {}).get("oos_gate")

    if sc_gate and sc_gate.get("all_pass"):
        fx_fail = not (fx_gate and fx_gate.get("positive_oos_expectancy"))
        if fx_fail:
            return (
                "PASS_SC_FX_SEPARATE",
                "STOCK/COMMODITY CANDIDATE PASSES — FX REQUIRES SEPARATE RESEARCH. "
                f"Candidate={sc_name}. Do NOT merge into live ORIGINAL.",
                sc_name,
            )
        return (
            "PASS",
            f"PASS — suitable to proceed to paper-trading validation ({sc_name}). "
            "Do NOT merge into live ORIGINAL automatically.",
            sc_name,
        )

    return (
        "FAIL",
        "FAIL — no sufficiently robust edge demonstrated. "
        "Live ORIGINAL untouched; V3 not merged.",
        sc_name,
    )


def format_v4_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V4 — RESEARCH / BACKTEST ONLY")
    a("Live ORIGINAL untouched | V3 not merged | no live/paper enablement")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Hypothesis: {payload.get('hypothesis')}")
    a(f"Variants tested: {payload.get('variants_tested_count')}")
    a(f"Selection: {payload.get('selection_rule')}")
    a(f"Frozen candidate: {payload.get('frozen_candidate')} {payload.get('frozen_spec')}")
    a("")

    def dump(name: str, s: dict) -> None:
        a("-" * 72)
        a(f"  {name}")
        a(f"  {s.get('notes')}")
        for period in ("train", "test", "test_2x_costs"):
            m = s.get(period) or {}
            a(
                f"    {period:14s} n={m.get('signals', 0):4d} win={_pct(m.get('win_rate'))} "
                f"avgW={_bps(m.get('avg_winner'))} avgL={_bps(m.get('avg_loser'))} "
                f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} "
                f"DD={_pct(m.get('max_drawdown'))} tot={_pct(m.get('total_return'))}"
            )
        a("    folds:")
        for fr in s.get("folds") or []:
            m = fr.get("metrics") or {}
            a(
                f"      fold {fr['fold']} n={m.get('signals', 0)} "
                f"exp={_bps(m.get('expectancy'))} DD={_pct(m.get('max_drawdown'))}"
            )
        a("    by symbol (TEST):")
        for sym, m in (s.get("by_symbol") or {}).items():
            a(f"      {sym:8s} n={m.get('signals', 0):3d} exp={_bps(m.get('expectancy'))}")
        g = s.get("oos_gate") or {}
        a(
            f"    OOS gate: pass={g.get('all_pass')} exp={_bps(g.get('test_expectancy'))} "
            f"2x={_bps(g.get('test_2x_expectancy'))} folds+={g.get('folds_positive_count')} "
            f"syms={g.get('symbols_positive_ge5')} one_sym={((g.get('concentration') or {}).get('dependent_on_one'))}"
        )

    a("# STAGE 1 — ASSET-CLASS ATTRIBUTION (V3_S2 unchanged)")
    a(f"  justifies_stage2={payload.get('stage1_justifies_stage2')}")
    a(f"  reason={payload.get('stage1_justify_reason')}")
    dump("ALL_CLASSES_BASELINE", payload.get("stage1_baseline_all") or {})
    for cls, s in (payload.get("stage1_attribution") or {}).items():
        dump(f"CLASS_{cls}", s)

    a("")
    a("# STAGE 2 — STRUCTURAL SWING BREAK (stock+commodity)")
    if not payload.get("stage2_structural_variants"):
        a("  (skipped)")
    for name, s in (payload.get("stage2_structural_variants") or {}).items():
        dump(name, s)

    a("")
    a("# STAGE 3 — FALSE-BREAKOUT FILTERS (individual, TRAIN-selected)")
    if not payload.get("stage3_filters"):
        a("  (skipped)")
    for name, s in (payload.get("stage3_filters") or {}).items():
        dump(name, s)

    a("")
    a("# STAGE 4 — FX CONTROL (frozen params)")
    if not payload.get("stage4_fx_control"):
        a("  (skipped)")
    for name, s in (payload.get("stage4_fx_control") or {}).items():
        dump(name, s)

    a("")
    a("# SELECTION LOG (TRAIN only)")
    for line in payload.get("selection_log") or []:
        a(f"  - {line}")
    a("")
    a("=" * 72)
    a(f"VERDICT: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    a(f"PROMOTED: {payload.get('promoted_candidate')}")
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
        return "  inf"
    return f"{x:5.2f}"


def write_v4_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v4_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
