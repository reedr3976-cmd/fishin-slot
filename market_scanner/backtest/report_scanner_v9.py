"""V9 macro/event-layer research reporting (research only)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import json
from pathlib import Path

from config import (
    V9_COMM_DEV,
    V9_COMM_FINAL_INST,
    V9_ENTRY_SLIP_ATR,
    V9_EVENT_WINDOWS,
    V9_FX_DEV,
    V9_FX_FINAL_INST,
    V9_MC_RUNS,
    V9_MC_SEED,
    V9_METALS_DEV,
    V9_METALS_FINAL_INST,
    V9_N_FOLDS,
    V9_STOCK_DEV,
    V9_STOCK_FINAL_INST,
    V9_STOCK_ROTATIONS,
    V9_TRAIN_END,
    V9_VAL_END,
)
from backtest.report_scanner_v8 import (
    _bps,
    _exp,
    _pct,
    _pf,
    by_symbol,
    decision_from_gate,
    gate_from,
    period_concentration,
    slice_metrics,
)
from backtest.scanner_v5 import leave_out_symbols, monte_carlo
from backtest.scanner_v8 import FAMILIES
from backtest.scanner_v9 import V9Spec, folds_for_spec, run_spec_on_map
from backtest.macro_features import MacroContext
from providers.macro_calendar import UNKNOWN


def _fam(key: str):
    return next(f for f in FAMILIES if f.key == key)


NONE_WIN = next(w for w in V9_EVENT_WINDOWS if w["key"] == "none")


def build_specs() -> dict[str, list[V9Spec]]:
    s1, c1, f1, b1 = _fam("S1"), _fam("C1"), _fam("F1"), _fam("B1")
    stocks: list[V9Spec] = [
        V9Spec("S_CTRL", "V9_S_CTRL_TREND", "stocks", s1, "V8 trend-pullback control (no macro).", NONE_WIN, is_control=True),
        V9Spec("S_DON", "V9_S_CTRL_DONCHIAN", "stocks", b1, "Simple Donchian baseline.", NONE_WIN, is_control=True),
    ]
    for w in V9_EVENT_WINDOWS:
        if w["key"] == "none":
            continue
        stocks.append(
            V9Spec(
                f"S_EV_{w['key']}",
                f"V9_S_EVENT_{w['key']}",
                "stocks",
                s1,
                f"Avoid new entries in HIGH-importance US event window {w['key']}.",
                w,
            )
        )
    comm: list[V9Spec] = [
        V9Spec("C_CTRL", "V9_C_CTRL_TREND", "commodities", c1, "Universal commodity trend control.", NONE_WIN, is_control=True),
        V9Spec("C_DON", "V9_C_CTRL_DONCHIAN", "commodities", b1, "Commodity Donchian baseline.", NONE_WIN, is_control=True),
    ]
    for w in V9_EVENT_WINDOWS:
        if w["key"] == "none":
            continue
        comm.append(
            V9Spec(
                f"C_EV_{w['key']}",
                f"V9_C_EVENT_{w['key']}",
                "commodities",
                c1,
                f"Event-window {w['key']} on universal commodity trend.",
                w,
            )
        )
    for gm in ("dxy_not_rising", "yields_not_rising", "not_tightening"):
        comm.append(
            V9Spec(
                f"M_{gm}",
                f"V9_M_GOLD_{gm}",
                "commodities",
                c1,
                f"Metals-only gold context filter: {gm}. Independent of direction prediction.",
                NONE_WIN,
                gold_mode=gm,
                universe_tag="metals",
            )
        )
    fx: list[V9Spec] = [
        V9Spec("F_CTRL", "V9_F_CTRL_TREND", "forex", f1, "FX daily-trend pullback control.", NONE_WIN, is_control=True),
        V9Spec("F_DON", "V9_F_CTRL_DONCHIAN", "forex", b1, "FX Donchian baseline.", NONE_WIN, is_control=True),
    ]
    for w in V9_EVENT_WINDOWS:
        if w["key"] == "none":
            continue
        fx.append(
            V9Spec(
                f"F_EV_{w['key']}",
                f"V9_F_EVENT_{w['key']}",
                "forex",
                f1,
                f"Event-window {w['key']} on FX trend-pullback.",
                w,
            )
        )
    fx.append(
        V9Spec(
            "F_CARRY",
            "V9_F_RELATIVE_CARRY",
            "forex",
            f1,
            "Relative policy: only take trades aligned with known rate differential (UNKNOWN ⇒ no filter).",
            NONE_WIN,
            fx_mode="carry_align",
        )
    )
    return {"stocks": stocks, "commodities": comm, "forex": fx}


CLASS_UNIVERSE = {
    "stocks": {
        "dev": V9_STOCK_DEV,
        "final_inst": V9_STOCK_FINAL_INST,
        "rotations": V9_STOCK_ROTATIONS,
    },
    "commodities": {
        "dev": V9_COMM_DEV,
        "final_inst": V9_COMM_FINAL_INST,
        "rotations": (("USOIL",), ("XAUUSD",), ("XAGUSD",)),
    },
    "forex": {
        "dev": V9_FX_DEV,
        "final_inst": V9_FX_FINAL_INST,
        "rotations": (("EURUSD",), ("GBPUSD",), ("USDJPY",)),
    },
}


def universe_for(spec: V9Spec, cls: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if spec.universe_tag == "metals":
        return V9_METALS_DEV, V9_METALS_FINAL_INST
    u = CLASS_UNIVERSE[cls]
    return tuple(u["dev"]), tuple(u["final_inst"])


def evaluate_spec(
    series_4h: dict,
    daily_map: dict,
    spy,
    spec: V9Spec,
    cls: str,
    ctx: Optional[MacroContext],
    control_val_exp: Optional[float],
) -> dict[str, Any]:
    dev, final_inst_syms = universe_for(spec, cls)
    print(f"  {spec.name} window={spec.event_window.get('key')} gold={spec.gold_mode} fx={spec.fx_mode}", flush=True)
    kw = dict(daily_map=daily_map, spy=spy)
    train = run_spec_on_map(series_4h, spec, dev, ctx, start_frac=0.0, end_frac=V9_TRAIN_END, **kw)
    val = run_spec_on_map(series_4h, spec, dev, ctx, start_frac=V9_TRAIN_END, end_frac=V9_VAL_END, **kw)
    final_time = run_spec_on_map(series_4h, spec, dev, ctx, start_frac=V9_VAL_END, end_frac=1.0, **kw)
    final_inst = run_spec_on_map(
        series_4h, spec, final_inst_syms, ctx, start_frac=V9_TRAIN_END, end_frac=1.0, **kw
    )
    val_2x = run_spec_on_map(
        series_4h, spec, dev, ctx, start_frac=V9_TRAIN_END, end_frac=V9_VAL_END, cost_mult=2.0, **kw
    )
    val_slip = run_spec_on_map(
        series_4h,
        spec,
        dev,
        ctx,
        start_frac=V9_TRAIN_END,
        end_frac=V9_VAL_END,
        entry_slip_atr=V9_ENTRY_SLIP_ATR,
        **kw,
    )
    folds = folds_for_spec(series_4h, spec, dev, ctx, daily_map=daily_map, spy=spy, n_folds=V9_N_FOLDS)
    fold_rows, fold_exps = [], []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append({"fold": fr["fold"], "metrics": m, "start_frac": fr["start_frac"], "end_frac": fr["end_frac"]})

    rotations = []
    rot_pos = 0
    for held_group in CLASS_UNIVERSE[cls].get("rotations") or []:
        held_group = tuple(s for s in held_group if s in dev)
        if not held_group:
            continue
        rot_trades = run_spec_on_map(
            series_4h, spec, held_group, ctx, start_frac=V9_TRAIN_END, end_frac=V9_VAL_END, **kw
        )
        e = _exp(rot_trades)
        ok = bool(e is not None and e > 0)
        rot_pos += int(ok)
        rotations.append(
            {
                "held_out_instruments": list(held_group),
                "n": len(rot_trades),
                "expectancy": e,
                "positive": ok,
            }
        )

    sens = []
    for ov in ({"atr_stop_mult": 1.25}, {"atr_stop_mult": 1.75}, {"max_hold": 20}, {"max_hold": 28}):
        t = run_spec_on_map(
            series_4h, spec, dev, ctx, start_frac=V9_TRAIN_END, end_frac=V9_VAL_END, **{**kw, **ov}
        )
        e = _exp(t)
        sens.append({"override": ov, "n": len(t), "expectancy": e, "positive": bool(e is not None and e > 0)})
    sens_pos = sum(1 for s in sens if s["positive"])

    val_sym = by_symbol(val)
    best = val_sym["best_symbols"]
    drop1 = leave_out_symbols(val, {best[0]} if best else set())
    val_exp = _exp(val)
    if spec.is_control:
        beats = True
    else:
        beats = bool(val_exp is not None and control_val_exp is not None and val_exp > control_val_exp)

    g = gate_from(
        val, val_2x, fold_exps, final_time, final_inst, sens_pos, len(sens), rot_pos, len(rotations), beats
    )
    decision = decision_from_gate(g)
    train_sym = by_symbol(train)
    return {
        "family_key": spec.key,
        "name": spec.name,
        "market_class": cls,
        "rationale": spec.rationale,
        "is_control": spec.is_control,
        "is_baseline": spec.is_control,
        "event_window": spec.event_window.get("key"),
        "gold_mode": spec.gold_mode,
        "fx_mode": spec.fx_mode,
        "frozen_rules": {
            "base_entry": spec.base.name,
            "event_window": spec.event_window,
            "gold_mode": spec.gold_mode,
            "fx_mode": spec.fx_mode,
            "surprise_used": False,
            "consensus_used": False,
            "unknown_policy": "UNKNOWN fields never generate entries or blocks except event timing when known",
        },
        "dev": list(dev),
        "final_inst_symbols": list(final_inst_syms),
        "train": slice_metrics("train", train),
        "train_n": len(train),
        "train_expectancy": _exp(train),
        "train_diversified": len(train_sym["symbols_positive_ge3"]) >= 2
        and not ((train_sym.get("best1_share") or 0) >= 0.70),
        "val": slice_metrics("val", val),
        "val_2x": slice_metrics("2x", val_2x),
        "val_slip": slice_metrics("slip", val_slip),
        "final_time": slice_metrics("final_time", final_time),
        "final_inst": slice_metrics("final_inst", final_inst),
        "final_inst_by_symbol": by_symbol(final_inst),
        "folds": fold_rows,
        "period_concentration": period_concentration(fold_rows),
        "instrument_rotations": rotations,
        "by_symbol": val_sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "sensitivity": sens,
        "monte_carlo": monte_carlo(val, n_runs=V9_MC_RUNS, seed=V9_MC_SEED),
        "beats_control_val": beats,
        "control_val_exp": control_val_exp,
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
    }


def select_on_train(results: list[dict]) -> Optional[dict]:
    """TRAIN-only among non-control specs with n≥20; else best control."""
    cands = [r for r in results if not r.get("is_control")]
    preferred = [
        r
        for r in cands
        if (r.get("train_n") or 0) >= 20
        and (r.get("train_expectancy") or 0) > 0
        and r.get("train_diversified")
    ]
    pool = preferred or [r for r in cands if (r.get("train_n") or 0) >= 15]
    if pool:
        return max(pool, key=lambda r: r.get("train_expectancy") or float("-inf"))
    ctrls = [r for r in results if r.get("is_control")]
    if not ctrls:
        return None
    return max(ctrls, key=lambda r: r.get("train_expectancy") or float("-inf"))


def _variable_scorecard(by_class: dict[str, Any]) -> list[dict[str, Any]]:
    """Which macro variables improved VAL vs class control (evaluation, not selection)."""
    rows = []
    for cls, block in by_class.items():
        fams = block.get("families") or {}
        ctrl = fams.get("V9_S_CTRL_TREND") or fams.get("V9_C_CTRL_TREND") or fams.get("V9_F_CTRL_TREND")
        ctrl_val = (ctrl or {}).get("val") or {}
        ctrl_exp = ctrl_val.get("expectancy")
        for name, r in fams.items():
            if r.get("is_control"):
                continue
            val = r.get("val") or {}
            ft = r.get("final_time") or {}
            fi = r.get("final_inst") or {}
            ve = val.get("expectancy")
            improved_val = bool(ve is not None and ctrl_exp is not None and ve > ctrl_exp)
            improved_unseen = bool(
                improved_val
                and (ft.get("expectancy") or 0) > 0
                and (fi.get("expectancy") or 0) > 0
            )
            rows.append(
                {
                    "market_class": cls,
                    "candidate": name,
                    "variable": {
                        "event_window": r.get("event_window"),
                        "gold_mode": r.get("gold_mode"),
                        "fx_mode": r.get("fx_mode"),
                    },
                    "train_exp": r.get("train_expectancy"),
                    "val_exp": ve,
                    "control_val_exp": ctrl_exp,
                    "improved_val_vs_control": improved_val,
                    "final_time_exp": ft.get("expectancy"),
                    "final_inst_exp": fi.get("expectancy"),
                    "improved_unseen_holdouts": improved_unseen,
                    "decision": r.get("decision"),
                    "reason": r.get("decision_reason"),
                }
            )
    return rows


def build_v9_payload(series_4h: dict, daily_map: dict, bundle: dict[str, Any], ctx: MacroContext) -> dict[str, Any]:
    spy = series_4h.get(("SPY", "4h"))
    spec_map = build_specs()
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []
    variants = 0

    for cls, specs in spec_map.items():
        print(f"ASSET CLASS: {cls}", flush=True)
        # Evaluate controls first
        results = []
        controls = [s for s in specs if s.is_control]
        others = [s for s in specs if not s.is_control]
        for s in controls:
            results.append(evaluate_spec(series_4h, daily_map, spy, s, cls, ctx, None))
            variants += 1
        trend_ctrl = next((r for r in results if "CTRL_TREND" in r["name"]), results[0] if results else None)
        ctrl_val = (trend_ctrl.get("val") or {}).get("expectancy") if trend_ctrl else None
        for s in others:
            results.append(evaluate_spec(series_4h, daily_map, spy, s, cls, ctx, ctrl_val))
            variants += 1
        pick = select_on_train(results)
        pick_name = pick["name"] if pick else None
        candidate = next((r for r in results if r["name"] == pick_name), None)
        class_verdict = "FAIL"
        if candidate and candidate.get("decision") == "PASS" and not candidate.get("is_control"):
            class_verdict = "PASS"
            promoted.append({"asset_class": cls, "family": candidate["name"], "reason": candidate.get("decision_reason")})
        by_class[cls] = {
            "dev_symbols": list(CLASS_UNIVERSE[cls]["dev"]),
            "final_inst_symbols": list(CLASS_UNIVERSE[cls]["final_inst"]),
            "families": {r["name"]: r for r in results},
            "train_selected_family": pick_name,
            "train_selection_rule": "Max TRAIN expectancy among non-control specs with n≥20; VAL/FINAL unused.",
            "control_val_expectancy": ctrl_val,
            "selected_decision": (candidate or {}).get("decision"),
            "selected_decision_reason": (candidate or {}).get("decision_reason"),
            "class_verdict": class_verdict,
        }

    scorecard = _variable_scorecard(by_class)
    genuine = [s for s in scorecard if s.get("improved_unseen_holdouts")]
    any_pass = bool(promoted)
    verdict = (
        "V9 PASS — at least one candidate has demonstrated sufficient generalisation "
        "and robustness to proceed to the NEXT validation stage only"
        if any_pass
        else "V9 FAIL — no candidate has demonstrated sufficient evidence of a generalisable edge"
    )
    events = bundle.get("events") or []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V9 macro/event-layer research after V8 FAIL; live ORIGINAL untouched",
        "live_scanner_protected": True,
        "auto_promote": False,
        "paper_trade": False,
        "variants_tested": variants,
        "news_safety": {
            "fabricated_events": False,
            "consensus_used_in_signals": False,
            "revised_prints_used_in_signals": False,
            "unknown_token": UNKNOWN,
            "unknown_catalog": bundle.get("unknown"),
            "fetch_errors": bundle.get("errors"),
            "event_counts": bundle.get("counts"),
            "events_with_known_time": sum(
                1 for e in events if getattr(e, "ts_unix", None) and getattr(e, "time_precision", "") != "date_only"
            ),
            "high_importance_event_sample": high_events,
        },
        "nested_design": {
            "train_end": V9_TRAIN_END,
            "val_end": V9_VAL_END,
            "selection": "TRAIN only",
            "no_repair_from_final": True,
        },
        "by_asset_class": by_class,
        "macro_variable_scorecard": scorecard,
        "variables_improving_unseen_data": genuine,
        "variables_no_measurable_advantage": [
            s for s in scorecard if not s.get("improved_unseen_holdouts")
        ],
        "promoted_candidates": promoted,
        "verdict_code": "V9_PASS" if any_pass else "V9_FAIL",
        "verdict": verdict,
        "next_stage": (
            ["Next validation stage only (not paper/live)"]
            if any_pass
            else ["No candidate strong enough for next validation stage"]
        ),
        "stop_note": (
            "Stop after this report. Wait for explicit approval before V10, promotion, "
            "merge to production, paper trading, or any live changes."
        ),
    }


def format_v9_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V9 — MACRO / EVENT-LAYER RESEARCH")
    a("Live ORIGINAL untouched | no paper/live | no auto-promote | no fabrication")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Variants tested: {payload.get('variants_tested')}")
    a(f"News safety: {payload.get('news_safety')}")
    a("")
    a("# UNKNOWN / UNAVAILABLE (not used in decisions)")
    for u in (payload.get("news_safety") or {}).get("unknown_catalog") or []:
        a(f"  - {u.get('item')}: {u.get('status')} — {u.get('reason')}")
    a("")
    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}")
        a(f"  DEV={block.get('dev_symbols')} FINAL_INST={block.get('final_inst_symbols')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}  verdict={block.get('class_verdict')}")
        a(f"  selected: {block.get('selected_decision')} — {block.get('selected_decision_reason')}")
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(f"  {name} ctrl={r.get('is_control')} window={r.get('event_window')} gold={r.get('gold_mode')} fx={r.get('fx_mode')}")
            a(f"    rationale: {r.get('rationale')}")
            for key in ("train", "val", "val_2x", "val_slip", "final_time", "final_inst"):
                m = r.get(key) or {}
                a(
                    f"    {key:12s} n={m.get('signals', 0):4d} win={_pct(m.get('win_rate'))} "
                    f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} "
                    f"DD={_pct(m.get('max_drawdown'))}"
                )
            a("    folds:")
            for fr in r.get("folds") or []:
                m = fr.get("metrics") or {}
                a(f"      fold {fr['fold']} n={m.get('signals', 0)} exp={_bps(m.get('expectancy'))}")
            a(f"    rotations: {r.get('instrument_rotations')}")
            a(f"    VAL pnl={(r.get('by_symbol') or {}).get('pnl_contribution')}")
            a(f"    FINAL_INST pnl={(r.get('final_inst_by_symbol') or {}).get('pnl_contribution')}")
            a(f"    beats_control_val={r.get('beats_control_val')} DECISION={r.get('decision')} — {r.get('decision_reason')}")
        a("")
    a("=" * 72)
    a("MACRO VARIABLE SCORECARD (unseen-data improvement vs control)")
    a("=" * 72)
    for row in payload.get("macro_variable_scorecard") or []:
        a(
            f"  [{row.get('decision')}] {row.get('market_class')} {row.get('candidate')} "
            f"var={row.get('variable')} VAL_impr={row.get('improved_val_vs_control')} "
            f"unseen_impr={row.get('improved_unseen_holdouts')} "
            f"VAL={_bps(row.get('val_exp'))} FT={_bps(row.get('final_time_exp'))} FI={_bps(row.get('final_inst_exp'))}"
        )
        a(f"      {row.get('reason')}")
    a("")
    a("# Improved unseen holdouts")
    for row in payload.get("variables_improving_unseen_data") or []:
        a(f"  {row}")
    if not payload.get("variables_improving_unseen_data"):
        a("  (none)")
    a("")
    a("=" * 72)
    a(f"FINAL DECISION: {payload.get('verdict')}")
    a(f"CODE: {payload.get('verdict_code')}")
    a(f"VARIANTS_TESTED: {payload.get('variants_tested')}")
    a(f"PROMOTED: {payload.get('promoted_candidates')}")
    a("Live scanner: NOT modified. No paper/live. No merge.")
    a(payload.get("stop_note"))
    a("=" * 72)
    return "\n".join(lines) + "\n"


def write_v9_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v9_report(payload)

    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return text
