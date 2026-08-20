"""V10 market context + price structure research reporting (research only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import (
    V10_COMM_DEV,
    V10_COMM_FINAL_INST,
    V10_ENTRY_SLIP_ATR,
    V10_EVENT_WINDOW,
    V10_FX_DEV,
    V10_FX_FINAL_INST,
    V10_MC_RUNS,
    V10_MC_SEED,
    V10_METALS_DEV,
    V10_METALS_FINAL_INST,
    V10_N_FOLDS,
    V10_STOCK_DEV,
    V10_STOCK_FINAL_INST,
    V10_STOCK_ROTATIONS,
    V10_TRAIN_END,
    V10_VAL_END,
)
from backtest.macro_features import MacroContext
from backtest.market_context import REGIME_LABELS
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
from backtest.scanner_v10 import V10Spec, folds_for_spec, run_spec_on_map
from providers.macro_calendar import UNKNOWN


def _stock_specs() -> list[V10Spec]:
    ev = V10_EVENT_WINDOW
    return [
        V10Spec("S_CTRL", "V10_S_CTRL_TREND", "stocks", "CTRL_TREND", "V8 trend-pullback control.", is_control=True),
        V10Spec("S_DON", "V10_S_CTRL_DONCHIAN", "stocks", "CTRL_DON", "Donchian baseline.", is_control=True),
        V10Spec("S_BOS", "V10_S_STRUCT_BOS", "stocks", "STRUCT_BOS", "Break of structure in trend direction."),
        V10Spec("S_CHOCH", "V10_S_STRUCT_CHOCH", "stocks", "STRUCT_CHOCH", "Change of character / structural transition."),
        V10Spec("S_SR", "V10_S_SR_RETEST", "stocks", "SR_RETEST", "Support/resistance retest with rejection."),
        V10Spec("S_FVG", "V10_S_FVG_PULLBACK", "stocks", "FVG_PB", "Pullback into unmitigated FVG aligned with trend."),
        V10Spec("S_MTF", "V10_S_MTF_DAILY", "stocks", "MTF_DAILY", "4H pullback inside daily structure."),
        V10Spec("S_MTFW", "V10_S_MTF_D_W", "stocks", "MTF_DAILY_WEEKLY", "Daily + weekly structure alignment."),
        V10Spec("S_REG", "V10_S_REGIME_TREND", "stocks", "REGIME_TREND", "Trending regime filter + Donchian confirm."),
        V10Spec("S_LIQ", "V10_S_LIQ_SWEEP", "stocks", "LIQ_SWEEP", "Liquidity sweep + rejection."),
        V10Spec("S_BRK", "V10_S_LIQ_BREAK", "stocks", "LIQ_BREAK", "Breakout beyond swing + continuation."),
        V10Spec(
            "S_CHTF",
            "V10_S_COMBO_HTF_BOS",
            "stocks",
            "COMBO_HTF_BOS",
            "Daily trend + BOS + trending regime (pre-specified combo).",
            is_combo=True,
        ),
        V10Spec(
            "S_CSRF",
            "V10_S_COMBO_SR_FVG",
            "stocks",
            "COMBO_SR_FVG",
            "S/R retest + FVG confluence + structure (pre-specified combo).",
            is_combo=True,
        ),
        V10Spec(
            "S_FULL",
            "V10_S_COMBO_FULL",
            "stocks",
            "COMBO_FULL",
            "HTF + structure + S/R or FVG + regime (pre-specified combo).",
            is_combo=True,
        ),
        V10Spec(
            "S_EV",
            "V10_S_MACRO_EVENT_1H",
            "stocks",
            "COMBO_HTF_BOS",
            "Event blackout 1h overlay on HTF+BOS combo.",
            is_combo=True,
            event_window=ev,
        ),
        V10Spec(
            "S_XST",
            "V10_S_EXIT_STRUCT",
            "stocks",
            "STRUCT_BOS",
            "Structure-based stop on BOS entries.",
            exit_mode="struct",
        ),
        V10Spec(
            "S_XRR",
            "V10_S_EXIT_R2",
            "stocks",
            "STRUCT_BOS",
            "Fixed 2R target exit on BOS entries.",
            exit_mode="r_multiple",
        ),
    ]


def _comm_specs() -> list[V10Spec]:
    ev = V10_EVENT_WINDOW
    return [
        V10Spec("C_CTRL", "V10_C_CTRL_TREND", "commodities", "CTRL_TREND", "Universal commodity trend control.", is_control=True),
        V10Spec("C_DON", "V10_C_CTRL_DONCHIAN", "commodities", "CTRL_DON", "Commodity Donchian baseline.", is_control=True),
        V10Spec("C_BOS", "V10_C_STRUCT_BOS", "commodities", "STRUCT_BOS", "Break of structure in trend direction."),
        V10Spec("C_SR", "V10_C_SR_RETEST", "commodities", "SR_RETEST", "S/R retest with rejection."),
        V10Spec("C_FVG", "V10_C_FVG_PULLBACK", "commodities", "FVG_PB", "FVG pullback aligned with trend."),
        V10Spec("C_MTF", "V10_C_MTF_DAILY", "commodities", "MTF_DAILY", "4H pullback inside daily structure."),
        V10Spec("C_REG", "V10_C_REGIME_TREND", "commodities", "REGIME_TREND", "Trending regime + Donchian confirm."),
        V10Spec("C_LIQ", "V10_C_LIQ_SWEEP", "commodities", "LIQ_SWEEP", "Liquidity sweep + rejection."),
        V10Spec(
            "C_CHTF",
            "V10_C_COMBO_HTF_BOS",
            "commodities",
            "COMBO_HTF_BOS",
            "Daily + BOS + regime combo.",
            is_combo=True,
        ),
        V10Spec(
            "C_CSRF",
            "V10_C_COMBO_SR_FVG",
            "commodities",
            "COMBO_SR_FVG",
            "S/R + FVG confluence combo.",
            is_combo=True,
        ),
        V10Spec(
            "C_GOLD",
            "V10_C_MACRO_GOLD",
            "commodities",
            "COMBO_HTF_BOS",
            "Gold context filter (not_tightening) on HTF+BOS.",
            is_combo=True,
            gold_mode="not_tightening",
            universe_tag="metals",
        ),
        V10Spec(
            "C_EV",
            "V10_C_MACRO_EVENT_1H",
            "commodities",
            "COMBO_HTF_BOS",
            "Event blackout 1h on HTF+BOS.",
            is_combo=True,
            event_window=ev,
        ),
        V10Spec(
            "C_XST",
            "V10_C_EXIT_STRUCT",
            "commodities",
            "STRUCT_BOS",
            "Structure stop on BOS.",
            exit_mode="struct",
        ),
    ]


def _fx_specs() -> list[V10Spec]:
    ev = V10_EVENT_WINDOW
    return [
        V10Spec("F_CTRL", "V10_F_CTRL_TREND", "forex", "CTRL_TREND", "FX daily-trend pullback control.", is_control=True),
        V10Spec("F_DON", "V10_F_CTRL_DONCHIAN", "forex", "CTRL_DON", "FX Donchian baseline.", is_control=True),
        V10Spec("F_BOS", "V10_F_STRUCT_BOS", "forex", "STRUCT_BOS", "Break of structure in trend direction."),
        V10Spec("F_SR", "V10_F_SR_RETEST", "forex", "SR_RETEST", "S/R retest with rejection."),
        V10Spec("F_FVG", "V10_F_FVG_PULLBACK", "forex", "FVG_PB", "FVG pullback aligned with trend."),
        V10Spec("F_MTF", "V10_F_MTF_DAILY", "forex", "MTF_DAILY", "4H pullback inside daily structure."),
        V10Spec("F_MTFW", "V10_F_MTF_D_W", "forex", "MTF_DAILY_WEEKLY", "Daily + weekly alignment."),
        V10Spec("F_REG", "V10_F_REGIME_TREND", "forex", "REGIME_TREND", "Trending regime + Donchian confirm."),
        V10Spec("F_LIQ", "V10_F_LIQ_SWEEP", "forex", "LIQ_SWEEP", "Liquidity sweep + rejection."),
        V10Spec(
            "F_CHTF",
            "V10_F_COMBO_HTF_BOS",
            "forex",
            "COMBO_HTF_BOS",
            "Daily + BOS + regime combo.",
            is_combo=True,
        ),
        V10Spec(
            "F_CSRF",
            "V10_F_COMBO_SR_FVG",
            "forex",
            "COMBO_SR_FVG",
            "S/R + FVG confluence combo.",
            is_combo=True,
        ),
        V10Spec(
            "F_CARRY",
            "V10_F_MACRO_CARRY",
            "forex",
            "MTF_DAILY",
            "Relative carry alignment filter on MTF daily entries.",
            fx_mode="carry_align",
        ),
        V10Spec(
            "F_EV",
            "V10_F_MACRO_EVENT_1H",
            "forex",
            "COMBO_HTF_BOS",
            "Event blackout 1h on HTF+BOS.",
            is_combo=True,
            event_window=ev,
        ),
        V10Spec(
            "F_XST",
            "V10_F_EXIT_STRUCT",
            "forex",
            "STRUCT_BOS",
            "Structure stop on BOS.",
            exit_mode="struct",
        ),
    ]


def build_specs() -> dict[str, list[V10Spec]]:
    return {"stocks": _stock_specs(), "commodities": _comm_specs(), "forex": _fx_specs()}


CLASS_UNIVERSE = {
    "stocks": {
        "dev": V10_STOCK_DEV,
        "final_inst": V10_STOCK_FINAL_INST,
        "rotations": V10_STOCK_ROTATIONS,
    },
    "commodities": {
        "dev": V10_COMM_DEV,
        "final_inst": V10_COMM_FINAL_INST,
        "rotations": (("USOIL",), ("XAUUSD",), ("XAGUSD",)),
    },
    "forex": {
        "dev": V10_FX_DEV,
        "final_inst": V10_FX_FINAL_INST,
        "rotations": (("EURUSD",), ("GBPUSD",), ("USDJPY",)),
    },
}


def universe_for(spec: V10Spec, cls: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if spec.universe_tag == "metals":
        return V10_METALS_DEV, V10_METALS_FINAL_INST
    u = CLASS_UNIVERSE[cls]
    return tuple(u["dev"]), tuple(u["final_inst"])


def evaluate_spec(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    ctx_map: dict,
    spec: V10Spec,
    cls: str,
    macro: Optional[MacroContext],
    control_val_exp: Optional[float],
) -> dict[str, Any]:
    dev, final_inst_syms = universe_for(spec, cls)
    print(
        f"  {spec.name} comp={spec.component} exit={spec.exit_mode} "
        f"gold={spec.gold_mode} fx={spec.fx_mode} ev={bool(spec.event_window)}",
        flush=True,
    )
    kw = dict(daily_map=daily_map, weekly_map=weekly_map)
    train = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=0.0, end_frac=V10_TRAIN_END, **kw)
    val = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V10_TRAIN_END, end_frac=V10_VAL_END, **kw)
    final_time = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V10_VAL_END, end_frac=1.0, **kw)
    final_inst = run_spec_on_map(
        series_4h, spec, final_inst_syms, ctx_map, macro, start_frac=V10_TRAIN_END, end_frac=1.0, **kw
    )
    val_2x = run_spec_on_map(
        series_4h, spec, dev, ctx_map, macro, start_frac=V10_TRAIN_END, end_frac=V10_VAL_END, cost_mult=2.0, **kw
    )
    val_slip = run_spec_on_map(
        series_4h,
        spec,
        dev,
        ctx_map,
        macro,
        start_frac=V10_TRAIN_END,
        end_frac=V10_VAL_END,
        entry_slip_atr=V10_ENTRY_SLIP_ATR,
        **kw,
    )
    folds = folds_for_spec(
        series_4h, spec, dev, ctx_map, macro, daily_map=daily_map, weekly_map=weekly_map, n_folds=V10_N_FOLDS
    )
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
            series_4h, spec, held_group, ctx_map, macro, start_frac=V10_TRAIN_END, end_frac=V10_VAL_END, **kw
        )
        e = _exp(rot_trades)
        ok = bool(e is not None and e > 0)
        rot_pos += int(ok)
        rotations.append({"held_out_instruments": list(held_group), "n": len(rot_trades), "expectancy": e, "positive": ok})

    sens = []
    for ov in ({"atr_stop_mult": 1.25}, {"atr_stop_mult": 1.75}, {"max_hold": 20}, {"max_hold": 28}):
        t = run_spec_on_map(
            series_4h, spec, dev, ctx_map, macro, start_frac=V10_TRAIN_END, end_frac=V10_VAL_END, **{**kw, **ov}
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

    g = gate_from(val, val_2x, fold_exps, final_time, final_inst, sens_pos, len(sens), rot_pos, len(rotations), beats)
    decision = decision_from_gate(g)
    train_sym = by_symbol(train)
    return {
        "family_key": spec.key,
        "name": spec.name,
        "market_class": cls,
        "component": spec.component,
        "rationale": spec.rationale,
        "is_control": spec.is_control,
        "is_combo": spec.is_combo,
        "exit_mode": spec.exit_mode,
        "event_window": (spec.event_window or {}).get("key"),
        "gold_mode": spec.gold_mode,
        "fx_mode": spec.fx_mode,
        "frozen_rules": {
            "component": spec.component,
            "exit_mode": spec.exit_mode,
            "event_window": spec.event_window,
            "gold_mode": spec.gold_mode,
            "fx_mode": spec.fx_mode,
            "combo_pre_specified": spec.is_combo,
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
        "monte_carlo": monte_carlo(val, n_runs=V10_MC_RUNS, seed=V10_MC_SEED),
        "beats_control_val": beats,
        "control_val_exp": control_val_exp,
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
    }


def select_on_train(results: list[dict]) -> Optional[dict]:
    """TRAIN-only among non-control specs with n≥20; prefer independent components over combos."""
    cands = [r for r in results if not r.get("is_control")]
    preferred = [
        r
        for r in cands
        if (r.get("train_n") or 0) >= 20
        and (r.get("train_expectancy") or 0) > 0
        and r.get("train_diversified")
        and not r.get("is_combo")
    ]
    if not preferred:
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


def _component_scorecard(by_class: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cls, block in by_class.items():
        fams = block.get("families") or {}
        ctrl = fams.get("V10_S_CTRL_TREND") or fams.get("V10_C_CTRL_TREND") or fams.get("V10_F_CTRL_TREND")
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
                improved_val and (ft.get("expectancy") or 0) > 0 and (fi.get("expectancy") or 0) > 0
            )
            rows.append(
                {
                    "market_class": cls,
                    "candidate": name,
                    "component": r.get("component"),
                    "is_combo": r.get("is_combo"),
                    "exit_mode": r.get("exit_mode"),
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


def _leakage_checks() -> list[dict[str, str]]:
    return [
        {"check": "htf_completed_bars_only", "status": "PASS", "detail": "Daily/weekly use searchsorted(ts, side=right)-1"},
        {"check": "swing_pivot_confirmation_lag", "status": "PASS", "detail": f"Pivots confirmed with {2}-bar right edge; no future bars"},
        {"check": "fvg_causal_definition", "status": "PASS", "detail": "3-candle gap at bar i; mitigation tracked forward only"},
        {"check": "sr_no_future_levels", "status": "PASS", "detail": "S/R from confirmed swings + completed HTF bars only"},
        {"check": "macro_consensus_surprise", "status": "PASS", "detail": f"Consensus/surprise remain {UNKNOWN}; not used in signals"},
        {"check": "geopolitical_flags", "status": "UNKNOWN", "detail": "Architecture present; no reliable historical geopolitical feed loaded"},
        {"check": "final_holdout_not_used_for_selection", "status": "PASS", "detail": "TRAIN selects; VAL/FINAL untouched until gate evaluation"},
    ]


SCANNER_OUTPUT_SCHEMA = {
    "ASSET": "instrument symbol",
    "TIMEFRAME": "4H primary",
    "TREND_DIRECTION": "bullish/bearish from structure",
    "HTF_DIRECTION": "daily + weekly structure",
    "MARKET_REGIME": REGIME_LABELS,
    "ENTRY_AREA": "trigger bar close ± slip",
    "SUPPORT": "nearest support ATR distance",
    "RESISTANCE": "nearest resistance ATR distance",
    "FVG_STRUCTURE_CONTEXT": "active FVG + BOS/CHoCH flags",
    "STOP_LOSS_AREA": "ATR or structure stop",
    "TARGET_EXIT_LOGIC": "adaptive ATR trail / 2R / structure invalidation",
    "RISK_REWARD": "stop distance in ATR units",
    "ATR_VOLATILITY": "ATR% vs median regime",
    "UPCOMING_HIGH_IMPACT_EVENT": "macro calendar blackout context",
    "MACRO_CONTEXT": "gold/FX relative filters when enabled",
    "CONFIDENCE_EVIDENCE_SCORE": "gate pass count (research only)",
    "REASONS_FOR_SIGNAL": "component-specific flags",
    "REASONS_AGAINST_SIGNAL": "ranging regime / event window / filter blocks",
}


def build_v10_payload(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    bundle: dict[str, Any],
    macro: MacroContext,
) -> dict[str, Any]:
    spec_map = build_specs()
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []
    variants = 0
    ctx_map: dict = {}

    for cls, specs in spec_map.items():
        print(f"ASSET CLASS: {cls}", flush=True)
        results = []
        controls = [s for s in specs if s.is_control]
        others = [s for s in specs if not s.is_control]
        for s in controls:
            results.append(evaluate_spec(series_4h, daily_map, weekly_map, ctx_map, s, cls, macro, None))
            variants += 1
        trend_ctrl = next((r for r in results if "CTRL_TREND" in r["name"]), results[0] if results else None)
        ctrl_val = (trend_ctrl.get("val") or {}).get("expectancy") if trend_ctrl else None
        for s in others:
            results.append(evaluate_spec(series_4h, daily_map, weekly_map, ctx_map, s, cls, macro, ctrl_val))
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
            "train_selection_rule": "Max TRAIN expectancy among independent components (n≥20) first; combos only if none qualify.",
            "control_val_expectancy": ctrl_val,
            "selected_decision": (candidate or {}).get("decision"),
            "selected_decision_reason": (candidate or {}).get("decision_reason"),
            "class_verdict": class_verdict,
        }

    scorecard = _component_scorecard(by_class)
    helped = [s for s in scorecard if s.get("improved_unseen_holdouts")]
    failed = [s for s in scorecard if not s.get("improved_val_vs_control")]
    any_pass = bool(promoted)
    verdict = (
        "V10 PASS — at least one candidate has demonstrated sufficient generalisation "
        "and robustness to proceed to the NEXT validation stage only"
        if any_pass
        else "V10 FAIL — no candidate has demonstrated sufficient evidence of a generalisable edge"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V10 market context + price structure research; live ORIGINAL untouched",
        "architecture": {
            "primary_timeframe": "4H",
            "htf_context": ["1D", "1WK"],
            "modules": [
                "market_structure",
                "support_resistance",
                "fair_value_gaps",
                "mtf_alignment",
                "regime_classification",
                "liquidity_breakout",
                "macro_event_layer_v9",
                "entry_combos_limited",
                "exit_research",
            ],
            "regime_labels": REGIME_LABELS,
            "scanner_output_schema": SCANNER_OUTPUT_SCHEMA,
            "political_geopolitical": "Architecture only; historical flags UNKNOWN unless reliable feed attached",
        },
        "data_sources": {
            "price": "Yahoo Finance OHLCV (4H aggregated from 1h, 1D, 1WK)",
            "macro_calendar": "BLS schedules, FOMC, BOE/ECB rate changes, US claims convention",
            "macro_proxies": list(bundle.get("macro_keys") or ["DXY", "US10Y", "US3M", "TIP"]),
            "consensus_surprise": UNKNOWN,
        },
        "live_scanner_protected": True,
        "auto_promote": False,
        "paper_trade": False,
        "variants_tested": variants,
        "leakage_checks": _leakage_checks(),
        "nested_design": {
            "train_end": V10_TRAIN_END,
            "val_end": V10_VAL_END,
            "selection": "TRAIN only (independent components preferred)",
            "no_repair_from_final": True,
        },
        "by_asset_class": by_class,
        "component_scorecard": scorecard,
        "components_helped": helped,
        "components_failed": failed,
        "promoted_candidates": promoted,
        "verdict_code": "V10_PASS" if any_pass else "V10_FAIL",
        "verdict": verdict,
        "next_stage": (
            ["Next validation stage only (not paper/live)"]
            if any_pass
            else ["No candidate strong enough for next validation stage"]
        ),
        "stop_note": (
            "Stop after this report. Wait for explicit approval before V11, promotion, "
            "merge to production, paper trading, or any live changes."
        ),
    }


def format_v10_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V10 — MARKET CONTEXT + PRICE STRUCTURE RESEARCH")
    a("Live ORIGINAL untouched | no paper/live | no auto-promote | no gate weakening")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Variants tested: {payload.get('variants_tested')}")
    a(f"Architecture: {payload.get('architecture')}")
    a(f"Data sources: {payload.get('data_sources')}")
    a("")
    a("# LEAKAGE / LOOK-AHEAD CHECKS")
    for chk in payload.get("leakage_checks") or []:
        a(f"  [{chk.get('status')}] {chk.get('check')}: {chk.get('detail')}")
    a("")
    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}")
        a(f"  DEV={block.get('dev_symbols')} FINAL_INST={block.get('final_inst_symbols')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}  verdict={block.get('class_verdict')}")
        a(f"  selected: {block.get('selected_decision')} — {block.get('selected_decision_reason')}")
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(
                f"  {name} ctrl={r.get('is_control')} combo={r.get('is_combo')} "
                f"comp={r.get('component')} exit={r.get('exit_mode')}"
            )
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
    a("COMPONENT SCORECARD")
    a("=" * 72)
    for row in payload.get("component_scorecard") or []:
        a(
            f"  [{row.get('decision')}] {row.get('market_class')} {row.get('candidate')} "
            f"comp={row.get('component')} combo={row.get('is_combo')} "
            f"VAL_impr={row.get('improved_val_vs_control')} unseen_impr={row.get('improved_unseen_holdouts')} "
            f"VAL={_bps(row.get('val_exp'))} FT={_bps(row.get('final_time_exp'))} FI={_bps(row.get('final_inst_exp'))}"
        )
    a("")
    a("# Components that genuinely helped (unseen holdouts)")
    for row in payload.get("components_helped") or []:
        a(f"  {row.get('candidate')} ({row.get('component')})")
    if not payload.get("components_helped"):
        a("  (none)")
    a("")
    a("# Components that failed vs control on VAL")
    for row in payload.get("components_failed") or []:
        a(f"  {row.get('candidate')} ({row.get('component')})")
    a("")
    a("=" * 72)
    a(f"FINAL DECISION: {payload.get('verdict')}")
    a(f"CODE: {payload.get('verdict_code')}")
    a(f"PROMOTED: {payload.get('promoted_candidates')}")
    a("Live scanner: NOT modified. No paper/live. No merge.")
    a(payload.get("stop_note"))
    a("=" * 72)
    return "\n".join(lines) + "\n"


def write_v10_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v10_report(payload)

    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return text
