"""V11 near-miss refinement reporting (research only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from config import (
    V11_COMM_DEV,
    V11_COMM_FINAL_INST,
    V11_ENTRY_SLIP_ATR,
    V11_FX_DEV,
    V11_FX_FINAL_INST,
    V11_MC_RUNS,
    V11_MC_SEED,
    V11_METALS_DEV,
    V11_METALS_FINAL_INST,
    V11_N_FOLDS,
    V11_STOCK_DEV,
    V11_STOCK_FINAL_INST,
    V11_STOCK_ROTATIONS,
    V11_TRAIN_END,
    V11_VAL_END,
)
from backtest.macro_features import MacroContext
from backtest.market_context_v11 import REGIME_V11_LABELS
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
from backtest.scanner_v11 import V11Spec, folds_for_spec, run_spec_on_map
from backtest.scanner_v2 import V2Trade
from providers.macro_calendar import UNKNOWN


def _liq_controls(market: str) -> list[V11Spec]:
    p = market[0].upper()
    return [
        V11Spec(
            f"{p}_CL",
            f"V11_{p}_CTRL_LIQ",
            market,
            "LIQ_BASE",
            "V10 LIQ_SWEEP baseline (ablation control).",
            baseline="liq",
            is_control=True,
        ),
        V11Spec(
            f"{p}_CM",
            f"V11_{p}_CTRL_MTF",
            market,
            "MTF_D_PB",
            "V10 MTF_DAILY baseline (ablation control).",
            baseline="mtf",
            is_control=True,
        ),
        V11Spec(
            f"{p}_CF",
            f"V11_{p}_CTRL_FVG",
            market,
            "FVG_FRESH",
            "V10 FVG_PULLBACK proxy — fresh FVG baseline.",
            baseline="fvg",
            is_control=True,
        ),
    ]


def _core_specs(market: str) -> list[V11Spec]:
    p = market[0].upper()
    return [
        V11Spec(f"{p}_LH", f"V11_{p}_LIQ_HTF", market, "LIQ_HTF", "Liquidity sweep + daily HTF structure.", baseline="liq"),
        V11Spec(f"{p}_LD", f"V11_{p}_LIQ_HTF_DW", market, "LIQ_HTF_DW", "Sweep + daily + weekly alignment.", baseline="liq"),
        V11Spec(f"{p}_LB", f"V11_{p}_LIQ_BOS", market, "LIQ_BOS", "Sweep then BOS within 3 bars.", baseline="liq"),
        V11Spec(f"{p}_LF", f"V11_{p}_LIQ_FVG", market, "LIQ_FVG", "HTF sweep + FVG pullback.", baseline="liq"),
        V11Spec(f"{p}_LS", f"V11_{p}_LIQ_SR", market, "LIQ_SR", "Sweep at HTF S/R zone.", baseline="liq"),
        V11Spec(
            f"{p}_LC",
            f"V11_{p}_LIQ_COMBO",
            market,
            "LIQ_COMBO",
            "HTF sweep + BOS after sweep.",
            baseline="liq",
            is_combo=True,
        ),
        V11Spec(f"{p}_MD", f"V11_{p}_MTF_D_PB", market, "MTF_D_PB", "Daily structure + 4H pullback.", baseline="mtf"),
        V11Spec(f"{p}_MW", f"V11_{p}_MTF_DW_PB", market, "MTF_DW_PB", "Daily + weekly + pullback.", baseline="mtf"),
        V11Spec(f"{p}_MB", f"V11_{p}_MTF_BOS", market, "MTF_BOS", "BOS in HTF direction.", baseline="mtf"),
        V11Spec(f"{p}_MF", f"V11_{p}_MTF_PB_FVG", market, "MTF_PB_FVG", "HTF + FVG pullback zone.", baseline="mtf"),
        V11Spec(f"{p}_MR", f"V11_{p}_MTF_PB_SR", market, "MTF_PB_SR", "HTF + strict S/R zone.", baseline="mtf"),
        V11Spec(f"{p}_FF", f"V11_{p}_FVG_FRESH", market, "FVG_FRESH", "Fresh unmitigated FVG + trend.", baseline="fvg"),
        V11Spec(f"{p}_FP", f"V11_{p}_FVG_PARTIAL", market, "FVG_PARTIAL", "Partially mitigated FVG entry.", baseline="fvg"),
        V11Spec(f"{p}_FH", f"V11_{p}_FVG_HTF", market, "FVG_HTF", "HTF-aligned FVG.", baseline="fvg"),
        V11Spec(f"{p}_FS", f"V11_{p}_FVG_SWEEP", market, "FVG_SWEEP", "FVG after liquidity sweep.", baseline="fvg"),
        V11Spec(f"{p}_FB", f"V11_{p}_FVG_BOS", market, "FVG_BOS", "FVG after BOS.", baseline="fvg"),
        V11Spec(f"{p}_SZ", f"V11_{p}_SR_ZONE", market, "SR_ZONE", "Strict S/R zone touch + rejection.", baseline="sr"),
        V11Spec(f"{p}_SF", f"V11_{p}_SR_FLIP", market, "SR_FLIP", "Breakout flip level retest.", baseline="sr"),
        V11Spec(
            f"{p}_SC",
            f"V11_{p}_SR_LIQ_FVG",
            market,
            "SR_LIQ_FVG",
            "S/R zone + liquidity or FVG confluence.",
            baseline="sr",
            is_combo=True,
        ),
        V11Spec(
            f"{p}_RL",
            f"V11_{p}_REG_LIQ",
            market,
            "LIQ_HTF",
            "LIQ_HTF gated to trend-eligible regime only.",
            baseline="liq",
            regime_gate="trend",
        ),
        V11Spec(
            f"{p}_RM",
            f"V11_{p}_REG_MTF",
            market,
            "MTF_D_PB",
            "MTF daily pullback gated to strong trend.",
            baseline="mtf",
            regime_gate="strong",
        ),
        V11Spec(
            f"{p}_EA",
            f"V11_{p}_EV_AVOID1H",
            market,
            "LIQ_HTF",
            "Avoid 1h around HIGH events on LIQ_HTF.",
            baseline="liq",
            event_mode="avoid_1h",
        ),
        V11Spec(
            f"{p}_EP",
            f"V11_{p}_EV_POST4H",
            market,
            "LIQ_HTF",
            "Trade first 4H bar after HIGH event only.",
            baseline="liq",
            event_mode="post_4h",
        ),
        V11Spec(
            f"{p}_XS",
            f"V11_{p}_EXIT_STRUCT",
            market,
            "LIQ_HTF",
            "Structure stop exit on LIQ_HTF entry.",
            baseline="liq",
            is_exit_variant=True,
            exit_mode="struct",
            entry_component="LIQ_HTF",
        ),
        V11Spec(
            f"{p}_XT",
            f"V11_{p}_EXIT_TRAIL",
            market,
            "LIQ_HTF",
            "Trailing structure stop on LIQ_HTF.",
            baseline="liq",
            is_exit_variant=True,
            exit_mode="trail_struct",
            entry_component="LIQ_HTF",
        ),
        V11Spec(
            f"{p}_XR",
            f"V11_{p}_EXIT_R2",
            market,
            "LIQ_HTF",
            "Fixed 2R target on LIQ_HTF.",
            baseline="liq",
            is_exit_variant=True,
            exit_mode="r2",
            entry_component="LIQ_HTF",
        ),
        V11Spec(
            f"{p}_XP",
            f"V11_{p}_EXIT_PARTIAL",
            market,
            "LIQ_HTF",
            "Partial 1R + breakeven trail on LIQ_HTF.",
            baseline="liq",
            is_exit_variant=True,
            exit_mode="partial_1r",
            entry_component="LIQ_HTF",
        ),
        V11Spec(
            f"{p}_XO",
            f"V11_{p}_EXIT_OPP_BOS",
            market,
            "LIQ_HTF",
            "Exit on opposite BOS/CHoCH.",
            baseline="liq",
            is_exit_variant=True,
            exit_mode="opp_bos",
            entry_component="LIQ_HTF",
        ),
    ]


def _comm_extra() -> list[V11Spec]:
    return [
        V11Spec(
            "M_GOLD",
            "V11_M_GOLD_NOT_TIGHT",
            "commodities",
            "LIQ_HTF",
            "Gold macro filter on LIQ_HTF (metals).",
            baseline="liq",
            gold_mode="not_tightening",
            universe_tag="metals",
        ),
    ]


def _fx_extra() -> list[V11Spec]:
    return [
        V11Spec(
            "F_CRY",
            "V11_F_MACRO_CARRY",
            "forex",
            "MTF_D_PB",
            "FX carry alignment on MTF daily entries.",
            baseline="mtf",
            fx_mode="carry_align",
        ),
    ]


def build_specs() -> dict[str, list[V11Spec]]:
    return {
        "stocks": _liq_controls("stocks") + _core_specs("stocks"),
        "commodities": _liq_controls("commodities") + _core_specs("commodities") + _comm_extra(),
        "forex": _liq_controls("forex") + _core_specs("forex") + _fx_extra(),
    }


CLASS_UNIVERSE = {
    "stocks": {"dev": V11_STOCK_DEV, "final_inst": V11_STOCK_FINAL_INST, "rotations": V11_STOCK_ROTATIONS},
    "commodities": {
        "dev": V11_COMM_DEV,
        "final_inst": V11_COMM_FINAL_INST,
        "rotations": (("USOIL",), ("XAUUSD",), ("XAGUSD",)),
    },
    "forex": {"dev": V11_FX_DEV, "final_inst": V11_FX_FINAL_INST, "rotations": (("EURUSD",), ("GBPUSD",), ("USDJPY",))},
}


def universe_for(spec: V11Spec, cls: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if spec.universe_tag == "metals":
        return V11_METALS_DEV, V11_METALS_FINAL_INST
    u = CLASS_UNIVERSE[cls]
    return tuple(u["dev"]), tuple(u["final_inst"])


def _direction_breakdown(trades: list[V2Trade]) -> dict[str, Any]:
    out = {}
    for d in ("bullish", "bearish"):
        sub = [t for t in trades if t.direction == d]
        m = slice_metrics(d, sub)
        out[d] = {
            "n": len(sub),
            "win_rate": m.get("win_rate"),
            "expectancy": m.get("expectancy"),
            "max_drawdown": m.get("max_drawdown"),
            "profit_factor": m.get("profit_factor"),
        }
    return out


def _regime_breakdown(trades: list[V2Trade]) -> dict[str, Any]:
    buckets: dict[str, list[V2Trade]] = {}
    for t in trades:
        code = int((t.feature_flags or {}).get("regime_v11", t.regime or 0))
        label = REGIME_V11_LABELS.get(code, f"code_{code}")
        buckets.setdefault(label, []).append(t)
    out = {}
    for label, sub in sorted(buckets.items()):
        m = slice_metrics(label, sub)
        out[label] = {"n": len(sub), "expectancy": m.get("expectancy"), "win_rate": m.get("win_rate")}
    return out


def _metrics_row(label: str, trades: list[V2Trade]) -> dict[str, Any]:
    m = slice_metrics(label, trades)
    return {
        "label": label,
        "n": len(trades),
        "win_rate": m.get("win_rate"),
        "expectancy": m.get("expectancy"),
        "profit_factor": m.get("profit_factor"),
        "max_drawdown": m.get("max_drawdown"),
        "long_short": _direction_breakdown(trades),
        "regime": _regime_breakdown(trades),
    }


def evaluate_spec(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    ctx_map: dict,
    spec: V11Spec,
    cls: str,
    macro: Optional[MacroContext],
    baselines: dict[str, float],
) -> dict[str, Any]:
    dev, final_inst_syms = universe_for(spec, cls)
    print(
        f"  {spec.name} comp={spec.component} exit={spec.exit_mode} "
        f"regime={spec.regime_gate} event={spec.event_mode}",
        flush=True,
    )
    kw = dict(daily_map=daily_map, weekly_map=weekly_map)
    train = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=0.0, end_frac=V11_TRAIN_END, **kw)
    val = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V11_TRAIN_END, end_frac=V11_VAL_END, **kw)
    final_time = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V11_VAL_END, end_frac=1.0, **kw)
    final_inst = run_spec_on_map(
        series_4h, spec, final_inst_syms, ctx_map, macro, start_frac=V11_TRAIN_END, end_frac=1.0, **kw
    )
    val_2x = run_spec_on_map(
        series_4h, spec, dev, ctx_map, macro, start_frac=V11_TRAIN_END, end_frac=V11_VAL_END, cost_mult=2.0, **kw
    )
    val_slip = run_spec_on_map(
        series_4h,
        spec,
        dev,
        ctx_map,
        macro,
        start_frac=V11_TRAIN_END,
        end_frac=V11_VAL_END,
        entry_slip_atr=V11_ENTRY_SLIP_ATR,
        **kw,
    )
    folds = folds_for_spec(
        series_4h, spec, dev, ctx_map, macro, daily_map=daily_map, weekly_map=weekly_map, n_folds=V11_N_FOLDS
    )
    fold_rows, fold_exps = [], []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append({"fold": fr["fold"], "metrics": m, "start_frac": fr["start_frac"], "end_frac": fr["end_frac"]})

    rotations, rot_pos = [], 0
    for held_group in CLASS_UNIVERSE[cls].get("rotations") or []:
        held_group = tuple(s for s in held_group if s in dev)
        if not held_group:
            continue
        rot_trades = run_spec_on_map(
            series_4h, spec, held_group, ctx_map, macro, start_frac=V11_TRAIN_END, end_frac=V11_VAL_END, **kw
        )
        e = _exp(rot_trades)
        ok = bool(e is not None and e > 0)
        rot_pos += int(ok)
        rotations.append({"held_out_instruments": list(held_group), "n": len(rot_trades), "expectancy": e, "positive": ok})

    sens, sens_pos = [], 0
    for ov in ({"max_hold": 20}, {"max_hold": 28}):
        t = run_spec_on_map(
            series_4h, spec, dev, ctx_map, macro, start_frac=V11_TRAIN_END, end_frac=V11_VAL_END, **{**kw, **ov}
        )
        e = _exp(t)
        ok = bool(e is not None and e > 0)
        sens_pos += int(ok)
        sens.append({"override": ov, "n": len(t), "expectancy": e, "positive": ok})

    val_sym = by_symbol(val)
    best = val_sym["best_symbols"]
    drop1 = leave_out_symbols(val, {best[0]} if best else set())
    val_exp = _exp(val)
    if spec.is_control:
        beats = True
        base_exp = None
    elif spec.is_exit_variant:
        base_exp = baselines.get("liq_htf") or baselines.get("liq")
        beats = bool(val_exp is not None and base_exp is not None and val_exp > base_exp)
    else:
        base_exp = baselines.get(spec.baseline)
        beats = bool(val_exp is not None and base_exp is not None and val_exp > base_exp)

    g = gate_from(val, val_2x, fold_exps, final_time, final_inst, sens_pos, len(sens), rot_pos, len(rotations), beats)
    decision = decision_from_gate(g)
    train_sym = by_symbol(train)
    return {
        "family_key": spec.key,
        "name": spec.name,
        "component": spec.component,
        "baseline": spec.baseline,
        "rationale": spec.rationale,
        "is_control": spec.is_control,
        "is_combo": spec.is_combo,
        "is_exit_variant": spec.is_exit_variant,
        "exit_mode": spec.exit_mode,
        "regime_gate": spec.regime_gate,
        "event_mode": spec.event_mode,
        "train": _metrics_row("train", train),
        "train_n": len(train),
        "train_expectancy": _exp(train),
        "train_diversified": len(train_sym["symbols_positive_ge3"]) >= 2
        and not ((train_sym.get("best1_share") or 0) >= 0.70),
        "val": _metrics_row("val", val),
        "val_2x": _metrics_row("2x", val_2x),
        "val_slip": _metrics_row("slip", val_slip),
        "final_time": _metrics_row("final_time", final_time),
        "final_inst": _metrics_row("final_inst", final_inst),
        "final_inst_by_symbol": by_symbol(final_inst),
        "folds": fold_rows,
        "period_concentration": period_concentration(fold_rows),
        "instrument_rotations": rotations,
        "by_symbol": val_sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "sensitivity": sens,
        "monte_carlo": monte_carlo(val, n_runs=V11_MC_RUNS, seed=V11_MC_SEED),
        "beats_baseline_val": beats,
        "baseline_val_exp": base_exp,
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "near_miss": bool(decision["decision"] == "FAIL" and g.get("positive_val_oos") and (
            not g.get("positive_final_time") or not g.get("positive_final_inst") or not g.get("folds_ge_3")
        )),
    }


def select_on_train(results: list[dict]) -> Optional[dict]:
    cands = [r for r in results if not r.get("is_control") and not r.get("is_exit_variant")]
    preferred = [
        r
        for r in cands
        if (r.get("train_n") or 0) >= 20
        and (r.get("train_expectancy") or 0) > 0
        and r.get("train_diversified")
        and r.get("baseline") == "liq"
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
    return max(ctrls, key=lambda r: r.get("train_expectancy") or float("-inf")) if ctrls else None


def _ablation_table(by_class: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cls, block in by_class.items():
        fams = block.get("families") or {}
        bases = {
            "liq": fams.get(f"V11_{cls[0].upper()}_CTRL_LIQ"),
            "mtf": fams.get(f"V11_{cls[0].upper()}_CTRL_MTF"),
            "fvg": fams.get(f"V11_{cls[0].upper()}_CTRL_FVG"),
        }
        for tag, base in bases.items():
            if not base:
                continue
            bval = (base.get("val") or {}).get("expectancy")
            for name, r in fams.items():
                if r.get("is_control") or r.get("baseline") != tag:
                    continue
                val = (r.get("val") or {}).get("expectancy")
                rows.append(
                    {
                        "market_class": cls,
                        "candidate": name,
                        "component": r.get("component"),
                        "baseline": tag,
                        "baseline_val_exp": bval,
                        "candidate_val_exp": val,
                        "delta_val_exp": (val - bval) if val is not None and bval is not None else None,
                        "improved_val": bool(val is not None and bval is not None and val > bval),
                        "final_time_exp": (r.get("final_time") or {}).get("expectancy"),
                        "final_inst_exp": (r.get("final_inst") or {}).get("expectancy"),
                        "decision": r.get("decision"),
                    }
                )
    return rows


def _rank_candidates(all_results: list[dict]) -> dict[str, list[dict]]:
    passed = [r for r in all_results if r.get("decision") == "PASS" and not r.get("is_control")]
    near = sorted(
        [r for r in all_results if r.get("near_miss") and not r.get("is_control")],
        key=lambda r: (
            int((r.get("gate") or {}).get("folds_positive_count") or 0),
            r.get("val", {}).get("expectancy") or 0,
        ),
        reverse=True,
    )
    failed = [r for r in all_results if r.get("decision") == "FAIL" and not r.get("near_miss") and not r.get("is_control")]
    return {"pass": passed, "near_miss": near, "failed": failed}


def build_v11_payload(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    bundle: dict[str, Any],
    macro: MacroContext,
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    spec_map = build_specs()
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []
    all_results: list[dict] = []
    variants = 0
    ctx_map: dict = {}

    for cls, specs in spec_map.items():
        print(f"ASSET CLASS: {cls}", flush=True)
        results = []
        controls = [s for s in specs if s.is_control]
        others = [s for s in specs if not s.is_control]
        baselines: dict[str, float] = {}
        for s in controls:
            r = evaluate_spec(series_4h, daily_map, weekly_map, ctx_map, s, cls, macro, baselines)
            results.append(r)
            variants += 1
            if s.baseline != "none":
                baselines[s.baseline] = (r.get("val") or {}).get("expectancy")
        for s in others:
            r = evaluate_spec(series_4h, daily_map, weekly_map, ctx_map, s, cls, macro, baselines)
            results.append(r)
            variants += 1
            if (
                s.component == "LIQ_HTF"
                and s.exit_mode == "atr"
                and s.regime_gate == "none"
                and s.event_mode == "none"
            ):
                baselines["liq_htf"] = (r.get("val") or {}).get("expectancy")
        all_results.extend(results)
        pick = select_on_train(results)
        candidate = pick
        class_verdict = "FAIL"
        if candidate and candidate.get("decision") == "PASS" and not candidate.get("is_control"):
            class_verdict = "PASS"
            promoted.append({"asset_class": cls, "family": candidate["name"], "reason": candidate.get("decision_reason")})
        by_class[cls] = {
            "dev_symbols": list(CLASS_UNIVERSE[cls]["dev"]),
            "final_inst_symbols": list(CLASS_UNIVERSE[cls]["final_inst"]),
            "families": {r["name"]: r for r in results},
            "train_selected_family": candidate["name"] if candidate else None,
            "selected_decision": (candidate or {}).get("decision"),
            "selected_decision_reason": (candidate or {}).get("decision_reason"),
            "class_verdict": class_verdict,
            "baselines_val_exp": baselines,
        }

    ablation = _ablation_table(by_class)
    ranking = _rank_candidates(all_results)
    any_pass = bool(promoted)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V11 near-miss refinement after V10 FAIL; live ORIGINAL untouched",
        "v10_near_miss_focus": ["LIQ_SWEEP", "MTF_DAILY", "FVG_PULLBACK"],
        "data_quality_audit": data_audit,
        "architecture": {
            "primary_tf": "4H",
            "htf": ["1D", "1WK"],
            "modules": [
                "liquidity_sweep_htf",
                "mtf_structure",
                "fvg_states",
                "strict_sr_zones",
                "regime_v11",
                "macro_event_timing",
                "exit_research",
            ],
            "regime_labels": REGIME_V11_LABELS,
            "speaker_architecture": "Fed/BOE/ECB/political flags reserved; historical feed UNKNOWN",
        },
        "data_sources": {
            "price_4h": "Yahoo 1h aggregated (730d) — see data_quality_audit",
            "daily_weekly": "Yahoo 1D/1WK",
            "macro": "V9 calendar + DXY/TNX/IRX/TIP",
            "consensus_surprise": UNKNOWN,
        },
        "live_scanner_protected": True,
        "auto_promote": False,
        "paper_trade": False,
        "variants_tested": variants,
        "nested_design": {"train_end": V11_TRAIN_END, "val_end": V11_VAL_END, "selection": "TRAIN only"},
        "by_asset_class": by_class,
        "component_ablation": ablation,
        "ranking": ranking,
        "promoted_candidates": promoted,
        "verdict_code": "V11_PASS" if any_pass else "V11_FAIL",
        "verdict": (
            "V11 PASS — at least one candidate passed all gates"
            if any_pass
            else "V11 FAIL — no candidate passed all validation gates"
        ),
        "stop_note": "Stop after this report. Wait for explicit approval before promotion, merge, paper, or live.",
    }


def format_v11_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V11 — NEAR-MISS REFINEMENT RESEARCH")
    a("Live untouched | gates unchanged | no paper/live")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Variants: {payload.get('variants_tested')}")
    a(f"Data audit: {payload.get('data_quality_audit')}")
    a("")
    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}  verdict={block.get('class_verdict')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}")
        a(f"  baselines VAL exp: {block.get('baselines_val_exp')}")
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(f"  {name} [{r.get('decision')}] near_miss={r.get('near_miss')} comp={r.get('component')} exit={r.get('exit_mode')}")
            for key in ("train", "val", "final_time", "final_inst"):
                m = r.get(key) or {}
                a(
                    f"    {key:11s} n={m.get('n',0):4d} win={_pct(m.get('win_rate'))} "
                    f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} DD={_pct(m.get('max_drawdown'))}"
                )
            a(f"    long/short VAL: {(r.get('val') or {}).get('long_short')}")
            a(f"    regime VAL: {(r.get('val') or {}).get('regime')}")
            a(f"    folds: {[ (fr.get('metrics') or {}).get('expectancy') for fr in r.get('folds') or [] ]}")
            a(f"    reason: {r.get('decision_reason')}")
        a("")
    a("=" * 72)
    a("COMPONENT ABLATION (vs family baseline on VAL)")
    a("=" * 72)
    for row in payload.get("component_ablation") or []:
        a(
            f"  {row.get('candidate')} base={row.get('baseline')} "
            f"impr={row.get('improved_val')} delta={_bps(row.get('delta_val_exp'))} "
            f"VAL={_bps(row.get('candidate_val_exp'))} FT={_bps(row.get('final_time_exp'))} FI={_bps(row.get('final_inst_exp'))}"
        )
    a("")
    a("RANKING")
    a(f"  PASS: {[x.get('name') for x in (payload.get('ranking') or {}).get('pass') or []]}")
    a(f"  NEAR-MISS: {[x.get('name') for x in (payload.get('ranking') or {}).get('near_miss') or []]}")
    a("")
    a(f"FINAL: {payload.get('verdict')} ({payload.get('verdict_code')})")
    a(payload.get("stop_note"))
    a("=" * 72)
    return "\n".join(lines) + "\n"


def write_v11_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v11_report(payload)

    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return text
