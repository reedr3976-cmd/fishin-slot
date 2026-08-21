"""V12 extended-data replication report (frozen V11_S_FVG_SWEEP)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    V12_COMM_DEV,
    V12_COMM_FINAL_INST,
    V12_ENTRY_SLIP_ATR,
    V12_FX_DEV,
    V12_FX_FINAL_INST,
    V12_MC_RUNS,
    V12_MC_SEED,
    V12_N_FOLDS,
    V12_STOCK_DEV,
    V12_STOCK_FINAL_INST,
    V12_STOCK_ROTATIONS,
    V12_TRAIN_END,
    V12_VAL_END,
    V12_YAHOO_APPROX_START_ISO,
)
from backtest.frozen_specs import FROZEN_V11_S_FVG_SWEEP, V12Experiment, build_v12_experiments, cross_asset_replication_specs
from backtest.macro_features import MacroContext
from backtest.market_context_v11 import REGIME_V11_LABELS
from backtest.report_scanner_v11 import _direction_breakdown, _metrics_row, _regime_breakdown
from backtest.report_scanner_v8 import (
    _bps,
    _exp,
    _pct,
    _pf,
    by_symbol,
    decision_from_gate,
    gate_from,
    slice_metrics,
)
from backtest.scanner_v11 import V11Spec, folds_for_spec, run_spec_on_map
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import leave_out_symbols, monte_carlo


CLASS_UNIVERSE = {
    "stocks": {"dev": V12_STOCK_DEV, "final_inst": V12_STOCK_FINAL_INST, "rotations": V12_STOCK_ROTATIONS},
    "commodities": {
        "dev": V12_COMM_DEV,
        "final_inst": V12_COMM_FINAL_INST,
        "rotations": (("USOIL",), ("XAUUSD",), ("XAGUSD",)),
    },
    "forex": {"dev": V12_FX_DEV, "final_inst": V12_FX_FINAL_INST, "rotations": (("EURUSD",), ("GBPUSD",), ("USDJPY",))},
}


def _max_losing_streak(trades: list[V2Trade]) -> int:
    streak = best = 0
    for t in sorted(trades, key=lambda x: x.entry_ts):
        if not t.win:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _extended_metrics(trades: list[V2Trade]) -> dict[str, Any]:
    m = slice_metrics("x", trades)
    wins = [t.net_return for t in trades if t.win]
    losses = [t.net_return for t in trades if not t.win]
    return {
        **m,
        "n": len(trades),
        "avg_win": float(np.mean(wins)) if wins else None,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "max_losing_streak": _max_losing_streak(trades),
        "long_short": _direction_breakdown(trades),
        "regime": _regime_breakdown(trades),
    }


def _year_breakdown(trades: list[V2Trade]) -> list[dict[str, Any]]:
    buckets: dict[int, list[V2Trade]] = defaultdict(list)
    for t in trades:
        y = datetime.fromtimestamp(int(t.entry_ts), tz=timezone.utc).year
        buckets[y].append(t)
    rows = []
    for y in sorted(buckets):
        m = _extended_metrics(buckets[y])
        rows.append({"year": y, **{k: m.get(k) for k in ("n", "expectancy", "win_rate", "profit_factor", "max_drawdown")}})
    return rows


def _rolling_12m(trades: list[V2Trade]) -> list[dict[str, Any]]:
    if not trades:
        return []
    sorted_t = sorted(trades, key=lambda t: t.entry_ts)
    start_ts = sorted_t[0].entry_ts
    end_ts = sorted_t[-1].entry_ts
    rows = []
    t = start_ts
    while t < end_ts:
        window_end = t + 365 * 86400
        sub = [x for x in sorted_t if t <= x.entry_ts < window_end]
        if len(sub) >= 5:
            m = slice_metrics("w", sub)
            rows.append(
                {
                    "start": datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat(),
                    "n": len(sub),
                    "expectancy": m.get("expectancy"),
                    "win_rate": m.get("win_rate"),
                }
            )
        t += 180 * 86400
    return rows


def _instrument_table(trades: list[V2Trade]) -> list[dict[str, Any]]:
    by_inst: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        by_inst[t.instrument].append(t)
    rows = []
    for inst, sub in sorted(by_inst.items()):
        m = slice_metrics(inst, sub)
        rows.append(
            {
                "instrument": inst,
                "n": len(sub),
                "expectancy": m.get("expectancy"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
                "max_drawdown": m.get("max_drawdown"),
            }
        )
    return sorted(rows, key=lambda r: r.get("expectancy") or 0, reverse=True)


def _loo_rows(trades: list[V2Trade]) -> list[dict[str, Any]]:
    insts = sorted({t.instrument for t in trades})
    rows = []
    for inst in insts:
        sub = leave_out_symbols(trades, {inst})
        m = slice_metrics("loo", sub)
        rows.append({"held_out": inst, "n": len(sub), "expectancy": m.get("expectancy"), "positive": bool((m.get("expectancy") or 0) > 0)})
    return rows


def _bootstrap_ci(trades: list[V2Trade], *, n: int = 500, seed: int = 29) -> dict[str, Any]:
    if len(trades) < 10:
        return {"n_boot": 0, "note": "insufficient trades"}
    rng = random.Random(seed)
    rets = [t.net_return for t in trades]
    exps = []
    for _ in range(n):
        sample = [rng.choice(rets) for _ in range(len(rets))]
        exps.append(float(np.mean(sample)))
    exps.sort()
    lo = exps[int(0.025 * n)]
    hi = exps[int(0.975 * n)]
    return {"n_boot": n, "expectancy_mean": float(np.mean(exps)), "ci95_low": lo, "ci95_high": hi, "pct_positive": sum(1 for e in exps if e > 0) / n}


def _filter_trades_after(trades: list[V2Trade], iso: str) -> list[V2Trade]:
    cut = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    return [t for t in trades if t.entry_ts >= cut]


def _filter_trades_before(trades: list[V2Trade], iso: str) -> list[V2Trade]:
    cut = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    return [t for t in trades if t.entry_ts < cut]


def evaluate_experiment(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    ctx_map: dict,
    exp: V12Experiment,
    macro: Optional[MacroContext],
    *,
    market_class: Optional[str] = None,
) -> dict[str, Any]:
    cls = market_class or exp.market_class
    u = CLASS_UNIVERSE[cls]
    dev, final_inst = tuple(u["dev"]), tuple(u["final_inst"])
    spec = exp.spec
    if exp.key in ("E4", "E5"):
        spec = V11Spec(
            spec.key,
            spec.name,
            "commodities",
            spec.component,
            spec.rationale,
            baseline=spec.baseline,
            is_exit_variant=True,
            exit_mode=spec.exit_mode,
            entry_component=spec.entry_component,
        )

    kw = dict(daily_map=daily_map, weekly_map=weekly_map)
    train = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=0.0, end_frac=V12_TRAIN_END, **kw)
    val = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V12_TRAIN_END, end_frac=V12_VAL_END, **kw)
    final_time = run_spec_on_map(series_4h, spec, dev, ctx_map, macro, start_frac=V12_VAL_END, end_frac=1.0, **kw)
    final_inst = run_spec_on_map(
        series_4h, spec, final_inst, ctx_map, macro, start_frac=V12_TRAIN_END, end_frac=1.0, **kw
    )
    val_2x = run_spec_on_map(
        series_4h, spec, dev, ctx_map, macro, start_frac=V12_TRAIN_END, end_frac=V12_VAL_END, cost_mult=2.0, **kw
    )
    val_slip = run_spec_on_map(
        series_4h,
        spec,
        dev,
        ctx_map,
        macro,
        start_frac=V12_TRAIN_END,
        end_frac=V12_VAL_END,
        entry_slip_atr=V12_ENTRY_SLIP_ATR,
        **kw,
    )
    folds = folds_for_spec(
        series_4h, spec, dev, ctx_map, macro, daily_map=daily_map, weekly_map=weekly_map, n_folds=V12_N_FOLDS
    )
    fold_exps = [(slice_metrics(f"f{fr['fold']}", fr["trades"]).get("expectancy") or 0) for fr in folds]

    rotations, rot_pos = [], 0
    for held in u.get("rotations") or ():
        held = tuple(s for s in held if s in dev)
        if not held:
            continue
        rt = run_spec_on_map(
            series_4h, spec, held, ctx_map, macro, start_frac=V12_TRAIN_END, end_frac=V12_VAL_END, **kw
        )
        e = _exp(rt)
        ok = bool(e is not None and e > 0)
        rot_pos += int(ok)
        rotations.append({"held_out": list(held), "n": len(rt), "expectancy": e, "positive": ok})

    sens_pos = 0
    for ov in ({"max_hold": 20}, {"max_hold": 28}):
        t = run_spec_on_map(
            series_4h, spec, dev, ctx_map, macro, start_frac=V12_TRAIN_END, end_frac=V12_VAL_END, **{**kw, **ov}
        )
        sens_pos += int((_exp(t) or 0) > 0)

    val_exp = _exp(val)
    g = gate_from(val, val_2x, fold_exps, final_time, final_inst, sens_pos, 2, rot_pos, len(rotations), True)
    decision = decision_from_gate(g)

    all_oos = val + final_time + final_inst
    pre_yahoo = _filter_trades_before(all_oos, V12_YAHOO_APPROX_START_ISO)
    yahoo_era = _filter_trades_after(all_oos, V12_YAHOO_APPROX_START_ISO)

    near_miss = bool(
        decision["decision"] == "FAIL"
        and g.get("positive_val_oos")
        and (not g.get("positive_final_inst") or not g.get("positive_final_time") or not g.get("folds_ge_3"))
    )

    return {
        "experiment": exp.name,
        "phase": exp.phase,
        "market_class": cls,
        "frozen": exp.frozen,
        "spec": {
            "component": spec.component,
            "exit_mode": spec.exit_mode,
            "event_mode": spec.event_mode,
            "entry_component": spec.entry_component,
        },
        "train": _extended_metrics(train),
        "val": _extended_metrics(val),
        "val_2x": _extended_metrics(val_2x),
        "val_slip": _extended_metrics(val_slip),
        "final_time": _extended_metrics(final_time),
        "final_inst": _extended_metrics(final_inst),
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "near_miss": near_miss,
        "year_breakdown_val_plus_holdouts": _year_breakdown(all_oos),
        "rolling_12m_val_plus_holdouts": _rolling_12m(all_oos),
        "instrument_val": _instrument_table(val),
        "instrument_final_inst": _instrument_table(final_inst),
        "loo_val": _loo_rows(val),
        "bootstrap_val": _bootstrap_ci(val),
        "monte_carlo_val": monte_carlo(val, n_runs=V12_MC_RUNS, seed=V12_MC_SEED),
        "extended_history_confirmation": {
            "pre_yahoo_era_oos": _extended_metrics(pre_yahoo),
            "yahoo_era_oos": _extended_metrics(yahoo_era),
            "cutoff_iso": V12_YAHOO_APPROX_START_ISO,
        },
        "folds": [
            {"fold": fr["fold"], "metrics": slice_metrics(f"f{fr['fold']}", fr["trades"])} for fr in folds
        ],
        "instrument_rotations": rotations,
    }


def build_v12_payload(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    data_audit: dict,
    integrity: dict,
    source_provenance: dict,
    macro: Optional[MacroContext],
) -> dict[str, Any]:
    ctx_map: dict = {}
    experiments = build_v12_experiments()
    results: list[dict] = []

    for exp in experiments:
        print(f"EXPERIMENT {exp.name} ({exp.phase})", flush=True)
        if exp.key == "E1":
            for cls in ("stocks", "commodities", "forex"):
                results.append(evaluate_experiment(series_4h, daily_map, weekly_map, ctx_map, exp, macro, market_class=cls))
        else:
            results.append(evaluate_experiment(series_4h, daily_map, weekly_map, ctx_map, exp, macro))

    for spec in cross_asset_replication_specs():
        exp = V12Experiment("DX", spec.name, spec.market_class, spec, "diagnostic")
        results.append(evaluate_experiment(series_4h, daily_map, weekly_map, ctx_map, exp, macro))

    passed = [r for r in results if r.get("decision") == "PASS" and r.get("phase") != "diagnostic"]
    near = [r for r in results if r.get("near_miss")]
    failed = [r for r in results if r.get("decision") == "FAIL" and not r.get("near_miss")]

    primary = next((r for r in results if r["experiment"] == "V12_FROZEN_FVG_SWEEP" and r["market_class"] == "stocks"), None)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict_code": "V12_PASS" if passed else "V12_FAIL",
        "verdict": "V12 PASS — frozen candidate passed all gates on expanded data" if passed else "V12 FAIL — no frozen candidate passed all gates on expanded data",
        "frozen_specification": FROZEN_V11_S_FVG_SWEEP,
        "data_source_audit": data_audit,
        "data_integrity": integrity,
        "source_provenance": source_provenance,
        "experiment_registry": [{"key": e.key, "name": e.name, "phase": e.phase} for e in experiments],
        "results": results,
        "ranking": {"pass": passed, "near_miss": near, "failed": failed},
        "primary_replication_stocks": primary,
        "v11_yahoo_reference": {
            "V11_S_FVG_SWEEP_VAL_exp_bp": 7.94,
            "V11_S_FVG_SWEEP_FINAL_TIME_exp_bp": 12.73,
            "V11_S_FVG_SWEEP_FINAL_INST_exp_bp": -3.45,
            "note": "Reference only — V12 uses expanded Dukascopy timeline",
        },
        "live_scanner_protected": True,
        "promoted": False,
        "stop_note": "Stop for user approval. Do NOT merge to live or enable paper/live.",
    }


def format_v12_report(payload: dict[str, Any]) -> str:
    lines = ["=" * 72, "SCANNER V12 — DATA EXPANSION + FROZEN FVG_SWEEP REPLICATION", "=" * 72]
    a = lines.append
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Verdict: {payload.get('verdict')} ({payload.get('verdict_code')})")
    a("")
    a("# DATA SOURCE AUDIT")
    a(json.dumps(payload.get("data_source_audit"), indent=2)[:4000])
    a("")
    a("# FROZEN SPECIFICATION")
    a(json.dumps(payload.get("frozen_specification"), indent=2))
    a("")
    a("# INTEGRITY")
    a(f"all_ok={payload.get('data_integrity', {}).get('all_ok')} issues={payload.get('data_integrity', {}).get('instruments_with_issues')}")
    a("")
    for r in payload.get("results") or []:
        a("-" * 72)
        a(f"{r.get('experiment')} [{r.get('market_class')}] phase={r.get('phase')} decision={r.get('decision')} near_miss={r.get('near_miss')}")
        for k in ("train", "val", "final_time", "final_inst"):
            m = r.get(k) or {}
            a(
                f"  {k:11s} n={m.get('n',0):4d} win={_pct(m.get('win_rate'))} exp={_bps(m.get('expectancy'))} "
                f"PF={_pf(m.get('profit_factor'))} DD={_pct(m.get('max_drawdown'))} streak={m.get('max_losing_streak')}"
            )
        a(f"  gate: {r.get('gate')}")
        a(f"  reason: {r.get('decision_reason')}")
        a(f"  extended_confirmation: {r.get('extended_history_confirmation')}")
        a(f"  bootstrap_val: {r.get('bootstrap_val')}")
        if r.get("near_miss"):
            a(f"  instrument_final_inst: {r.get('instrument_final_inst')[:8]}")
            a(f"  loo_val: {r.get('loo_val')[:6]}")
    a("")
    a(f"RANKING pass={[x.get('experiment') for x in (payload.get('ranking') or {}).get('pass') or []]}")
    a(f"RANKING near_miss={[x.get('experiment') for x in (payload.get('ranking') or {}).get('near_miss') or []]}")
    a(payload.get("stop_note"))
    a("=" * 72)
    return "\n".join(lines) + "\n"


def write_v12_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v12_report(payload)

    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return text
