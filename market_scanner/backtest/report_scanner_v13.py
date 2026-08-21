"""V13 confirmation research report for frozen E3 (no optimisation)."""

from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    ROUND_TRIP_COST,
    V13_COST_MULTS,
    V13_ENTRY_SLIP_ATR,
    V13_ERA_PRE_YAHOO_END,
    V13_MC_RUNS,
    V13_MC_SEED,
    V13_N_FOLDS,
    V13_STOCK_ALL,
    V13_STOCK_DEV,
    V13_STOCK_FINAL_INST,
    V13_TRAIN_END,
    V13_VAL_END,
    V13_YAHOO_CUTOFF_ISO,
)
from backtest.causal_audit_e3 import run_causal_audit
from backtest.frozen_e3_spec import FROZEN_E3_DOCUMENT, FROZEN_E3_SPEC, FROZEN_E3_VERSION, frozen_e3_hash
from backtest.macro_features import MacroContext
from backtest.market_context_v11 import REGIME_V11_LABELS, clear_v11_cache
from backtest.report_scanner_v8 import _bps, _exp, _pct, _pf, gate_from, slice_metrics
from backtest.report_scanner_v11 import _direction_breakdown, _regime_breakdown
from backtest.scanner_v11 import folds_for_spec, run_spec_on_map
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import leave_out_symbols, monte_carlo
from backtest.scanner_v2 import rescale_cost
import backtest.market_context_v11 as mc11
import backtest.scanner_v11 as sv11


def _iso_ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _filter_ts(trades: list[V2Trade], start: Optional[float] = None, end: Optional[float] = None) -> list[V2Trade]:
    out = trades
    if start is not None:
        out = [t for t in out if t.entry_ts >= start]
    if end is not None:
        out = [t for t in out if t.entry_ts < end]
    return out


def _max_losing_streak(trades: list[V2Trade]) -> int:
    streak = best = 0
    for t in sorted(trades, key=lambda x: x.entry_ts):
        if not t.win:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _metrics(trades: list[V2Trade]) -> dict[str, Any]:
    m = slice_metrics("x", trades)
    wins = [t.net_return for t in trades if t.win]
    losses = [t.net_return for t in trades if not t.win]
    costs = [getattr(t, "cost", 0.0) or 0.0 for t in trades]
    gross = [t.net_return + (getattr(t, "cost", 0.0) or 0.0) for t in trades]
    return {
        **m,
        "n": len(trades),
        "expectancy": m.get("expectancy"),
        "avg_win": float(np.mean(wins)) if wins else None,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "max_losing_streak": _max_losing_streak(trades),
        "avg_cost": float(np.mean(costs)) if costs else None,
        "gross_expectancy": float(np.mean(gross)) if gross else None,
        "long_short": _direction_breakdown(trades),
        "regime": _regime_breakdown(trades),
    }


def _year_rows(trades: list[V2Trade]) -> list[dict[str, Any]]:
    buckets: dict[int, list[V2Trade]] = defaultdict(list)
    for t in trades:
        buckets[datetime.fromtimestamp(int(t.entry_ts), tz=timezone.utc).year].append(t)
    rows = []
    for y in sorted(buckets):
        m = _metrics(buckets[y])
        rows.append(
            {
                "year": y,
                "n": m["n"],
                "expectancy": m.get("expectancy"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
                "max_drawdown": m.get("max_drawdown"),
                "max_losing_streak": m.get("max_losing_streak"),
                "avg_win": m.get("avg_win"),
                "avg_loss": m.get("avg_loss"),
            }
        )
    return rows


def _instrument_table(trades: list[V2Trade]) -> list[dict[str, Any]]:
    by_inst: dict[str, list[V2Trade]] = defaultdict(list)
    for t in trades:
        by_inst[t.instrument].append(t)
    total_pnl = sum(t.net_return for t in trades) or 0.0
    rows = []
    for inst, sub in sorted(by_inst.items()):
        m = slice_metrics(inst, sub)
        pnl = sum(t.net_return for t in sub)
        rows.append(
            {
                "instrument": inst,
                "n": len(sub),
                "expectancy": m.get("expectancy"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
                "max_drawdown": m.get("max_drawdown"),
                "total_pnl": pnl,
                "pnl_share": (pnl / total_pnl) if total_pnl != 0 else None,
            }
        )
    return sorted(rows, key=lambda r: r.get("total_pnl") or 0, reverse=True)


def _concentration(trades: list[V2Trade]) -> dict[str, Any]:
    by_inst: dict[str, float] = defaultdict(float)
    for t in trades:
        by_inst[t.instrument] += t.net_return
    # Positive contribution only for concentration of edge
    pos = {k: v for k, v in by_inst.items() if v > 0}
    total_pos = sum(pos.values()) or 0.0
    ranked = sorted(pos.items(), key=lambda x: x[1], reverse=True)
    def share(k: int) -> Optional[float]:
        if not ranked or total_pos <= 0:
            return None
        return sum(v for _, v in ranked[:k]) / total_pos
    return {
        "best1_share_of_positive_pnl": share(1),
        "best2_share_of_positive_pnl": share(2),
        "best3_share_of_positive_pnl": share(3),
        "ranked_positive": [{"instrument": k, "pnl": v} for k, v in ranked],
        "instruments_positive": len(pos),
        "instruments_total": len(by_inst),
    }


def _loo(trades: list[V2Trade]) -> list[dict[str, Any]]:
    insts = sorted({t.instrument for t in trades})
    rows = []
    for inst in insts:
        sub = leave_out_symbols(trades, {inst})
        m = slice_metrics("loo", sub)
        rows.append(
            {
                "held_out": inst,
                "n": len(sub),
                "expectancy": m.get("expectancy"),
                "positive": bool((m.get("expectancy") or 0) > 0),
            }
        )
    return rows


def _bootstrap(trades: list[V2Trade], *, n: int = 1000, seed: int = 31) -> dict[str, Any]:
    if len(trades) < 10:
        return {"n_boot": 0, "note": "insufficient trades"}
    rng = random.Random(seed)
    rets = [t.net_return for t in trades]
    exps = []
    for _ in range(n):
        sample = [rng.choice(rets) for _ in range(len(rets))]
        exps.append(float(np.mean(sample)))
    exps.sort()
    return {
        "n_boot": n,
        "expectancy_mean": float(np.mean(exps)),
        "ci95_low": exps[int(0.025 * n)],
        "ci95_high": exps[int(0.975 * n)],
        "pct_positive": sum(1 for e in exps if e > 0) / n,
        "note": "Bootstrap resamples trade returns with replacement; not a proof of edge.",
    }


def _rolling_12m(trades: list[V2Trade]) -> list[dict[str, Any]]:
    if not trades:
        return []
    sorted_t = sorted(trades, key=lambda t: t.entry_ts)
    t0, t1 = sorted_t[0].entry_ts, sorted_t[-1].entry_ts
    rows = []
    t = t0
    while t < t1:
        w_end = t + 365 * 86400
        sub = [x for x in sorted_t if t <= x.entry_ts < w_end]
        if len(sub) >= 5:
            m = slice_metrics("w", sub)
            rows.append(
                {
                    "start": datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat(),
                    "n": len(sub),
                    "expectancy": m.get("expectancy"),
                    "win_rate": m.get("win_rate"),
                    "max_drawdown": m.get("max_drawdown"),
                }
            )
        t += 90 * 86400  # quarterly steps
    return rows


def _temporal_robustness(trades: list[V2Trade]) -> dict[str, Any]:
    years = _year_rows(trades)
    rolling = _rolling_12m(trades)
    pos_years = [y for y in years if (y.get("expectancy") or 0) > 0]
    neg_years = [y for y in years if (y.get("expectancy") or 0) <= 0]

    # Longest consecutive positive/negative years
    def longest_run(pred) -> int:
        best = cur = 0
        for y in years:
            if pred(y):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    # Underwater from equity curve
    eq = 0.0
    peak = 0.0
    underwater = 0
    max_under = 0
    for t in sorted(trades, key=lambda x: x.entry_ts):
        eq += t.net_return
        peak = max(peak, eq)
        if eq < peak:
            underwater += 1
            max_under = max(max_under, underwater)
        else:
            underwater = 0

    roll_exps = [r["expectancy"] for r in rolling if r.get("expectancy") is not None]
    return {
        "year_by_year": years,
        "longest_positive_year_streak": longest_run(lambda y: (y.get("expectancy") or 0) > 0),
        "longest_negative_year_streak": longest_run(lambda y: (y.get("expectancy") or 0) <= 0),
        "n_positive_years": len(pos_years),
        "n_negative_years": len(neg_years),
        "longest_underwater_trades": max_under,
        "worst_rolling_12m_expectancy": min(roll_exps) if roll_exps else None,
        "best_rolling_12m_expectancy": max(roll_exps) if roll_exps else None,
        "rolling_12m": rolling,
    }


def _eras_from_panel(series_4h: dict) -> list[dict[str, Any]]:
    """Build independent eras from available stock panel span."""
    ts_list = []
    for key in V13_STOCK_ALL:
        s = series_4h.get((key, "4h"))
        if s is not None and len(s):
            ts_list.append(int(s.timestamps[0]))
            ts_list.append(int(s.timestamps[-1]))
    if not ts_list:
        return []
    first, last = min(ts_list), max(ts_list)
    yahoo = _iso_ts(V13_YAHOO_CUTOFF_ISO)
    pre_end = yahoo
    span = max(pre_end - first, 1.0)
    t1 = first + span / 3
    t2 = first + 2 * span / 3
    eras = [
        {"name": "earliest", "start": first, "end": t1},
        {"name": "middle", "start": t1, "end": t2},
        {"name": "late_pre_yahoo", "start": t2, "end": pre_end},
        {"name": "pre_oct_2023", "start": first, "end": pre_end},
        {"name": "oct_2023_onward", "start": yahoo, "end": last + 1},
    ]
    for e in eras:
        e["start_iso"] = datetime.fromtimestamp(e["start"], tz=timezone.utc).date().isoformat()
        e["end_iso"] = datetime.fromtimestamp(e["end"], tz=timezone.utc).date().isoformat()
    return eras


def _run_all(
    series_4h,
    daily_map,
    weekly_map,
    ctx_map,
    macro,
    instruments,
    *,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
) -> list[V2Trade]:
    return run_spec_on_map(
        series_4h,
        FROZEN_E3_SPEC,
        instruments,
        ctx_map,
        macro,
        daily_map=daily_map,
        weekly_map=weekly_map,
        start_frac=0.0,
        end_frac=1.0,
        cost_mult=cost_mult,
        entry_slip_atr=entry_slip_atr,
    )


def _perturbation_battery(series_4h, daily_map, weekly_map, macro, instruments) -> dict[str, Any]:
    """Diagnostic only — does not replace frozen E3.

    Uses DEV instruments only for speed; stop-distance perturbations reuse context.
    """
    # Restrict to a fixed DEV subset for diagnostic speed (same frozen rules)
    from config import V13_STOCK_DEV

    diag_inst = [k for k in ("SPY", "QQQ", "AAPL", "GOOGL", "META") if k in V13_STOCK_DEV and (k, "4h") in series_4h]
    if len(diag_inst) < 3:
        diag_inst = [k for k in V13_STOCK_DEV if (k, "4h") in series_4h][:5]
    results = []
    baseline_ctx: dict = {}
    baseline = _run_all(series_4h, daily_map, weekly_map, baseline_ctx, macro, diag_inst)
    base_exp = _exp(baseline)

    # Context-affecting perturbations (minimal diagnostic set — not a search)
    ctx_perturbs = [
        ("htf_adx_weak", "V11_ADX_WEAK", mc11, 15.0),
        ("htf_adx_weak", "V11_ADX_WEAK", mc11, 21.0),
        ("pivot", "V11_PIVOT", mc11, 1),
        ("fvg_max_age", "V11_FVG_MAX_AGE", mc11, 20),
    ]
    originals = {attr: getattr(mod, attr) for _, attr, mod, _ in ctx_perturbs}
    # dedupe originals by attr
    originals = {}
    for _, attr, mod, _ in ctx_perturbs:
        originals[attr] = (mod, getattr(mod, attr))

    for label, attr, mod, value in ctx_perturbs:
        for a, (m, ov) in originals.items():
            setattr(m, a, ov)
        setattr(mod, attr, value)
        clear_v11_cache()
        print(f"    perturb {label}={value}...", flush=True)
        ctx: dict = {}
        trades = _run_all(series_4h, daily_map, weekly_map, ctx, macro, diag_inst)
        exp = _exp(trades)
        results.append(
            {
                "perturbation": f"{label}={value}",
                "n": len(trades),
                "expectancy": exp,
                "delta_vs_frozen": (exp - base_exp) if exp is not None and base_exp is not None else None,
                "positive": bool(exp is not None and exp > 0),
                "collapsed": bool(base_exp is not None and base_exp > 0 and (exp is None or exp <= 0)),
            }
        )

    # Restore context params
    for a, (m, ov) in originals.items():
        setattr(m, a, ov)
    clear_v11_cache()

    # Stop-distance perturbations — reuse baseline context (signal unchanged)
    stop_orig = sv11.V11_ATR_STOP_MULT
    for value in (1.25, 1.75):
        print(f"    perturb stop_atr_mult={value}...", flush=True)
        sv11.V11_ATR_STOP_MULT = value
        # Must not reuse ctx for stop changes? Signals identical — reuse baseline_ctx
        trades = run_spec_on_map(
            series_4h,
            FROZEN_E3_SPEC,
            diag_inst,
            baseline_ctx,
            macro,
            daily_map=daily_map,
            weekly_map=weekly_map,
            start_frac=0.0,
            end_frac=1.0,
        )
        # Re-simulate exits with new stop mult by clearing nothing but forcing backtest
        # backtest_spec reads V11_ATR_STOP_MULT at call time — OK
        exp = _exp(trades)
        results.append(
            {
                "perturbation": f"stop_atr_mult={value}",
                "n": len(trades),
                "expectancy": exp,
                "delta_vs_frozen": (exp - base_exp) if exp is not None and base_exp is not None else None,
                "positive": bool(exp is not None and exp > 0),
                "collapsed": bool(base_exp is not None and base_exp > 0 and (exp is None or exp <= 0)),
            }
        )
    sv11.V11_ATR_STOP_MULT = stop_orig

    return {
        "frozen_baseline_expectancy": base_exp,
        "frozen_baseline_n": len(baseline),
        "diagnostic_universe": diag_inst,
        "rows": results,
        "any_collapse": any(r.get("collapsed") for r in results),
        "note": "Diagnostic only on DEV universe. Frozen E3 remains the candidate regardless.",
    }


def _yahoo_overlap_independence(pre_metrics: dict, post_metrics: dict) -> dict[str, Any]:
    """Document independence options; report Dukascopy pre/post Yahoo eras (no paid key)."""
    return {
        "purpose": (
            "Independent multi-year US equity/ETF intraday for the SAME E3 instruments "
            "requires an API key or paid vendor. Not integrated per V13 policy."
        ),
        "dukascopy_pre_yahoo": pre_metrics,
        "dukascopy_yahoo_era": post_metrics,
        "yahoo_feed": {
            "skipped": True,
            "reason": (
                "Yahoo 1h depth ~730d overlaps only the recent era. Independent long history "
                "for like-for-like E3 names is not available without registration/API key."
            ),
        },
        "sign_agreement": None,
        "independent_long_history": {
            "available_without_key": False,
            "blocked_options": [
                {
                    "name": "HF Data Library",
                    "requires": "free account + email verification + API key",
                    "why": "Best free multi-year US equity/ETF 1-min history for like-for-like E3 instruments",
                    "url": "https://hfdatalibrary.com/",
                },
                {
                    "name": "EODHD Intraday",
                    "requires": "API key; freemium/paid tiers",
                    "url": "https://eodhd.com/",
                },
                {
                    "name": "Polygon / Norgate / Bloomberg",
                    "requires": "paid subscription",
                },
            ],
            "policy": "NOT integrated. STOP — awaiting user approval before any registration/API key/purchase.",
            "free_no_key_probe": "No usable free no-key multi-year US equity 1h feed found for E3 universe (Stooq/AV demo unusable).",
        },
    }


def build_v13_payload(
    series_4h: dict,
    daily_map: dict,
    weekly_map: dict,
    integrity: dict,
    provenance: dict,
    macro: Optional[MacroContext],
    *,
    git_commit: str = "",
) -> dict[str, Any]:
    clear_v11_cache()
    instruments = list(V13_STOCK_ALL)
    ctx_map: dict = {}

    print("  V13: full-history frozen E3 trades...", flush=True)
    all_trades = _run_all(series_4h, daily_map, weekly_map, ctx_map, macro, instruments)
    yahoo_cut = _iso_ts(V13_YAHOO_CUTOFF_ISO)
    pre = _filter_ts(all_trades, end=yahoo_cut)
    post = _filter_ts(all_trades, start=yahoo_cut)

    # Nested splits (same fractions as V12, for continuity)
    print("  V13: TRAIN/VAL/FINAL splits...", flush=True)
    train = run_spec_on_map(
        series_4h, FROZEN_E3_SPEC, V13_STOCK_DEV, ctx_map, macro,
        daily_map=daily_map, weekly_map=weekly_map, start_frac=0.0, end_frac=V13_TRAIN_END,
    )
    val = run_spec_on_map(
        series_4h, FROZEN_E3_SPEC, V13_STOCK_DEV, ctx_map, macro,
        daily_map=daily_map, weekly_map=weekly_map, start_frac=V13_TRAIN_END, end_frac=V13_VAL_END,
    )
    final_time = run_spec_on_map(
        series_4h, FROZEN_E3_SPEC, V13_STOCK_DEV, ctx_map, macro,
        daily_map=daily_map, weekly_map=weekly_map, start_frac=V13_VAL_END, end_frac=1.0,
    )
    final_inst = run_spec_on_map(
        series_4h, FROZEN_E3_SPEC, V13_STOCK_FINAL_INST, ctx_map, macro,
        daily_map=daily_map, weekly_map=weekly_map, start_frac=V13_TRAIN_END, end_frac=1.0,
    )

    print("  V13: cost stress...", flush=True)
    cost_rows = []
    # Simulate once at 1x then rescale (cost_r scales linearly)
    base_all = all_trades
    base_val = val
    for cm in V13_COST_MULTS:
        t = [rescale_cost(x, cm) for x in base_all] if cm != 1.0 else base_all
        t_val = [rescale_cost(x, cm) for x in base_val] if cm != 1.0 else base_val
        cost_rows.append(
            {
                "cost_mult": cm,
                "all_history": _metrics(t),
                "val_window": _metrics(t_val),
                "pre_yahoo": _metrics(_filter_ts(t, end=yahoo_cut)),
                "post_yahoo": _metrics(_filter_ts(t, start=yahoo_cut)),
            }
        )

    slip_trades = _run_all(
        series_4h, daily_map, weekly_map, {}, macro, instruments, entry_slip_atr=V13_ENTRY_SLIP_ATR
    )
    # Break-even approx: cost_mult where expectancy ~ 0 on all_history
    be = None
    for row in cost_rows:
        if (row["all_history"].get("expectancy") or 0) <= 0:
            be = row["cost_mult"]
            break

    print("  V13: eras...", flush=True)
    eras = _eras_from_panel(series_4h)
    era_rows = []
    for e in eras:
        sub = _filter_ts(all_trades, start=e["start"], end=e["end"])
        era_rows.append({"era": e["name"], "start_iso": e["start_iso"], "end_iso": e["end_iso"], **_metrics(sub)})

    print("  V13: perturbations (diagnostic)...", flush=True)
    pert = _perturbation_battery(series_4h, daily_map, weekly_map, macro, instruments)

    print("  V13: data-source independence note...", flush=True)
    independence = _yahoo_overlap_independence(_metrics(pre), _metrics(post))

    print("  V13: causal audit...", flush=True)
    sample_key = "SPY"
    sample_s = series_4h.get((sample_key, "4h"))
    sample_ctx = ctx_map.get(sample_key)
    causal = run_causal_audit(sample_s, sample_ctx)

    print("  V13: folds / MC / bootstrap...", flush=True)
    folds = folds_for_spec(
        series_4h, FROZEN_E3_SPEC, V13_STOCK_DEV, ctx_map, macro,
        daily_map=daily_map, weekly_map=weekly_map, n_folds=V13_N_FOLDS,
    )
    fold_exps = [(slice_metrics(f"f{fr['fold']}", fr["trades"]).get("expectancy") or 0) for fr in folds]
    val_2x = [rescale_cost(x, 2.0) for x in val]
    g = gate_from(val, val_2x, fold_exps, final_time, final_inst, 2, 2, 2, 3, True)

    boot_all = _bootstrap(all_trades, n=1000, seed=V13_MC_SEED)
    boot_pre = _bootstrap(pre, n=1000, seed=V13_MC_SEED)
    boot_val = _bootstrap(val, n=1000, seed=V13_MC_SEED)
    mc = monte_carlo(val, n_runs=V13_MC_RUNS, seed=V13_MC_SEED)

    inst_all = _instrument_table(all_trades)
    conc = _concentration(all_trades)
    loo = _loo(all_trades)
    temporal = _temporal_robustness(all_trades)

    # --- Decision logic (strict; do not weaken) ---
    pre_exp = _exp(pre)
    post_exp = _exp(post)
    look_ahead = bool(causal.get("look_ahead_detected"))
    era_dependent = bool(
        pre_exp is not None and pre_exp <= 0 and post_exp is not None and post_exp > 0
    )
    concentrated = bool(
        (conc.get("best1_share_of_positive_pnl") or 0) >= 0.70
        or (conc.get("best2_share_of_positive_pnl") or 0) >= 0.90
    )
    # Use cost_rows for 2x all-history (index of 2.0 in V13_COST_MULTS)
    cost_2x = next((r for r in cost_rows if r.get("cost_mult") == 2.0), None)
    cost_kills = bool(cost_2x is None or (cost_2x["all_history"].get("expectancy") or 0) <= 0)
    fragile = bool(pert.get("any_collapse"))
    independent_long_blocked = True  # no free no-key long independent source integrated

    fail_reasons = []
    if look_ahead:
        fail_reasons.append("Daily HTF look-ahead leakage detected (same-day OHLC available mid-day)")
    if era_dependent or (pre_exp is not None and pre_exp <= 0):
        fail_reasons.append(
            f"Pre-Oct-2023 confirmation materially non-positive (exp={pre_exp}, n={len(pre)})"
        )
    if concentrated:
        fail_reasons.append(
            f"Edge concentrated in few instruments (best1={conc.get('best1_share_of_positive_pnl')}, "
            f"best2={conc.get('best2_share_of_positive_pnl')})"
        )
    if cost_kills:
        fail_reasons.append("2× (or higher) transaction costs remove the edge on full history")
    if fragile:
        fail_reasons.append("Small parameter perturbations collapse expectancy to non-positive")
    if independent_long_blocked and era_dependent:
        fail_reasons.append(
            "Independent long-history confirmation unavailable without API key; "
            "era-dependence cannot be independently falsified"
        )
    # Technical V12-style gates alone are insufficient
    if not g.get("all_pass"):
        fail_reasons.append(f"Nested validation gates not all_pass: {g}")

    # Persistent edge across eras: require pre_yahoo positive AND post positive AND earliest positive
    earliest = next((r for r in era_rows if r["era"] == "earliest"), None)
    if earliest and (earliest.get("expectancy") or 0) <= 0:
        if "earliest-era expectancy non-positive" not in " ".join(fail_reasons):
            fail_reasons.append(
                f"Earliest-era expectancy non-positive (exp={earliest.get('expectancy')}, n={earliest.get('n')})"
            )

    verdict = "V13 FAIL" if fail_reasons else "V13 PASS"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict_code": "V13_PASS" if verdict == "V13 PASS" else "V13_FAIL",
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "frozen_specification": FROZEN_E3_DOCUMENT,
        "frozen_spec_hash": frozen_e3_hash(),
        "frozen_spec_version": FROZEN_E3_VERSION,
        "git_commit": git_commit,
        "data_integrity": integrity,
        "source_provenance": provenance,
        "reproducibility": {
            "primary_data_source": "dukascopy_bid_cfd",
            "yahoo_cutoff_iso": V13_YAHOO_CUTOFF_ISO,
            "timezone": "UTC",
            "aggregation": "1h→4h causal UTC buckets; daily/weekly from 4h",
            "spec_hash": frozen_e3_hash(),
            "spec_version": FROZEN_E3_VERSION,
            "git_commit": git_commit,
            "stock_round_trip_cost": ROUND_TRIP_COST.get("stock"),
        },
        "phase2_data_source_independence": independence,
        "phase3_pre_yahoo": {
            **_metrics(pre),
            "year_breakdown": _year_rows(pre),
            "instrument_table": _instrument_table(pre),
            "critical": True,
            "note": "CRITICAL confirmation — reported separately, not buried in averages.",
        },
        "phase3_post_yahoo": _metrics(post),
        "phase4_eras": era_rows,
        "phase5_instruments": {
            "all_history": inst_all,
            "concentration": conc,
            "leave_one_out": loo,
            "pre_yahoo_instruments": _instrument_table(pre),
        },
        "phase6_perturbations": pert,
        "phase7_cost_stress": {
            "rows": cost_rows,
            "entry_slippage_stress": _metrics(slip_trades),
            "approx_breakeven_cost_mult": be,
        },
        "phase8_statistics": {
            "bootstrap_all": boot_all,
            "bootstrap_pre_yahoo": boot_pre,
            "bootstrap_val": boot_val,
            "monte_carlo_val": mc,
            "year_by_year": temporal["year_by_year"],
        },
        "phase9_temporal": temporal,
        "phase10_causal_audit": causal,
        "nested_splits": {
            "train": _metrics(train),
            "val": _metrics(val),
            "final_time": _metrics(final_time),
            "final_inst": _metrics(final_inst),
            "gate": g,
        },
        "folds": [
            {"fold": fr["fold"], "metrics": slice_metrics(f"f{fr['fold']}", fr["trades"])} for fr in folds
        ],
        "live_scanner_protected": True,
        "promoted": False,
        "stop_note": "STOP for user approval. Do NOT merge to live or enable paper/live trading.",
    }


def format_v13_report(payload: dict[str, Any]) -> str:
    lines = ["=" * 72, "SCANNER V13 — FROZEN E3 CONFIRMATION RESEARCH (NO OPTIMISATION)", "=" * 72]
    a = lines.append
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Verdict: {payload.get('verdict')} ({payload.get('verdict_code')})")
    a(f"Spec version: {payload.get('frozen_spec_version')} hash={payload.get('frozen_spec_hash')}")
    a(f"Git commit: {payload.get('git_commit')}")
    if payload.get("fail_reasons"):
        a("")
        a("# FAIL REASONS")
        for r in payload["fail_reasons"]:
            a(f"  - {r}")
    a("")
    a("# PHASE 1 — FROZEN SPEC")
    a(json.dumps(payload.get("frozen_specification"), indent=2)[:3500])
    a("")
    a("# PHASE 2 — DATA-SOURCE INDEPENDENCE")
    ind = payload.get("phase2_data_source_independence") or {}
    a(f"  Yahoo overlap Dukascopy-era metrics: n={((ind.get('dukascopy_yahoo_era') or {}).get('n'))} "
      f"exp={_bps((ind.get('dukascopy_yahoo_era') or {}).get('expectancy'))}")
    a(f"  Yahoo feed metrics: n={((ind.get('yahoo_feed') or {}).get('n'))} "
      f"exp={_bps((ind.get('yahoo_feed') or {}).get('expectancy'))}")
    a(f"  Sign agreement: {ind.get('sign_agreement')}")
    a(f"  Long independent history: {json.dumps(ind.get('independent_long_history'), indent=2)}")
    a("")
    a("# PHASE 3 — PRE-YAHOO CONFIRMATION (CRITICAL)")
    pre = payload.get("phase3_pre_yahoo") or {}
    a(
        f"  n={pre.get('n')} win={_pct(pre.get('win_rate'))} exp={_bps(pre.get('expectancy'))} "
        f"PF={_pf(pre.get('profit_factor'))} DD={_pct(pre.get('max_drawdown'))} "
        f"streak={pre.get('max_losing_streak')} avg_win={_bps(pre.get('avg_win'))} avg_loss={_bps(pre.get('avg_loss'))}"
    )
    a(f"  long_short: {pre.get('long_short')}")
    a(f"  regime: {pre.get('regime')}")
    a(f"  year_breakdown: {pre.get('year_breakdown')}")
    a(f"  instruments: {pre.get('instrument_table')}")
    post = payload.get("phase3_post_yahoo") or {}
    a(
        f"  POST-Yahoo: n={post.get('n')} exp={_bps(post.get('expectancy'))} "
        f"win={_pct(post.get('win_rate'))} PF={_pf(post.get('profit_factor'))}"
    )
    a("")
    a("# PHASE 4 — ERA STABILITY")
    for e in payload.get("phase4_eras") or []:
        a(
            f"  {e.get('era'):16s} {e.get('start_iso')}→{e.get('end_iso')} "
            f"n={e.get('n')} exp={_bps(e.get('expectancy'))} win={_pct(e.get('win_rate'))} "
            f"PF={_pf(e.get('profit_factor'))} DD={_pct(e.get('max_drawdown'))}"
        )
    a("")
    a("# PHASE 5 — INSTRUMENT ROBUSTNESS")
    a(f"  concentration: {payload.get('phase5_instruments', {}).get('concentration')}")
    a(f"  instruments: {payload.get('phase5_instruments', {}).get('all_history')}")
    a(f"  LOO: {payload.get('phase5_instruments', {}).get('leave_one_out')}")
    a("")
    a("# PHASE 6 — PARAMETER FRAGILITY (diagnostic)")
    a(json.dumps(payload.get("phase6_perturbations"), indent=2)[:2500])
    a("")
    a("# PHASE 7 — COST / SLIPPAGE STRESS")
    for row in (payload.get("phase7_cost_stress") or {}).get("rows") or []:
        ah = row.get("all_history") or {}
        a(f"  cost×{row.get('cost_mult')}: n={ah.get('n')} exp={_bps(ah.get('expectancy'))} "
          f"pre={_bps((row.get('pre_yahoo') or {}).get('expectancy'))} "
          f"post={_bps((row.get('post_yahoo') or {}).get('expectancy'))}")
    slip = (payload.get("phase7_cost_stress") or {}).get("entry_slippage_stress") or {}
    a(f"  entry slip {V13_ENTRY_SLIP_ATR}×ATR: n={slip.get('n')} exp={_bps(slip.get('expectancy'))}")
    a(f"  approx breakeven cost_mult: {(payload.get('phase7_cost_stress') or {}).get('approx_breakeven_cost_mult')}")
    a("")
    a("# PHASE 8 — STATISTICS")
    a(f"  bootstrap_all: {payload.get('phase8_statistics', {}).get('bootstrap_all')}")
    a(f"  bootstrap_pre_yahoo: {payload.get('phase8_statistics', {}).get('bootstrap_pre_yahoo')}")
    a(f"  monte_carlo_val: {payload.get('phase8_statistics', {}).get('monte_carlo_val')}")
    a("")
    a("# PHASE 9 — TEMPORAL ROBUSTNESS")
    t = payload.get("phase9_temporal") or {}
    a(f"  longest_pos_years={t.get('longest_positive_year_streak')} "
      f"longest_neg_years={t.get('longest_negative_year_streak')} "
      f"max_underwater_trades={t.get('longest_underwater_trades')}")
    a(f"  best_roll12={_bps(t.get('best_rolling_12m_expectancy'))} "
      f"worst_roll12={_bps(t.get('worst_rolling_12m_expectancy'))}")
    a("")
    a("# PHASE 10 — CAUSAL / LOOK-AHEAD AUDIT")
    a(json.dumps(payload.get("phase10_causal_audit"), indent=2)[:3000])
    a("")
    a("# NESTED SPLITS (continuity with V12)")
    for k in ("train", "val", "final_time", "final_inst"):
        m = (payload.get("nested_splits") or {}).get(k) or {}
        a(f"  {k:11s} n={m.get('n')} exp={_bps(m.get('expectancy'))} win={_pct(m.get('win_rate'))}")
    a(f"  gate: {(payload.get('nested_splits') or {}).get('gate')}")
    a("")
    a(payload.get("stop_note"))
    a("=" * 72)
    return "\n".join(lines) + "\n"


def write_v13_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v13_report(payload)

    def _default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    return text
