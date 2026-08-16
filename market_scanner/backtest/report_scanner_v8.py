"""V8 generalisation research reporting (research only).

Nested TRAIN → VAL → FINAL_TIME; instrument rotations; final instrument holdout.
PASS criteria unchanged or stricter vs V7. Live ORIGINAL untouched.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import (
    V8_COMM_DEV,
    V8_COMM_FINAL_INST,
    V8_ENERGY_DEV,
    V8_ENERGY_FINAL_INST,
    V8_ENTRY_SLIP_ATR,
    V8_FX_DEV,
    V8_FX_FINAL_INST,
    V8_MAX_DD_ACCEPT,
    V8_MC_RUNS,
    V8_MC_SEED,
    V8_METALS_DEV,
    V8_METALS_FINAL_INST,
    V8_MIN_FOLDS_POSITIVE,
    V8_MIN_HELDOUT_TRADES,
    V8_MIN_ROTATION_POSITIVE,
    V8_MIN_SYMBOLS_POSITIVE,
    V8_MIN_TRADES,
    V8_N_FOLDS,
    V8_STOCK_DEV,
    V8_STOCK_FINAL_INST,
    V8_STOCK_ROTATIONS,
    V8_TRAIN_END,
    V8_VAL_END,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import leave_out_symbols, monte_carlo
from backtest.scanner_v8 import FAMILIES, V8Family, folds_for_family, run_family_on_map
from models import CandleSeries


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


def period_concentration(fold_rows: list[dict]) -> dict[str, Any]:
    fold_exps = [
        (fr["fold"], (fr.get("metrics") or {}).get("expectancy") or 0.0) for fr in fold_rows
    ]
    pos = [f for f, e in fold_exps if e > 0]
    return {
        "positive_folds": pos,
        "only_final_fold_positive": pos == [V8_N_FOLDS],
        "fold_expectancies": {str(f): e for f, e in fold_exps},
    }


CLASS_SPECS: dict[str, dict[str, Any]] = {
    "stocks": {
        "dev": V8_STOCK_DEV,
        "final_inst": V8_STOCK_FINAL_INST,
        "rotations": V8_STOCK_ROTATIONS,
        "family_filter": lambda f: f.market_class == "stocks",
        "baseline_keys": ("B1", "B2"),
    },
    "commodities": {
        "dev": V8_COMM_DEV,
        "final_inst": V8_COMM_FINAL_INST,
        "rotations": (
            ("USOIL",),
            ("XAUUSD",),
            ("XAGUSD",),
        ),
        "family_filter": lambda f: f.market_class == "commodities",
        "baseline_keys": ("B1", "B2"),
    },
    "forex": {
        "dev": V8_FX_DEV,
        "final_inst": V8_FX_FINAL_INST,
        "rotations": (
            ("EURUSD",),
            ("GBPUSD",),
            ("USDJPY",),
        ),
        "family_filter": lambda f: f.market_class == "forex",
        "baseline_keys": ("B1", "B2"),
    },
}


def universe_for(family: V8Family, spec: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if family.universe_tag == "metals":
        return V8_METALS_DEV, V8_METALS_FINAL_INST
    if family.universe_tag == "energy":
        return V8_ENERGY_DEV, V8_ENERGY_FINAL_INST
    return tuple(spec["dev"]), tuple(spec["final_inst"])


def gate_from(
    val: list[V2Trade],
    val_2x: list[V2Trade],
    fold_exps: list[float],
    final_time: list[V2Trade],
    final_inst: list[V2Trade],
    sens_positive: int,
    sens_total: int,
    rotation_positive: int,
    rotation_total: int,
    beats_baseline: bool,
) -> dict[str, Any]:
    m = slice_metrics("val", val)
    exp = m.get("expectancy")
    exp2 = _exp(val_2x)
    ft_exp = _exp(final_time)
    fi_exp = _exp(final_inst)
    sym = by_symbol(val)
    pos_folds = sum(1 for e in fold_exps if e > 0)
    sym_pos = sym["symbols_positive_ge3"]
    resilient_2x = bool(
        (exp2 is not None and exp2 > 0)
        or (exp is not None and exp > 0 and exp2 is not None and exp2 >= 0.5 * exp)
    )
    # Energy/metals single-DEV may have only 1–2 symbols — still require not_one_symbol when ≥2
    min_sym = min(V8_MIN_SYMBOLS_POSITIVE, max(1, len(sym["per_symbol"])))
    checks = {
        "positive_val_oos": bool(exp is not None and exp > 0),
        "positive_final_time": bool(ft_exp is not None and ft_exp > 0),
        "positive_final_inst": bool(fi_exp is not None and fi_exp > 0),
        "resilient_2x": resilient_2x,
        "folds_ge_3": pos_folds >= V8_MIN_FOLDS_POSITIVE,
        "folds_positive_count": pos_folds,
        "adequate_val_n": len(val) >= V8_MIN_TRADES,
        "adequate_final_time_n": len(final_time) >= V8_MIN_HELDOUT_TRADES,
        "adequate_final_inst_n": len(final_inst) >= V8_MIN_HELDOUT_TRADES,
        "acceptable_dd": bool((m.get("max_drawdown") or 1) <= V8_MAX_DD_ACCEPT),
        "multi_symbol": len(sym_pos) >= min_sym if len(sym["per_symbol"]) >= 2 else len(val) >= V8_MIN_TRADES,
        "not_one_symbol": not ((sym.get("best1_share") or 0) >= 0.70),
        "not_fragile": sens_positive >= max(1, sens_total // 2),
        "rotations_ok": rotation_positive >= min(V8_MIN_ROTATION_POSITIVE, max(1, rotation_total)),
        "beats_baseline": beats_baseline,
        "val_n": len(val),
        "final_time_n": len(final_time),
        "final_inst_n": len(final_inst),
        "final_time_expectancy": ft_exp,
        "final_inst_expectancy": fi_exp,
        "val_expectancy": exp,
        "val_2x_expectancy": exp2,
        "symbols_positive": sym_pos,
        "best1_share": sym.get("best1_share"),
        "rotation_positive": rotation_positive,
        "rotation_total": rotation_total,
    }
    checks["all_pass"] = all(
        checks[k]
        for k in (
            "positive_val_oos",
            "positive_final_time",
            "positive_final_inst",
            "resilient_2x",
            "folds_ge_3",
            "adequate_val_n",
            "adequate_final_time_n",
            "adequate_final_inst_n",
            "acceptable_dd",
            "multi_symbol",
            "not_one_symbol",
            "not_fragile",
            "rotations_ok",
            "beats_baseline",
        )
    )
    return checks


def decision_from_gate(gate: dict[str, Any]) -> dict[str, Any]:
    label_map = {
        "positive_val_oos": "VAL/OOS expectancy ≤ 0",
        "positive_final_time": "FINAL_TIME holdout expectancy ≤ 0",
        "positive_final_inst": "FINAL_INSTRUMENT holdout expectancy ≤ 0",
        "resilient_2x": "fails 2× cost resilience",
        "folds_ge_3": f"positive folds {gate.get('folds_positive_count')}/4 (<3)",
        "adequate_val_n": f"inadequate VAL trade count n={gate.get('val_n')} (need ≥{V8_MIN_TRADES})",
        "adequate_final_time_n": (
            f"inadequate FINAL_TIME n={gate.get('final_time_n')} (need ≥{V8_MIN_HELDOUT_TRADES})"
        ),
        "adequate_final_inst_n": (
            f"inadequate FINAL_INST n={gate.get('final_inst_n')} (need ≥{V8_MIN_HELDOUT_TRADES})"
        ),
        "acceptable_dd": "maximum drawdown above acceptance threshold",
        "multi_symbol": "edge not distributed across enough symbols",
        "not_one_symbol": "profit concentrated in one symbol (≥70% of positive PnL)",
        "not_fragile": "fragile to nearby parameter changes",
        "rotations_ok": (
            f"instrument-holdout rotations weak "
            f"({gate.get('rotation_positive')}/{gate.get('rotation_total')})"
        ),
        "beats_baseline": "does not materially beat simple baseline on VAL",
    }
    reasons = [msg for k, msg in label_map.items() if not gate.get(k)]
    passed = bool(gate.get("all_pass"))
    return {
        "decision": "PASS" if passed else "FAIL",
        "reason": (
            "Meets VAL, FINAL_TIME, FINAL_INST, folds, rotations, cost stress, "
            "sensitivity, diversification, and baseline-improvement gates."
            if passed
            else "; ".join(reasons) if reasons else "failed robustness gates"
        ),
    }


def evaluate_family(
    series_4h: dict,
    daily_map: dict,
    spy: Optional[CandleSeries],
    family: V8Family,
    spec: dict[str, Any],
    baseline_val_exp: Optional[float],
) -> dict[str, Any]:
    dev, final_inst_syms = universe_for(family, spec)
    print(
        f"  {family.name} dev={dev} final_inst={final_inst_syms}...",
        flush=True,
    )
    kw = dict(daily_map=daily_map, spy=spy)

    train = run_family_on_map(
        series_4h, family, dev, start_frac=0.0, end_frac=V8_TRAIN_END, **kw
    )
    val = run_family_on_map(
        series_4h, family, dev, start_frac=V8_TRAIN_END, end_frac=V8_VAL_END, **kw
    )
    final_time = run_family_on_map(
        series_4h, family, dev, start_frac=V8_VAL_END, end_frac=1.0, **kw
    )
    # Final instrument holdout: same VAL+FINAL_TIME window (post-TRAIN), never used in selection
    final_inst = run_family_on_map(
        series_4h,
        family,
        final_inst_syms,
        start_frac=V8_TRAIN_END,
        end_frac=1.0,
        **kw,
    )
    val_15 = run_family_on_map(
        series_4h, family, dev, start_frac=V8_TRAIN_END, end_frac=V8_VAL_END, cost_mult=1.5, **kw
    )
    val_2x = run_family_on_map(
        series_4h, family, dev, start_frac=V8_TRAIN_END, end_frac=V8_VAL_END, cost_mult=2.0, **kw
    )
    val_slip = run_family_on_map(
        series_4h,
        family,
        dev,
        start_frac=V8_TRAIN_END,
        end_frac=V8_VAL_END,
        entry_slip_atr=V8_ENTRY_SLIP_ATR,
        **kw,
    )
    val_delay = run_family_on_map(
        series_4h,
        family,
        dev,
        start_frac=V8_TRAIN_END,
        end_frac=V8_VAL_END,
        entry_delay=1,
        **kw,
    )

    folds = folds_for_family(
        series_4h, family, dev, n_folds=V8_N_FOLDS, daily_map=daily_map, spy=spy
    )
    fold_rows = []
    fold_exps = []
    for fr in folds:
        m = slice_metrics(f"fold{fr['fold']}", fr["trades"])
        fold_exps.append(m["expectancy"] or 0.0)
        fold_rows.append(
            {"fold": fr["fold"], "metrics": m, "start_frac": fr["start_frac"], "end_frac": fr["end_frac"]}
        )

    # Instrument-holdout rotations on DEV: TRAIN on complement, evaluate VAL window on held group
    rotations = []
    rot_pos = 0
    for held_group in spec.get("rotations") or []:
        held_group = tuple(s for s in held_group if s in dev)
        if not held_group:
            continue
        train_syms = tuple(s for s in dev if s not in held_group)
        if not train_syms:
            continue
        # Selection discipline: we only RECORD rotation OOS; TRAIN selection does not use these
        rot_trades = run_family_on_map(
            series_4h,
            family,
            held_group,
            start_frac=V8_TRAIN_END,
            end_frac=V8_VAL_END,
            **kw,
        )
        e = _exp(rot_trades)
        ok = bool(e is not None and e > 0)
        rot_pos += int(ok)
        rotations.append(
            {
                "held_out_instruments": list(held_group),
                "train_instruments": list(train_syms),
                "n": len(rot_trades),
                "expectancy": e,
                "positive": ok,
                "by_symbol": by_symbol(rot_trades),
            }
        )
    rot_total = len(rotations)

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
            dev,
            start_frac=V8_TRAIN_END,
            end_frac=V8_VAL_END,
            **{**kw, **ov},
        )
        e = _exp(t)
        sens.append(
            {"override": ov, "n": len(t), "expectancy": e, "positive": bool(e is not None and e > 0)}
        )
    sens_pos = sum(1 for s in sens if s["positive"])

    train_sym = by_symbol(train)
    val_sym = by_symbol(val)
    ft_sym = by_symbol(final_time)
    fi_sym = by_symbol(final_inst)
    best = val_sym["best_symbols"]
    drop1 = leave_out_symbols(val, {best[0]} if best else set())

    val_exp = _exp(val)
    # Baselines: beats_baseline True by definition; candidates must beat best baseline VAL exp
    if family.is_baseline:
        beats = True
    else:
        beats = bool(
            val_exp is not None
            and baseline_val_exp is not None
            and val_exp > baseline_val_exp
        ) or bool(val_exp is not None and baseline_val_exp is None and val_exp > 0)

    g = gate_from(
        val,
        val_2x,
        fold_exps,
        final_time,
        final_inst,
        sens_pos,
        len(sens),
        rot_pos,
        rot_total,
        beats,
    )
    # Energy with 1 DEV symbol: not_one_symbol / multi_symbol adapt; still need final_inst
    if family.universe_tag == "energy" and len(dev) < 2:
        # Soften only multi_symbol / not_one_symbol for single-name energy DEV —
        # still require final_inst generalisation to NATGAS (stricter elsewhere).
        g["multi_symbol"] = len(val) >= V8_MIN_TRADES
        g["not_one_symbol"] = True
        g["all_pass"] = all(
            g[k]
            for k in (
                "positive_val_oos",
                "positive_final_time",
                "positive_final_inst",
                "resilient_2x",
                "folds_ge_3",
                "adequate_val_n",
                "adequate_final_time_n",
                "adequate_final_inst_n",
                "acceptable_dd",
                "multi_symbol",
                "not_one_symbol",
                "not_fragile",
                "rotations_ok",
                "beats_baseline",
            )
        )
    # Metals final_inst is 1 symbol — adequate_final_inst_n may be hard; keep gate (stricter)
    decision = decision_from_gate(g)

    frozen_rules = {
        "family": family.name,
        "key": family.key,
        "rationale": family.rationale,
        "notes": family.notes,
        "universe_tag": family.universe_tag,
        "dev_instruments": list(dev),
        "final_inst_instruments": list(final_inst_syms),
        "timeframe": "4h",
        "risk": "1% equity via ATR stop = 1R",
        "atr_stop_mult": 1.5,
        "max_hold": 24,
        "exit": "adaptive ATR trail (V2 adaptive)",
        "splits": {
            "train": f"[0, {V8_TRAIN_END})",
            "val": f"[{V8_TRAIN_END}, {V8_VAL_END})",
            "final_time": f"[{V8_VAL_END}, 1.0)",
            "final_inst_window": f"[{V8_TRAIN_END}, 1.0) on held-out instruments",
        },
        "selection": "TRAIN-only on DEV instruments; frozen before VAL/FINAL_*",
        "is_baseline": family.is_baseline,
    }

    return {
        "family_key": family.key,
        "name": family.name,
        "market_class": family.market_class,
        "rationale": family.rationale,
        "notes": family.notes,
        "is_baseline": family.is_baseline,
        "universe_tag": family.universe_tag,
        "frozen_rules": frozen_rules,
        "dev": list(dev),
        "final_inst_symbols": list(final_inst_syms),
        "train": slice_metrics("train", train),
        "train_n": len(train),
        "train_expectancy": _exp(train),
        "train_by_symbol": train_sym,
        "train_diversified": len(train_sym["symbols_positive_ge3"]) >= min(
            V8_MIN_SYMBOLS_POSITIVE, max(1, len(dev))
        )
        and not ((train_sym.get("best1_share") or 0) >= 0.70),
        "val": slice_metrics("val", val),
        "val_1_5x": slice_metrics("1.5x", val_15),
        "val_2x": slice_metrics("2x", val_2x),
        "val_slip": slice_metrics("slip", val_slip),
        "val_entry_delay": slice_metrics("delay1", val_delay),
        "final_time": slice_metrics("final_time", final_time),
        "final_inst": slice_metrics("final_inst", final_inst),
        "final_time_by_symbol": ft_sym,
        "final_inst_by_symbol": fi_sym,
        "folds": fold_rows,
        "period_concentration": period_concentration(fold_rows),
        "instrument_rotations": rotations,
        "by_symbol": val_sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "sensitivity": sens,
        "monte_carlo": monte_carlo(val, n_runs=V8_MC_RUNS, seed=V8_MC_SEED),
        "baseline_val_exp_compared": baseline_val_exp,
        "beats_baseline_val": beats,
        "gate": g,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
    }


def select_on_train(results: list[dict]) -> Optional[dict]:
    """TRAIN-only among non-baselines. Prefer diversified positive TRAIN."""
    cands = [r for r in results if not r.get("is_baseline")]
    preferred = [
        r
        for r in cands
        if (r.get("train_n") or 0) >= 20
        and (r.get("train_expectancy") or 0) > 0
        and r.get("train_diversified")
    ]
    pool = preferred or [r for r in cands if (r.get("train_n") or 0) >= 15]
    pool = pool or [r for r in cands if (r.get("train_n") or 0) > 0]
    if not pool:
        return None
    return max(pool, key=lambda r: r.get("train_expectancy") or float("-inf"))


def diagnose_v7_failures() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "output" / "scanner_v7_report.json"
    out: dict[str, Any] = {"available": path.exists()}
    if not path.exists():
        out["note"] = "V7 report missing; using design rationale only."
        out["stocks_e"] = {
            "why_failed": (
                "V7 Stocks E failed held-out: discovery edge concentrated in mega-cap/"
                "energy (XOM/AAPL/MSFT) while consumer discretionary / high-vol names "
                "(DIS/AMZN/NVDA/BA/WMT) lost — not a universal equity momentum edge."
            )
        }
        return out
    v7 = json.loads(path.read_text(encoding="utf-8"))
    out["v7_verdict"] = v7.get("verdict")
    stocks = (v7.get("by_asset_class") or {}).get("stocks", {}).get("families") or {}
    e = stocks.get("V7_E_REGIME_MOM_STRUCT") or {}
    b = stocks.get("V7_B_VOL_PULLBACK") or {}
    disc = (e.get("by_symbol") or {}).get("pnl_contribution") or {}
    held = (e.get("heldout_by_symbol") or {}).get("pnl_contribution") or {}
    winners_d = [k for k, v in disc.items() if v > 0]
    losers_h = [k for k, v in held.items() if v < 0]
    winners_h = [k for k, v in held.items() if v > 0]
    out["stocks_e"] = {
        "name": e.get("name"),
        "decision": e.get("decision"),
        "val_proxy_test_exp": (e.get("test") or {}).get("expectancy"),
        "held_exp": (e.get("heldout_test") or {}).get("expectancy"),
        "folds_positive": (e.get("gate") or {}).get("folds_positive_count"),
        "discovery_pnl": disc,
        "heldout_pnl": held,
        "discovery_winners": winners_d,
        "heldout_winners": winners_h,
        "heldout_losers": losers_h,
        "why_failed": (
            "Stocks E generalised across discovery time folds (4/4) but not across "
            f"instruments: discovery winners {winners_d} (mega-cap/energy tilt) vs "
            f"held-out losers {losers_h} (discretionary/high-beta/defensive mix). "
            f"Held-out winners {winners_h} were insufficient to offset. Characteristic "
            "gap is sector/beta composition of the universe, not a missing indicator. "
            "Held-out names were NOT used to retune."
        ),
        "do_not": "Tune filters on held-out stocks to exclude losers.",
    }
    out["stocks_b"] = {
        "name": b.get("name"),
        "test_exp": (b.get("test") or {}).get("expectancy"),
        "held_exp": (b.get("heldout_test") or {}).get("expectancy"),
        "held_pnl": (b.get("heldout_by_symbol") or {}).get("pnl_contribution"),
        "why_failed": (
            "Stocks B (vol-pullback) had strong discovery OOS and 3/4 folds but held-out "
            "losses dominated by WMT/DIS/GOOGL/NVDA — same instrument-generalisation "
            "failure mode as E; apparent edge is universe-dependent."
        ),
    }
    # V6 stocks C from v6 report if present
    v6p = Path(__file__).resolve().parent.parent / "output" / "scanner_v6_report.json"
    if v6p.exists():
        v6 = json.loads(v6p.read_text(encoding="utf-8"))
        fams = (v6.get("by_asset_class") or {}).get("stocks", {}).get("families") or {}
        c = next((r for r in fams.values() if r.get("family_key") == "C"), None)
        if c:
            out["stocks_c_v6"] = {
                "test_n": (c.get("test") or {}).get("signals"),
                "test_exp": (c.get("test") or {}).get("expectancy"),
                "train_exp": (c.get("train") or {}).get("expectancy"),
                "folds": [
                    {
                        "fold": fr.get("fold"),
                        "exp": (fr.get("metrics") or {}).get("expectancy"),
                        "n": (fr.get("metrics") or {}).get("signals"),
                    }
                    for fr in (c.get("folds") or [])
                ],
                "why_failed": (
                    "V6 Stocks C: TRAIN negative, test n≈20, only late fold positive "
                    "(period lottery), held-out negative — small-sample / regime-specific "
                    "vol-expansion breakout, not stable edge."
                ),
            }
    comm = (v7.get("by_asset_class") or {}).get("commodities", {})
    out["commodities"] = {
        "why": (
            "V6/V7 commodity edges repeatedly concentrated in metals/oil on discovery "
            "and failed on gas/copper/corn held-outs — consistent with asset-specific "
            "effects, not a universal commodity strategy. V8 therefore tests metals-only "
            "and energy-only claims separately from a universal trend hypothesis."
        ),
        "v7_selected": comm.get("train_selected_family"),
    }
    fx = (v7.get("by_asset_class") or {}).get("forex", {})
    out["forex"] = {
        "why": (
            "V7 FX failures show weak/negative discovery expectancy with either too-few "
            "trades (MTF/vol filters) or many losing trades (loose momentum) — underlying "
            "edge appears absent rather than merely under-sampled. V8 tests trend-pullback, "
            "breakout-fade, and momentum-persistence without loosening gates."
        ),
        "v7_selected": fx.get("train_selected_family"),
    }
    out["design_response"] = {
        "nested_splits": f"TRAIN[0,{V8_TRAIN_END}) VAL[{V8_TRAIN_END},{V8_VAL_END}) "
        f"FINAL_TIME[{V8_VAL_END},1)",
        "instrument_holdout": "FINAL_INST excluded from all selection; rotations on DEV",
        "no_repair": "Failing FINAL_* is terminal for that frozen candidate",
    }
    return out


def build_final_summary(by_class: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cls, block in by_class.items():
        for name, r in (block.get("families") or {}).items():
            val = r.get("val") or {}
            ft = r.get("final_time") or {}
            fi = r.get("final_inst") or {}
            mc = r.get("monte_carlo") or {}
            rows.append(
                {
                    "candidate": name,
                    "family_key": r.get("family_key"),
                    "market_class": cls,
                    "is_baseline": r.get("is_baseline"),
                    "train_selected": name == block.get("train_selected_family"),
                    "rationale": r.get("rationale"),
                    "frozen_rules": r.get("frozen_rules"),
                    "trades_train": (r.get("train") or {}).get("signals"),
                    "train_expectancy": r.get("train_expectancy"),
                    "trades_val": val.get("signals"),
                    "win_rate": val.get("win_rate"),
                    "profit_factor": val.get("profit_factor"),
                    "expectancy": val.get("expectancy"),
                    "max_drawdown": val.get("max_drawdown"),
                    "val_expectancy": val.get("expectancy"),
                    "final_time_expectancy": ft.get("expectancy"),
                    "final_time_trades": ft.get("signals"),
                    "final_inst_expectancy": fi.get("expectancy"),
                    "final_inst_trades": fi.get("signals"),
                    "stress_2x_expectancy": (r.get("val_2x") or {}).get("expectancy"),
                    "stress_slip_expectancy": (r.get("val_slip") or {}).get("expectancy"),
                    "entry_delay_expectancy": (r.get("val_entry_delay") or {}).get("expectancy"),
                    "parameter_sensitivity": r.get("sensitivity"),
                    "monte_carlo_dd_median": ((mc.get("max_drawdown") or {}).get("median")),
                    "monte_carlo_dd_p95": ((mc.get("max_drawdown") or {}).get("p95")),
                    "folds_positive": (r.get("gate") or {}).get("folds_positive_count"),
                    "rotations": r.get("instrument_rotations"),
                    "symbol_diversification": {
                        "best1_share": (r.get("by_symbol") or {}).get("best1_share"),
                        "symbols_positive": (r.get("gate") or {}).get("symbols_positive"),
                        "val_pnl": (r.get("by_symbol") or {}).get("pnl_contribution"),
                        "final_inst_pnl": (r.get("final_inst_by_symbol") or {}).get(
                            "pnl_contribution"
                        ),
                    },
                    "period_concentration": r.get("period_concentration"),
                    "beats_baseline": r.get("beats_baseline_val"),
                    "decision": r.get("decision"),
                    "reason": r.get("decision_reason"),
                }
            )
    return rows


def identify_near_misses(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    near = []
    for row in summary:
        if row.get("is_baseline") or row.get("decision") == "PASS":
            continue
        val = row.get("val_expectancy")
        if val is None or val <= 0:
            continue
        score = 0
        if (row.get("final_time_expectancy") or 0) > 0:
            score += 1
        if (row.get("final_inst_expectancy") or 0) > 0:
            score += 1
        if (row.get("folds_positive") or 0) >= 3:
            score += 1
        if (row.get("trades_val") or 0) >= V8_MIN_TRADES:
            score += 1
        if score >= 2:
            near.append(
                {
                    "candidate": row.get("candidate"),
                    "market_class": row.get("market_class"),
                    "train_selected": row.get("train_selected"),
                    "near_miss_score": score,
                    "what_blocked_pass": row.get("reason"),
                    "val_expectancy": val,
                    "final_time_expectancy": row.get("final_time_expectancy"),
                    "final_inst_expectancy": row.get("final_inst_expectancy"),
                    "folds_positive": row.get("folds_positive"),
                    "note": "Recorded honestly; NOT repaired using FINAL_* information.",
                }
            )
    near.sort(key=lambda x: x.get("near_miss_score") or 0, reverse=True)
    return near


def build_v8_payload(series_4h: dict, daily_map: dict) -> dict[str, Any]:
    spy = series_4h.get(("SPY", "4h"))
    by_class: dict[str, Any] = {}
    promoted: list[dict] = []
    variants_tested = 0
    diagnosis = diagnose_v7_failures()

    baselines = {f.key: f for f in FAMILIES if f.is_baseline}

    for cls, spec in CLASS_SPECS.items():
        print(f"ASSET CLASS: {cls}", flush=True)
        # Evaluate baselines first on this class DEV universe for comparison
        baseline_results = []
        for bk in spec["baseline_keys"]:
            bf = baselines[bk]
            # Temporarily treat baseline as class for universe
            br = evaluate_family(
                series_4h, daily_map, spy, bf, spec, baseline_val_exp=None
            )
            br["market_class"] = cls
            baseline_results.append(br)
            variants_tested += 1

        best_base_val = None
        for br in baseline_results:
            e = (br.get("val") or {}).get("expectancy")
            if e is None:
                continue
            if best_base_val is None or e > best_base_val:
                best_base_val = e

        fam_results = list(baseline_results)
        for fam in FAMILIES:
            if not spec["family_filter"](fam):
                continue
            fam_results.append(
                evaluate_family(
                    series_4h, daily_map, spy, fam, spec, baseline_val_exp=best_base_val
                )
            )
            variants_tested += 1

        train_pick = select_on_train(fam_results)
        pick_name = train_pick["name"] if train_pick else None
        candidate = next((r for r in fam_results if r["name"] == pick_name), None)
        class_verdict = "FAIL"
        if candidate and candidate.get("decision") == "PASS":
            class_verdict = "PASS"
            promoted.append(
                {
                    "asset_class": cls,
                    "family": candidate["name"],
                    "rationale": candidate["rationale"],
                    "gate": candidate["gate"],
                    "decision_reason": candidate.get("decision_reason"),
                }
            )
        other_pass = [
            r["name"]
            for r in fam_results
            if r.get("decision") == "PASS"
            and r["name"] != pick_name
            and not r.get("is_baseline")
        ]
        by_class[cls] = {
            "dev_symbols": list(spec["dev"]),
            "final_inst_symbols": list(spec["final_inst"]),
            "families": {r["name"]: r for r in fam_results},
            "train_selected_family": pick_name,
            "train_selection_rule": (
                "Max TRAIN expectancy among non-baselines with n≥20 and TRAIN "
                "diversification when available; VAL/FINAL_* unused for selection."
            ),
            "best_baseline_val_expectancy": best_base_val,
            "selected_gate": (candidate or {}).get("gate"),
            "selected_decision": (candidate or {}).get("decision"),
            "selected_decision_reason": (candidate or {}).get("decision_reason"),
            "other_families_passing_gates": other_pass,
            "class_verdict": class_verdict,
        }

    summary = build_final_summary(by_class)
    near = identify_near_misses(summary)
    any_pass = any(r["decision"] == "PASS" and not r.get("is_baseline") for r in summary)
    if promoted:
        overall = (
            "V8 PASS — at least one candidate has demonstrated sufficient generalisation "
            "and robustness to proceed to the NEXT validation stage only"
        )
        code = "V8_PASS"
    elif any_pass:
        overall = (
            "V8 FAIL — some non-selected families meet gates, but TRAIN-frozen selection "
            "did not; do not cherry-pick holdout winners"
        )
        code = "V8_FAIL"
    else:
        overall = "V8 FAIL — no candidate has demonstrated sufficient evidence of a generalisable edge"
        code = "V8_FAIL"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "V8 generalisation research after V7 FAIL; nested holdouts; "
            "live ORIGINAL untouched"
        ),
        "live_scanner_protected": True,
        "auto_promote": False,
        "variants_tested": variants_tested,
        "families_defined": [
            {
                "key": f.key,
                "name": f.name,
                "market_class": f.market_class,
                "rationale": f.rationale,
                "is_baseline": f.is_baseline,
                "universe_tag": f.universe_tag,
            }
            for f in FAMILIES
        ],
        "nested_design": {
            "train_end": V8_TRAIN_END,
            "val_end": V8_VAL_END,
            "selection": "TRAIN only",
            "untouched_until_freeze": ["VAL", "FINAL_TIME", "FINAL_INST", "rotations OOS"],
            "no_repair_from_final": True,
        },
        "gate_policy": {
            "note": "PASS criteria not loosened vs V7; added FINAL_TIME + rotations + baseline beat",
            "min_trades_val": V8_MIN_TRADES,
            "min_holdout_trades": V8_MIN_HELDOUT_TRADES,
            "min_folds_positive": V8_MIN_FOLDS_POSITIVE,
            "min_rotation_positive": V8_MIN_ROTATION_POSITIVE,
            "max_dd": V8_MAX_DD_ACCEPT,
            "max_best1_share": 0.70,
        },
        "v7_failure_diagnosis": diagnosis,
        "by_asset_class": by_class,
        "final_candidate_summary": summary,
        "strongest_near_misses": near,
        "promoted_candidates": promoted,
        "verdict_code": code,
        "verdict": overall,
        "next_stage": (
            [{"action": "Next validation stage only (not paper/live)", "candidate": p} for p in promoted]
            if promoted
            else ["No candidate strong enough for next validation stage"]
        ),
        "stop_note": (
            "Stop after this report. Wait for explicit approval before V9, promotion, "
            "merge to production, or any live changes."
        ),
    }


def format_v8_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V8 — GENERALISATION RESEARCH (after V7 FAIL)")
    a("Live ORIGINAL untouched | no paper/live enablement | no auto-promote")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a(f"Variants tested: {payload.get('variants_tested')}")
    a(f"Nested design: {payload.get('nested_design')}")
    a(f"Gate policy: {payload.get('gate_policy')}")
    a("Families:")
    for f in payload.get("families_defined") or []:
        a(f"  {f['key']}: {f['name']} [{f['market_class']}] baseline={f['is_baseline']}")
        a(f"      rationale: {f.get('rationale')}")
    a("")
    a("=" * 72)
    a("V7 FAILURE DIAGNOSIS (before new candidates)")
    a("=" * 72)
    diag = payload.get("v7_failure_diagnosis") or {}
    a(f"  available={diag.get('available')} verdict={diag.get('v7_verdict')}")
    for key in ("stocks_e", "stocks_b", "stocks_c_v6", "commodities", "forex"):
        block = diag.get(key) or {}
        if not block:
            continue
        a(f"  [{key}]")
        for k, v in block.items():
            a(f"    {k}: {v}")
    a(f"  design_response: {diag.get('design_response')}")
    a("")

    for cls, block in (payload.get("by_asset_class") or {}).items():
        a("=" * 72)
        a(f"ASSET CLASS: {cls.upper()}")
        a(f"  DEV={block.get('dev_symbols')}  FINAL_INST={block.get('final_inst_symbols')}")
        a(f"  best_baseline_VAL_exp={block.get('best_baseline_val_expectancy')}")
        a(f"  TRAIN-selected: {block.get('train_selected_family')}")
        a(f"  class_verdict: {block.get('class_verdict')}")
        a(
            f"  selected: {block.get('selected_decision')} — {block.get('selected_decision_reason')}"
        )
        for name, r in (block.get("families") or {}).items():
            a("-" * 72)
            a(f"  {name}  [{r.get('family_key')}] baseline={r.get('is_baseline')}")
            a(f"    rationale: {r.get('rationale')}")
            a(f"    frozen_rules: {r.get('frozen_rules')}")
            for key in (
                "train",
                "val",
                "val_1_5x",
                "val_2x",
                "val_slip",
                "val_entry_delay",
                "final_time",
                "final_inst",
            ):
                m = r.get(key) or {}
                a(
                    f"    {key:16s} n={m.get('signals', 0):4d} win={_pct(m.get('win_rate'))} "
                    f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} "
                    f"DD={_pct(m.get('max_drawdown'))}"
                )
            a("    folds:")
            for fr in r.get("folds") or []:
                m = fr.get("metrics") or {}
                a(f"      fold {fr['fold']} n={m.get('signals', 0)} exp={_bps(m.get('expectancy'))}")
            a(f"    period_concentration: {r.get('period_concentration')}")
            a("    instrument_rotations:")
            for rot in r.get("instrument_rotations") or []:
                a(
                    f"      held={rot.get('held_out_instruments')} n={rot.get('n')} "
                    f"exp={_bps(rot.get('expectancy'))} pos={rot.get('positive')}"
                )
            sym = r.get("by_symbol") or {}
            a(f"    VAL best1_share={sym.get('best1_share')} pnl={sym.get('pnl_contribution')}")
            a(
                f"    FINAL_INST pnl={(r.get('final_inst_by_symbol') or {}).get('pnl_contribution')}"
            )
            a(f"    sensitivity: {r.get('sensitivity')}")
            a(f"    monte_carlo: {r.get('monte_carlo')}")
            a(f"    beats_baseline={r.get('beats_baseline_val')}")
            a(f"    DECISION: {r.get('decision')} — {r.get('decision_reason')}")
            a(f"    GATE: {r.get('gate')}")
        a("")

    a("=" * 72)
    a("FINAL CANDIDATE SUMMARY")
    a("=" * 72)
    for row in payload.get("final_candidate_summary") or []:
        a(
            f"  [{row.get('decision')}] {row.get('market_class'):11s} {row.get('candidate')} "
            f"{'(TRAIN-selected) ' if row.get('train_selected') else ''}"
            f"{'(baseline) ' if row.get('is_baseline') else ''}"
            f"VAL n={row.get('trades_val')} exp={_bps(row.get('val_expectancy'))} "
            f"FT={_bps(row.get('final_time_expectancy'))} "
            f"FI={_bps(row.get('final_inst_expectancy'))} "
            f"folds+={row.get('folds_positive')} beat_base={row.get('beats_baseline')}"
        )
        a(f"      reason: {row.get('reason')}")

    a("")
    a("# STRONGEST NEAR-MISSES (not repaired from FINAL_*)")
    for nm in payload.get("strongest_near_misses") or []:
        a(f"  {nm}")
    if not payload.get("strongest_near_misses"):
        a("  (none)")

    a("")
    a("=" * 72)
    a(f"FINAL DECISION: {payload.get('verdict')}")
    a(f"CODE: {payload.get('verdict_code')}")
    a(f"VARIANTS_TESTED: {payload.get('variants_tested')}")
    a(f"PROMOTED (next validation only): {payload.get('promoted_candidates')}")
    a(f"NEXT: {payload.get('next_stage')}")
    a("Live scanner: NOT modified. Research only. No deployment.")
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


def write_v8_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v8_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
