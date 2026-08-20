"""V5 independent robustness validation report (analysis only)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from config import (
    V5_COMMODITIES,
    V5_ENTRY_SLIP_ATR,
    V5_FROZEN_ATR_STOP_MULT,
    V5_FROZEN_LOOKBACK,
    V5_FROZEN_MAX_HOLD,
    V5_HELD_OUT_STOCKS,
    V5_MAX_DD_ACCEPT,
    V5_MC_RUNS,
    V5_MC_SEED,
    V5_MIN_FOLDS_POSITIVE,
    V5_MIN_SYMBOLS_POSITIVE,
    V5_MIN_TRADES,
    V5_N_FOLDS,
    V5_TRAIN_FRACTION,
    V5_V4_STOCKS,
)
from backtest.metrics import summarize_trades
from backtest.scanner_v2 import V2Trade
from backtest.scanner_v5 import (
    AUDIT_FINDINGS,
    FROZEN,
    chronological_folds,
    leave_out_symbols,
    monte_carlo,
    run_on_map,
)


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
    out = {}
    contrib = {}
    for sym, ts in sorted(groups.items()):
        m = slice_metrics(sym, ts)
        out[sym] = m
        contrib[sym] = float(sum(t.net_return for t in ts))
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    pos = sum(v for _, v in ranked if v > 0)
    def share(n: int) -> Optional[float]:
        if pos <= 0:
            return None
        return float(sum(v for _, v in ranked[:n] if v > 0) / pos)

    return {
        "per_symbol": out,
        "pnl_contribution": {k: float(v) for k, v in ranked},
        "best1_share_of_positive_pnl": share(1),
        "best3_share_of_positive_pnl": share(3),
        "best5_share_of_positive_pnl": share(5),
        "best_symbols": [k for k, _ in ranked[:5]],
    }


def evaluate_window(
    series_map: dict,
    instruments: tuple[str, ...] | list[str],
    *,
    start_frac: float,
    end_frac: float,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    **overrides,
) -> dict[str, Any]:
    trades = run_on_map(
        series_map,
        instruments=instruments,
        start_frac=start_frac,
        end_frac=end_frac,
        cost_mult=cost_mult,
        entry_slip_atr=entry_slip_atr,
        **overrides,
    )
    return {"trades": trades, "metrics": slice_metrics("window", trades)}


def oos_bundle(
    series_map: dict,
    instruments: tuple[str, ...] | list[str],
    *,
    train_frac: float = V5_TRAIN_FRACTION,
    n_folds: int = V5_N_FOLDS,
) -> dict[str, Any]:
    train = run_on_map(
        series_map, instruments=instruments, start_frac=0.0, end_frac=train_frac
    )
    test = run_on_map(
        series_map, instruments=instruments, start_frac=train_frac, end_frac=1.0
    )
    test_15 = run_on_map(
        series_map,
        instruments=instruments,
        start_frac=train_frac,
        end_frac=1.0,
        cost_mult=1.5,
    )
    test_2x = run_on_map(
        series_map,
        instruments=instruments,
        start_frac=train_frac,
        end_frac=1.0,
        cost_mult=2.0,
    )
    test_slip = run_on_map(
        series_map,
        instruments=instruments,
        start_frac=train_frac,
        end_frac=1.0,
        cost_mult=1.0,
        entry_slip_atr=V5_ENTRY_SLIP_ATR,
    )
    test_2x_slip = run_on_map(
        series_map,
        instruments=instruments,
        start_frac=train_frac,
        end_frac=1.0,
        cost_mult=2.0,
        entry_slip_atr=V5_ENTRY_SLIP_ATR,
    )
    folds = chronological_folds(
        series_map, instruments=instruments, n_folds=n_folds, cost_mult=1.0
    )
    fold_rows = []
    fold_exps = []
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
    sym = by_symbol(test)
    best = sym["best_symbols"]
    drop1 = leave_out_symbols(test, {best[0]} if best else set())
    drop3 = leave_out_symbols(test, set(best[:3]))
    gate = {
        "positive_oos_expectancy": bool((_exp(test) or 0) > 0),
        "positive_after_normal_costs": bool((_exp(test) or 0) > 0),
        "resilient_2x": bool(
            (_exp(test_2x) or -1) > 0
            or (
                (_exp(test) or 0) > 0
                and (_exp(test_2x) or -1) >= 0.5 * (_exp(test) or 0)
            )
        ),
        "resilient_2x_slip": bool(
            (_exp(test_2x_slip) or -1) > 0
            or (
                (_exp(test) or 0) > 0
                and (_exp(test_2x_slip) or -1) >= 0.5 * (_exp(test) or 0)
            )
        ),
        "folds_positive_ge_3": sum(1 for e in fold_exps if e > 0) >= V5_MIN_FOLDS_POSITIVE,
        "folds_positive_count": sum(1 for e in fold_exps if e > 0),
        "adequate_trade_count": len(test) >= V5_MIN_TRADES,
        "acceptable_max_drawdown": bool(
            (slice_metrics("t", test).get("max_drawdown") or 1) <= V5_MAX_DD_ACCEPT
        ),
        "multi_symbol": len(
            [
                s
                for s, m in sym["per_symbol"].items()
                if (m.get("expectancy") or 0) > 0 and (m.get("signals") or 0) >= 3
            ]
        )
        >= V5_MIN_SYMBOLS_POSITIVE,
        "not_one_symbol_dominant": not (
            (sym.get("best1_share_of_positive_pnl") or 0) >= 0.70
        ),
    }
    gate["all_core_pass"] = all(
        gate[k]
        for k in (
            "positive_oos_expectancy",
            "positive_after_normal_costs",
            "resilient_2x",
            "folds_positive_ge_3",
            "adequate_trade_count",
            "acceptable_max_drawdown",
            "multi_symbol",
            "not_one_symbol_dominant",
        )
    )
    return {
        "instruments": list(instruments),
        "train": slice_metrics("train", train),
        "test": slice_metrics("test", test),
        "test_1_5x_costs": slice_metrics("1.5x", test_15),
        "test_2x_costs": slice_metrics("2x", test_2x),
        "test_entry_slip": slice_metrics("slip", test_slip),
        "test_2x_and_slip": slice_metrics("2x_slip", test_2x_slip),
        "folds": fold_rows,
        "symbol_robustness": sym,
        "leave_out_best1": slice_metrics("drop1", drop1),
        "leave_out_best3": slice_metrics("drop3", drop3),
        "monte_carlo": monte_carlo(test, n_runs=V5_MC_RUNS, seed=V5_MC_SEED),
        "gate": gate,
        "test_trades_n": len(test),
    }


def parameter_sensitivity(series_map: dict, instruments: tuple[str, ...]) -> dict[str, Any]:
    """Small perturbations around frozen params — report only, do not pick a winner."""
    base = oos_bundle(series_map, instruments)
    base_exp = base["test"]["expectancy"]
    variants = []
    grid = [
        {"lookback": 15},
        {"lookback": 25},
        {"atr_stop_mult": 1.25},
        {"atr_stop_mult": 1.75},
        {"max_hold": 20},
        {"max_hold": 28},
    ]
    for ov in grid:
        test = run_on_map(
            series_map,
            instruments=instruments,
            start_frac=V5_TRAIN_FRACTION,
            end_frac=1.0,
            **ov,
        )
        exp = _exp(test)
        variants.append(
            {
                "override": ov,
                "test_n": len(test),
                "test_expectancy": exp,
                "test_metrics": slice_metrics("sens", test),
                "still_positive": bool(exp is not None and exp > 0),
                "delta_vs_frozen_bp": None
                if exp is None or base_exp is None
                else (exp - base_exp) * 10000.0,
            }
        )
    pos = sum(1 for v in variants if v["still_positive"])
    fragile = pos < max(1, len(variants) // 2)
    return {
        "frozen_reference": {
            "lookback": V5_FROZEN_LOOKBACK,
            "atr_stop_mult": V5_FROZEN_ATR_STOP_MULT,
            "max_hold": V5_FROZEN_MAX_HOLD,
            "test_expectancy": base_exp,
        },
        "variants": variants,
        "positive_variant_count": pos,
        "variant_count": len(variants),
        "fragile": fragile,
        "note": "Sensitivity only — no new parameter selected from these results.",
    }


def build_v5_payload(series_map: dict) -> dict[str, Any]:
    print("V5 audit (frozen V4_S1_STOCK)...", flush=True)
    audit = {
        "frozen_spec": {
            "name": FROZEN.name,
            "require_structure": FROZEN.require_structure,
            "require_ma": FROZEN.require_ma,
            "require_adx": FROZEN.require_adx,
            "break_mode": FROZEN.break_mode,
            "lookback": FROZEN.lookback,
            "atr_stop_mult": FROZEN.atr_stop_mult,
            "max_hold": FROZEN.max_hold,
            "filter_name": FROZEN.filter_name,
            "exit_policy": FROZEN.exit_policy,
            "risk_fraction": FROZEN.risk_fraction,
        },
        "findings": AUDIT_FINDINGS,
        "blocking_issues": [
            f for f in AUDIT_FINDINGS if f["severity"] == "error"
        ],
        "continue_after_audit": True,
    }
    # No hard blockers — warn-level items recorded

    print("V5 — original V4 stocks...", flush=True)
    v4_stocks = oos_bundle(series_map, V5_V4_STOCKS)

    print("V5 — held-out stocks (independent)...", flush=True)
    held_out = oos_bundle(series_map, V5_HELD_OUT_STOCKS)

    print("V5 — combined stocks (V4 + held-out)...", flush=True)
    all_stocks = oos_bundle(series_map, tuple(V5_V4_STOCKS) + tuple(V5_HELD_OUT_STOCKS))

    print("V5 — commodities separate...", flush=True)
    commodities = oos_bundle(series_map, V5_COMMODITIES)

    print("V5 — parameter sensitivity (report only)...", flush=True)
    # Sensitivity on held-out + V4 stocks combined for stability view, but label clearly
    sens = parameter_sensitivity(
        series_map, tuple(V5_V4_STOCKS) + tuple(V5_HELD_OUT_STOCKS)
    )

    forex_note = (
        "FX is not optimised in V5. Prior research finding stands: FX requires a "
        "separate strategy-family research project. No FX retune performed."
    )

    verdict_code, verdict, protocol = final_verdict(
        audit=audit,
        v4_stocks=v4_stocks,
        held_out=held_out,
        all_stocks=all_stocks,
        commodities=commodities,
        sens=sens,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V5 independent robustness validation of frozen V4_S1_STOCK",
        "live_status": "ORIGINAL/live untouched; no paper/live enablement",
        "audit": audit,
        "v4_original_stocks": v4_stocks,
        "held_out_stocks": held_out,
        "all_stocks": all_stocks,
        "commodities_separate": commodities,
        "parameter_sensitivity": sens,
        "forex_note": forex_note,
        "verdict_code": verdict_code,
        "verdict": verdict,
        "proposed_paper_protocol": protocol,
        "instruments_loaded": sorted({k for k, _ in series_map}),
    }


def final_verdict(
    *,
    audit: dict,
    v4_stocks: dict,
    held_out: dict,
    all_stocks: dict,
    commodities: dict,
    sens: dict,
) -> tuple[str, str, Optional[dict]]:
    if audit.get("blocking_issues"):
        return (
            "V5_FAIL",
            "V5 FAIL — HISTORICAL EDGE NOT ROBUST ENOUGH (blocking audit issues).",
            None,
        )

    # Primary falsification target: held-out stocks must support the edge
    h = held_out["gate"]
    a = all_stocks["gate"]
    # Require held-out core pass OR (held-out positive OOS + all_stocks core pass)
    held_ok = h.get("all_core_pass")
    # If held-out n is small, allow all_stocks as supporting but still require held-out positive
    held_positive = h.get("positive_oos_expectancy") and h.get("folds_positive_ge_3")
    all_ok = a.get("all_core_pass")
    not_fragile = not sens.get("fragile", True)
    leave1 = (held_out["leave_out_best1"].get("expectancy") or 0) > 0 or (
        all_stocks["leave_out_best1"].get("expectancy") or 0
    ) > 0

    stock_pass = bool(
        (held_ok or (held_positive and all_ok))
        and a.get("resilient_2x")
        and not_fragile
        and leave1
        and a.get("not_one_symbol_dominant")
    )

    # Stricter: prefer held-out all_core_pass
    if not held_ok:
        # Fail if held-out does not independently clear core gates
        stock_pass = False

    commodity_pass = bool(commodities["gate"].get("all_core_pass"))

    protocol = None
    if stock_pass:
        protocol = {
            "status": "PROPOSED — not enabled",
            "universe": "stocks only (V4_S1_STOCK frozen rules)",
            "timeframe": "4H",
            "risk_per_trade": "1% equity (ATR stop = 1R)",
            "duration_target": "minimum 90 calendar days OR 40 completed paper trades (whichever later)",
            "logging": [
                "signal timestamp (bar close)",
                "instrument, direction, entry, stop, planned size",
                "exit timestamp, exit reason, R-multiple, costs",
                "equity snapshot after each trade",
            ],
            "risk_controls": [
                "max 1 open risk unit per symbol",
                "max 3 concurrent open stock positions",
                "daily stop: halt new entries if day equity DD ≥ 3%",
                "strategy stop: pause if rolling 20-trade expectancy < 0 and DD ≥ 15%",
            ],
            "success_criteria_to_continue": [
                "paper expectancy > 0 after estimated costs",
                "max DD within 1.5× V5 historical TEST DD band",
                "no single symbol > 50% of paper PnL",
            ],
            "stop_criteria": [
                "expectancy ≤ 0 after ≥ 40 trades",
                "DD ≥ 25% from paper peak",
                "execution/slippage systematically worse than 2× stress assumption",
            ],
            "explicitly_excluded": ["forex", "live order routing", "parameter changes mid-test"],
            "commodities": (
                "PASS independently — may start separate paper track"
                if commodity_pass
                else "Do NOT combine with stocks; commodities did not independently clear V5 gates"
            ),
        }
        msg = (
            "V5 PASS — READY FOR FORWARD PAPER-TRADING VALIDATION. "
            "Paper trading is NOT enabled in code; review the proposed protocol first. "
            "ORIGINAL/live remains untouched."
        )
        if not commodity_pass:
            msg += " Commodities remain separate and do not independently pass."
        return "V5_PASS", msg, protocol

    reasons = []
    if not held_ok:
        reasons.append(
            f"held-out stocks failed core gates "
            f"(exp={held_out['test'].get('expectancy')}, "
            f"folds+={h.get('folds_positive_count')}, n={held_out.get('test_trades_n')})"
        )
    if sens.get("fragile"):
        reasons.append("parameter sensitivity classified fragile")
    if not a.get("resilient_2x"):
        reasons.append("all-stocks 2× cost resilience failed")
    return (
        "V5_FAIL",
        "V5 FAIL — HISTORICAL EDGE NOT ROBUST ENOUGH. " + "; ".join(reasons),
        None,
    )


def format_v5_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("=" * 72)
    a("SCANNER V5 — INDEPENDENT ROBUSTNESS VALIDATION")
    a("Frozen candidate: V4_S1_STOCK | no retune | live untouched")
    a("=" * 72)
    a(f"Generated: {payload.get('generated_at')}")
    a("")
    a("# 1. AUDIT")
    spec = (payload.get("audit") or {}).get("frozen_spec") or {}
    a(f"  Frozen spec: {spec}")
    for f in (payload.get("audit") or {}).get("findings") or []:
        a(f"  [{f.get('severity')}] {f.get('id')} {f.get('title')}")
        a(f"      {f.get('detail')}")
    a(f"  Continue after audit: {(payload.get('audit') or {}).get('continue_after_audit')}")
    a("")

    def dump(title: str, block: dict) -> None:
        a("-" * 72)
        a(f"  {title}  instruments={block.get('instruments')}")
        for key in (
            "train",
            "test",
            "test_1_5x_costs",
            "test_2x_costs",
            "test_entry_slip",
            "test_2x_and_slip",
        ):
            m = block.get(key) or {}
            a(
                f"    {key:18s} n={m.get('signals', 0):4d} win={_pct(m.get('win_rate'))} "
                f"avgW={_bps(m.get('avg_winner'))} avgL={_bps(m.get('avg_loser'))} "
                f"exp={_bps(m.get('expectancy'))} PF={_pf(m.get('profit_factor'))} "
                f"DD={_pct(m.get('max_drawdown'))} tot={_pct(m.get('total_return'))}"
            )
        a("    folds:")
        for fr in block.get("folds") or []:
            m = fr.get("metrics") or {}
            a(
                f"      fold {fr['fold']} n={m.get('signals', 0)} "
                f"exp={_bps(m.get('expectancy'))} DD={_pct(m.get('max_drawdown'))} "
                f"tot={_pct(m.get('total_return'))}"
            )
        sym = block.get("symbol_robustness") or {}
        a(
            f"    best1/3/5 positive-PnL share: "
            f"{sym.get('best1_share_of_positive_pnl')}, "
            f"{sym.get('best3_share_of_positive_pnl')}, "
            f"{sym.get('best5_share_of_positive_pnl')}"
        )
        a("    per symbol (TEST):")
        for s, m in (sym.get("per_symbol") or {}).items():
            pnl = (sym.get("pnl_contribution") or {}).get(s)
            a(
                f"      {s:6s} n={m.get('signals', 0):3d} exp={_bps(m.get('expectancy'))} "
                f"pnl_sum={_bps(pnl)}"
            )
        d1 = block.get("leave_out_best1") or {}
        d3 = block.get("leave_out_best3") or {}
        a(
            f"    leave-out best1: n={d1.get('signals')} exp={_bps(d1.get('expectancy'))} | "
            f"best3: n={d3.get('signals')} exp={_bps(d3.get('expectancy'))}"
        )
        mc = block.get("monte_carlo") or {}
        a(f"    Monte Carlo: {mc}")
        a(f"    GATE: {block.get('gate')}")

    a("# 2–7. STOCK VALIDATION")
    dump("V4 original stocks", payload.get("v4_original_stocks") or {})
    dump("Held-out stocks (independent)", payload.get("held_out_stocks") or {})
    dump("All stocks (V4 + held-out)", payload.get("all_stocks") or {})

    a("")
    a("# 5. PARAMETER SENSITIVITY (not optimisation)")
    sens = payload.get("parameter_sensitivity") or {}
    a(f"  frozen ref: {sens.get('frozen_reference')}")
    a(f"  fragile={sens.get('fragile')} positive_variants={sens.get('positive_variant_count')}/{sens.get('variant_count')}")
    for v in sens.get("variants") or []:
        a(
            f"    {v.get('override')} n={v.get('test_n')} exp={_bps(v.get('test_expectancy'))} "
            f"pos={v.get('still_positive')} dBp={v.get('delta_vs_frozen_bp')}"
        )

    a("")
    a("# 8. COMMODITIES (separate — not combined with stocks)")
    dump("Commodities", payload.get("commodities_separate") or {})

    a("")
    a("# 9. FOREX")
    a(f"  {payload.get('forex_note')}")

    a("")
    a("=" * 72)
    a(f"VERDICT: {payload.get('verdict')}")
    a(f"CODE:    {payload.get('verdict_code')}")
    if payload.get("proposed_paper_protocol"):
        a("PROPOSED PAPER PROTOCOL (not enabled):")
        a(json.dumps(payload["proposed_paper_protocol"], indent=2))
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


def write_v5_reports(payload: dict[str, Any], txt_path: str, json_path: str) -> str:
    text = format_v5_report(payload)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return text
