"""Beginner-friendly out-of-sample validation report."""

from __future__ import annotations

from typing import Any

from config import (
    FORWARD_BARS,
    MIN_SIGNALS_FOR_CONCLUSION,
    ROUND_TRIP_COST,
    VALIDATION_TRAIN_FRACTION,
)
from backtest.metrics import MetricBag


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "∞"
    return f"{x * 100:.{digits}f}%"


def _pf(x: float | None) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "∞"
    return f"{x:.2f}"


def _bag_lines(bag: MetricBag, indent: str = "  ") -> list[str]:
    return [
        f"{indent}n={bag.signals}  win={_pct(bag.win_rate)}  "
        f"avg={_pct(bag.avg_return)}  median={_pct(bag.median_return)}",
        f"{indent}cum={_pct(bag.cumulative_return)}  PF={_pf(bag.profit_factor)}  "
        f"maxDD={_pct(bag.max_drawdown)}  reliable={bag.reliable}",
    ]


def _conf_section(title: str, by_conf: dict[str, MetricBag]) -> list[str]:
    lines = [title]
    for conf in ("HIGH", "MEDIUM", "LOW"):
        bag = by_conf[conf]
        lines.append(f"  [{conf}]")
        lines.extend(_bag_lines(bag, "    "))
        if bag.note:
            lines.append(f"    note: {bag.note}")
    return lines


def build_validation_report(result: dict[str, Any]) -> str:
    rec = result["recommendation"]
    o_test = result["metrics"]["original_test"]
    r_test = result["metrics"]["revised_test"]
    o_train = result["metrics"]["original_train"]
    rules_o = result["original_rules"]
    rules_r = result["revised_rules"]

    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║   CONFIDENCE VALIDATION REPORT  ·  Beginner Friendly     ║",
        "║   Chronological train/test  ·  No look-ahead leakage     ║",
        "║   Analysis only  ·  NO brokerage  ·  NO live trades      ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {result['mode']}",
        f"Train fraction (by time, per series): {result['train_fraction']:.0%} earliest bars",
        f"Test fraction: {1 - result['train_fraction']:.0%} most recent bars (unseen for tuning)",
        f"Bars loaded: {result['bars_loaded']}",
        "",
        "═" * 60,
        "1) WHY HIGH WAS SO RARE",
        "═" * 60,
        result["why_high_rare"],
        "",
        "═" * 60,
        "2) HOW THIS VALIDATION WORKS",
        "═" * 60,
        "  • Each series is split chronologically (no shuffling).",
        f"  • First ~{VALIDATION_TRAIN_FRACTION:.0%} = TRAIN (diagnosis + propose revised rules).",
        "  • Last remaining window = TEST (frozen rules only).",
        "  • Signals still use only past bars at entry (no feature look-ahead).",
        "  • Exit uses later closes for the fixed hold period (normal for returns).",
        "  • Costs retained:",
    ]
    for cls, c in ROUND_TRIP_COST.items():
        lines.append(f"      - {cls}: {c * 100:.2f}% round-trip")
    lines.append("  • Forward holds:")
    for tf, n in FORWARD_BARS.items():
        lines.append(f"      - {tf}: {n} bars")
    lines.extend(
        [
            "  • Threshold quantiles and factor gates were PRE-SPECIFIED",
            "    (not searched to maximize train profit).",
            "",
            "═" * 60,
            "3) DO EXISTING FEATURES ADD PREDICTIVE VALUE? (TRAIN only)",
            "═" * 60,
        ]
    )
    for edge in result["feature_edges_train"]:
        lift = "n/a" if edge.lift is None else _pct(edge.lift)
        lines.append(
            f"  • {edge.feature}: hits={edge.hits}  lift(on-off)={lift}  "
            f"helpful={edge.helpful}  — {edge.note}"
        )

    lines.extend(
        [
            "",
            "═" * 60,
            "4) WHAT CHANGED IN THE REVISED CANDIDATE (from TRAIN)",
            "═" * 60,
            f"  Original thresholds: HIGH≥{rules_o.score_high} MED≥{rules_o.score_medium} "
            f"LOW≥{rules_o.score_low}",
            f"  Revised thresholds:  HIGH≥{rules_r.score_high} MED≥{rules_r.score_medium} "
            f"LOW≥{rules_r.score_low}  (HIGH min factors={rules_r.high_min_factors})",
            "  Rationale:",
        ]
    )
    for r in result["rationale"]:
        lines.append(f"    - {r}")
    if rules_r.notes:
        lines.append(f"  Notes: {rules_r.notes}")

    lines.extend(
        [
            "",
            "═" * 60,
            "5) ORIGINAL vs REVISED — OUT-OF-SAMPLE (TEST)",
            "═" * 60,
            "",
            *_conf_section("ORIGINAL rules on TEST:", o_test["by_confidence"]),
            "",
            *_conf_section("REVISED candidate on TEST:", r_test["by_confidence"]),
            "",
            "Overall TEST:",
            f"  Original: n={o_test['overall'].signals} avg={_pct(o_test['overall'].avg_return)} "
            f"median={_pct(o_test['overall'].median_return)} "
            f"cum={_pct(o_test['overall'].cumulative_return)} "
            f"DD={_pct(o_test['overall'].max_drawdown)}",
            f"  Revised:  n={r_test['overall'].signals} avg={_pct(r_test['overall'].avg_return)} "
            f"median={_pct(r_test['overall'].median_return)} "
            f"cum={_pct(r_test['overall'].cumulative_return)} "
            f"DD={_pct(r_test['overall'].max_drawdown)}",
        ]
    )

    # By asset class / timeframe on TEST for both
    lines.extend(["", "TEST by asset class (overall):"])
    for cls in sorted(set(o_test["by_asset_class"]) | set(r_test["by_asset_class"])):
        ob = o_test["by_asset_class"].get(cls)
        rb = r_test["by_asset_class"].get(cls)
        if ob:
            lines.append(f"  {cls} ORIGINAL: n={ob.signals} avg={_pct(ob.avg_return)} win={_pct(ob.win_rate)}")
        if rb:
            lines.append(f"  {cls} REVISED:  n={rb.signals} avg={_pct(rb.avg_return)} win={_pct(rb.win_rate)}")

    lines.extend(["", "TEST by timeframe (overall):"])
    for tf in sorted(set(o_test["by_timeframe"]) | set(r_test["by_timeframe"])):
        ob = o_test["by_timeframe"].get(tf)
        rb = r_test["by_timeframe"].get(tf)
        if ob:
            lines.append(f"  {tf} ORIGINAL: n={ob.signals} avg={_pct(ob.avg_return)} win={_pct(ob.win_rate)}")
        if rb:
            lines.append(f"  {tf} REVISED:  n={rb.signals} avg={_pct(rb.avg_return)} win={_pct(rb.win_rate)}")

    lines.extend(
        [
            "",
            "═" * 60,
            "6) IS HIGH > MEDIUM > LOW SUPPORTED?",
            "═" * 60,
            f"  Minimum signals for a trustworthy bucket: {MIN_SIGNALS_FOR_CONCLUSION}",
            f"  Original TEST HIGH n={o_test['by_confidence']['HIGH'].signals} "
            f"(reliable={o_test['by_confidence']['HIGH'].reliable})",
            f"  Revised TEST HIGH n={r_test['by_confidence']['HIGH'].signals} "
            f"(reliable={r_test['by_confidence']['HIGH'].reliable})",
            "  Compare average TEST returns only when samples are large enough.",
            "  Tiny HIGH samples can look great or terrible by luck — do not trust them alone.",
            "",
            "═" * 60,
            "7) TRAIN REFERENCE (not for final decision)",
            "═" * 60,
            *_conf_section("ORIGINAL on TRAIN:", o_train["by_confidence"]),
            "",
            "═" * 60,
            "8) RECOMMENDATION",
            "═" * 60,
            f"  >>> {rec['decision']} <<<",
            "",
            "  Reasons:",
        ]
    )
    for reason in rec["reasons"]:
        lines.append(f"    - {reason}")

    lines.extend(
        [
            "",
            "═" * 60,
            "9) LIMITATIONS",
            "═" * 60,
            "  • Public delayed Yahoo data; futures proxies for gold/silver/oil.",
            "  • Fixed hold periods; no stops, targets, or position sizing.",
            "  • Costs are simple round-trip assumptions, not a live broker model.",
            "  • Max drawdown chains paper trades sequentially (stress view).",
            "  • One chronological split — not a full multi-fold walk-forward grid.",
            "  • Feature lifts on train are noisy; weight tweaks are deliberately modest.",
            "  • Past TEST results still do not guarantee future performance.",
            "  • Live scanner continues to use ORIGINAL rules unless you approve a change.",
            "",
            "═" * 60,
            "End of validation · Analysis only · Not financial advice · DO NOT MERGE without review",
            "═" * 60,
            "",
        ]
    )
    return "\n".join(lines)
