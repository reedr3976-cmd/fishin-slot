"""Beginner-friendly ORIGINAL vs MTF-filter comparison report."""

from __future__ import annotations

from typing import Any

from config import FORWARD_BARS, MIN_SIGNALS_FOR_CONCLUSION, ROUND_TRIP_COST
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


def _bag_block(title: str, bag: MetricBag) -> list[str]:
    return [
        title,
        f"  n={bag.signals}  win={_pct(bag.win_rate)}  avg={_pct(bag.avg_return)}  "
        f"median={_pct(bag.median_return)}",
        f"  cum={_pct(bag.cumulative_return)}  PF={_pf(bag.profit_factor)}  "
        f"maxDD={_pct(bag.max_drawdown)}  reliable={bag.reliable}",
    ]


def build_mtf_comparison_report(result: dict[str, Any]) -> str:
    o = result["metrics"]["original_test"]
    m = result["metrics"]["mtf_test"]
    mh_o = result["high_medium_test"]["original"]
    mh_m = result["high_medium_test"]["mtf"]
    st = result["mtf_test_stats"]

    removed = st["suppressed_disagree"] + st["suppressed_missing"]
    cand = st["candidates_medium_high"]

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  MULTI-TIMEFRAME FILTER COMPARISON  ·  Beginner Report   ║",
        "║  ORIGINAL vs MTF-filtered (1d ↔ 1wk agreement)           ║",
        "║  Scoring thresholds UNCHANGED  ·  Crypto disabled        ║",
        "║  Analysis only  ·  Live scanner NOT switched yet         ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {result['mode']}",
        f"Train/Test split: {result['train_fraction']:.0%} / {1 - result['train_fraction']:.0%} chronological",
        f"Instruments: {', '.join(result['instruments'])}",
        f"Bars loaded: {result['bars_loaded']}",
        "",
        "FILTER RULE (MEDIUM/HIGH only):",
        "  Keep only if 1d direction and 1wk direction agree (both bullish or both bearish).",
        "  If they disagree (or the other TF has no clear direction), suppress the signal.",
        "  LOW signals are unchanged (no MTF gate).",
        "  Score thresholds are NOT changed.",
        "",
        "Costs (unchanged):",
    ]
    for cls, c in ROUND_TRIP_COST.items():
        lines.append(f"  • {cls}: {c * 100:.2f}% round-trip")
    lines.append("Forward holds:")
    for tf, n in FORWARD_BARS.items():
        if tf in ("1d", "1wk"):
            lines.append(f"  • {tf}: {n} bars")

    lines.extend(
        [
            "",
            "═" * 60,
            "SIGNALS REMOVED BY MTF FILTER (TEST window)",
            "═" * 60,
            f"  MEDIUM/HIGH candidates evaluated: {cand}",
            f"  Kept (1d & 1wk agree):           {st['kept_agree']}",
            f"  Removed (direction disagree):   {st['suppressed_disagree']}",
            f"  Removed (missing/unclear pair): {st['suppressed_missing']}",
            f"  Total removed:                  {removed}",
            f"  LOW signals kept (ungated):     {st['low_kept']}",
            "",
            "═" * 60,
            "OUT-OF-SAMPLE TEST — ORIGINAL",
            "═" * 60,
        ]
    )
    for conf in ("HIGH", "MEDIUM", "LOW"):
        lines.extend(_bag_block(f"[{conf}]", o["by_confidence"][conf]))
    lines.extend(_bag_block("[OVERALL]", o["overall"]))
    lines.extend(_bag_block("[MEDIUM+HIGH only]", mh_o))

    lines.extend(["", "TEST by asset class (ORIGINAL):"])
    for cls, bag in o["by_asset_class"].items():
        lines.append(
            f"  {cls}: n={bag.signals} win={_pct(bag.win_rate)} avg={_pct(bag.avg_return)} "
            f"med={_pct(bag.median_return)} cum={_pct(bag.cumulative_return)} dd={_pct(bag.max_drawdown)}"
        )

    lines.extend(
        [
            "",
            "═" * 60,
            "OUT-OF-SAMPLE TEST — MTF FILTERED",
            "═" * 60,
        ]
    )
    for conf in ("HIGH", "MEDIUM", "LOW"):
        lines.extend(_bag_block(f"[{conf}]", m["by_confidence"][conf]))
    lines.extend(_bag_block("[OVERALL]", m["overall"]))
    lines.extend(_bag_block("[MEDIUM+HIGH only]", mh_m))

    lines.extend(["", "TEST by asset class (MTF):"])
    for cls, bag in m["by_asset_class"].items():
        lines.append(
            f"  {cls}: n={bag.signals} win={_pct(bag.win_rate)} avg={_pct(bag.avg_return)} "
            f"med={_pct(bag.median_return)} cum={_pct(bag.cumulative_return)} dd={_pct(bag.max_drawdown)}"
        )

    # Improvement verdict on MEDIUM+HIGH (the filter's target)
    lines.extend(
        [
            "",
            "═" * 60,
            "DOES MTF AGREEMENT IMPROVE TEST PERFORMANCE?",
            "═" * 60,
        ]
    )
    if mh_o.signals == 0 and mh_m.signals == 0:
        lines.append("  No MEDIUM/HIGH signals to compare.")
        verdict = "NEED MORE DATA"
    else:
        lines.append(
            f"  ORIGINAL MEDIUM+HIGH: n={mh_o.signals} avg={_pct(mh_o.avg_return)} "
            f"win={_pct(mh_o.win_rate)} cum={_pct(mh_o.cumulative_return)}"
        )
        lines.append(
            f"  MTF MEDIUM+HIGH:      n={mh_m.signals} avg={_pct(mh_m.avg_return)} "
            f"win={_pct(mh_m.win_rate)} cum={_pct(mh_m.cumulative_return)}"
        )
        lines.append(
            f"  ORIGINAL overall avg={_pct(o['overall'].avg_return)} n={o['overall'].signals}"
        )
        lines.append(
            f"  MTF overall avg={_pct(m['overall'].avg_return)} n={m['overall'].signals}"
        )

        improved_mh = (
            mh_m.avg_return is not None
            and mh_o.avg_return is not None
            and mh_m.signals >= 10
            and mh_m.avg_return > mh_o.avg_return
        )
        improved_overall = (
            m["overall"].avg_return is not None
            and o["overall"].avg_return is not None
            and m["overall"].avg_return >= (o["overall"].avg_return or 0) - 1e-6
        )
        if mh_m.signals < MIN_SIGNALS_FOR_CONCLUSION:
            verdict = "NEED MORE DATA"
            lines.append(
                f"  MTF MEDIUM+HIGH sample ({mh_m.signals}) is below "
                f"{MIN_SIGNALS_FOR_CONCLUSION} — treat results cautiously."
            )
        elif improved_mh and improved_overall:
            verdict = "MTF LOOKS HELPFUL (not live yet)"
            lines.append("  MTF improved MEDIUM+HIGH average return without hurting overall.")
        elif improved_mh and not improved_overall:
            verdict = "MIXED"
            lines.append("  MTF helped MEDIUM+HIGH averages but overall TEST did not clearly improve.")
        else:
            verdict = "NO CLEAR IMPROVEMENT"
            lines.append("  Requiring 1d/1wk agreement did not clearly improve out-of-sample results.")

    lines.extend(
        [
            "",
            f"VERDICT: {verdict}",
            "",
            "Live scanner remains on ORIGINAL rules with MTF filter OFF by default.",
            "Do NOT merge / enable live until you explicitly approve.",
            "",
            "Limitations: one chronological split; public data; simple costs; fixed holds;",
            "weekly confirmation uses last completed weekly bar at or before the signal time.",
            "",
        ]
    )
    return "\n".join(lines)
