"""Beginner report: AGREE / DISAGREE / WEEKLY_UNKNOWN and policies A/B/C."""

from __future__ import annotations

from typing import Any

from config import MIN_SIGNALS_FOR_CONCLUSION
from backtest.metrics import MetricBag


def _pct(x: float | None, d: int = 2) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "∞"
    return f"{x * 100:.{d}f}%"


def _line(bag: MetricBag) -> str:
    return (
        f"n={bag.signals} win={_pct(bag.win_rate)} avg={_pct(bag.avg_return)} "
        f"med={_pct(bag.median_return)} cum={_pct(bag.cumulative_return)} "
        f"dd={_pct(bag.max_drawdown)} reliable={bag.reliable}"
    )


def build_mtf_group_report(result: dict[str, Any]) -> str:
    tc = result["test_counts"]
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  MTF GROUP STUDY  ·  AGREE vs DISAGREE vs WEEKLY UNKNOWN ║",
        "║  Policies A / B / C on chronological TEST                ║",
        "║  Scoring thresholds UNCHANGED · Crypto disabled          ║",
        "║  Live scanner NOT changed · DO NOT MERGE yet             ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {result['mode']}",
        f"Focus: {result['focus']}",
        f"Train/Test: {result['train_fraction']:.0%} / {1 - result['train_fraction']:.0%}",
        f"Instruments: {', '.join(result['instruments'])}",
        f"Bars loaded: {result['bars_loaded']}",
        "",
        "TEST classification counts (1d signals):",
        f"  AGREE:           {tc['AGREE']}",
        f"  DISAGREE:        {tc['DISAGREE']}",
        f"  WEEKLY_UNKNOWN:  {tc['WEEKLY_UNKNOWN']}",
        f"  LOW (ungated):   {tc['LOW']}",
        "",
        "═" * 60,
        "GROUP PERFORMANCE ON TEST (MEDIUM+HIGH, then by confidence)",
        "═" * 60,
    ]

    for g in ("AGREE", "DISAGREE", "WEEKLY_UNKNOWN"):
        block = result["groups_test"][g]
        lines.append(f"\n[{g}] MEDIUM+HIGH combined: {_line(block['all_mh'])}")
        lines.append(f"  HIGH:   {_line(block['HIGH'])}")
        lines.append(f"  MEDIUM: {_line(block['MEDIUM'])}")
        if block["by_asset_class"]:
            lines.append("  By asset class:")
            for cls, bag in block["by_asset_class"].items():
                lines.append(f"    {cls}: {_line(bag)}")
        else:
            lines.append("  By asset class: (no signals)")

    lines.extend(
        [
            "",
            "═" * 60,
            "POLICY COMPARISON ON TEST",
            "═" * 60,
            "A = Original (keep all MEDIUM/HIGH)",
            "B = Suppress only genuine DISAGREE (keep AGREE + WEEKLY_UNKNOWN)",
            "C = Require explicit AGREE (suppress DISAGREE + WEEKLY_UNKNOWN)",
            "",
        ]
    )

    best_name = None
    best_avg = None
    for name, pol in result["policies_test"].items():
        lines.append(f"[{name}]")
        lines.append(f"  Kept MH groups: {pol['kept_mh_groups']}")
        lines.append(f"  OVERALL:     {_line(pol['overall'])}")
        lines.append(f"  MEDIUM+HIGH: {_line(pol['medium_high'])}")
        for conf in ("HIGH", "MEDIUM", "LOW"):
            lines.append(f"  {conf}: {_line(pol['by_confidence'][conf])}")
        lines.append("  By asset class:")
        for cls, bag in pol["by_asset_class"].items():
            lines.append(f"    {cls}: {_line(bag)}")
        lines.append("")
        mh = pol["medium_high"]
        if mh.avg_return is not None and mh.signals > 0:
            if best_avg is None or mh.avg_return > best_avg:
                best_avg = mh.avg_return
                best_name = name

    # Meaningful differences?
    a = result["policies_test"]["A_original"]["medium_high"]
    b = result["policies_test"]["B_suppress_disagree_only"]["medium_high"]
    c = result["policies_test"]["C_require_agree"]["medium_high"]

    lines.extend(
        [
            "═" * 60,
            "BEGINNER CONCLUSION",
            "═" * 60,
            f"Best MEDIUM+HIGH avg return on this TEST split: {best_name} "
            f"(avg={_pct(best_avg)}, n={result['policies_test'][best_name]['medium_high'].signals})",
            "",
            "Are differences meaningful?",
        ]
    )

    def note(label: str, bag: MetricBag) -> str:
        if bag.signals < MIN_SIGNALS_FOR_CONCLUSION:
            return f"  {label}: n={bag.signals} — BELOW reliability threshold ({MIN_SIGNALS_FOR_CONCLUSION})"
        return f"  {label}: n={bag.signals} — meets minimum sample threshold"

    lines.append(note("AGREE group", result["groups_test"]["AGREE"]["all_mh"]))
    lines.append(note("DISAGREE group", result["groups_test"]["DISAGREE"]["all_mh"]))
    lines.append(note("WEEKLY_UNKNOWN group", result["groups_test"]["WEEKLY_UNKNOWN"]["all_mh"]))
    lines.append(note("Policy A MH", a))
    lines.append(note("Policy B MH", b))
    lines.append(note("Policy C MH", c))

    # Interpret disagreement predictive value
    dis = result["groups_test"]["DISAGREE"]["all_mh"]
    agr = result["groups_test"]["AGREE"]["all_mh"]
    unk = result["groups_test"]["WEEKLY_UNKNOWN"]["all_mh"]
    lines.append("")
    if dis.signals >= 10 and agr.signals >= 10 and dis.avg_return is not None and agr.avg_return is not None:
        if dis.avg_return < agr.avg_return:
            lines.append(
                "  DISAGREE averaged worse than AGREE — genuine conflict may have predictive value."
            )
        else:
            lines.append(
                "  DISAGREE did not clearly underperform AGREE on this TEST split."
            )
    else:
        lines.append(
            "  Too few DISAGREE and/or AGREE signals to judge disagreement’s predictive value firmly."
        )

    if unk.signals >= MIN_SIGNALS_FOR_CONCLUSION and unk.avg_return is not None and agr.avg_return is not None:
        if unk.avg_return >= agr.avg_return - 0.0005:
            lines.append(
                "  WEEKLY_UNKNOWN was not clearly worse than AGREE — auto-suppressing unknowns may discard OK signals."
            )
        else:
            lines.append(
                "  WEEKLY_UNKNOWN looked weaker than AGREE — being stricter may help, but check sample sizes."
            )

    lines.extend(
        [
            "",
            "Recommendation posture: analysis only. Live scanner unchanged. Do not merge yet.",
            "Wait for explicit approval before enabling any MTF policy live.",
            "",
        ]
    )
    return "\n".join(lines)
