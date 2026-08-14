"""Beginner-friendly backtest report (analysis only)."""

from __future__ import annotations

from typing import Any

from config import (
    FORWARD_BARS,
    MIN_SIGNALS_FOR_CONCLUSION,
    ROUND_TRIP_COST,
)
from backtest.engine import BacktestRun
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


def _bag_block(bag: MetricBag) -> list[str]:
    return [
        f"  Signals:        {bag.signals}",
        f"  Wins / Losses:  {bag.wins} / {bag.losses}",
        f"  Win rate:       {_pct(bag.win_rate)}",
        f"  Avg return:     {_pct(bag.avg_return)}",
        f"  Avg winner:     {_pct(bag.avg_winner)}",
        f"  Avg loser:      {_pct(bag.avg_loser)}",
        f"  Profit factor:  {_pf(bag.profit_factor)}",
        f"  Max drawdown:   {_pct(bag.max_drawdown)}",
        f"  Reliable?:      {'YES' if bag.reliable else 'NO — ' + bag.note}",
    ]


def _ranking_conclusion(by_conf: dict[str, MetricBag]) -> list[str]:
    lines = [
        "Did HIGH signals actually beat MEDIUM and LOW?",
        "",
    ]
    high, med, low = by_conf["HIGH"], by_conf["MEDIUM"], by_conf["LOW"]
    counts = (high.signals, med.signals, low.signals)
    if sum(counts) == 0:
        lines.append("No actionable historical signals were found, so no conclusion is possible.")
        return lines

    insufficient = [
        name
        for name, bag in (("HIGH", high), ("MEDIUM", med), ("LOW", low))
        if bag.signals < MIN_SIGNALS_FOR_CONCLUSION
    ]
    if insufficient:
        lines.append(
            "Not enough signals for a reliable ranking comparison yet."
        )
        lines.append(
            f"Buckets below the minimum of {MIN_SIGNALS_FOR_CONCLUSION} signals: "
            + ", ".join(insufficient)
            + "."
        )
        lines.append("")
        lines.append("Observed averages (use cautiously — small samples):")
    else:
        lines.append(
            f"Each confidence bucket has at least {MIN_SIGNALS_FOR_CONCLUSION} signals."
        )
        lines.append("Observed averages:")

    for name, bag in (("HIGH", high), ("MEDIUM", med), ("LOW", low)):
        lines.append(
            f"  • {name}: n={bag.signals}, win_rate={_pct(bag.win_rate)}, "
            f"avg_return={_pct(bag.avg_return)}, PF={_pf(bag.profit_factor)}"
        )

    # Ordering check on avg_return when samples exist
    scored = [
        (name, bag.avg_return)
        for name, bag in (("HIGH", high), ("MEDIUM", med), ("LOW", low))
        if bag.signals > 0 and bag.avg_return is not None
    ]
    if len(scored) >= 2:
        ordered = sorted(scored, key=lambda x: x[1], reverse=True)
        lines.append("")
        lines.append(
            "Average-return order (best → worst): "
            + " > ".join(n for n, _ in ordered)
        )
        if ordered[0][0] == "HIGH" and not insufficient:
            lines.append(
                "Baseline result: HIGH looked better than lower tiers on average return "
                "in this sample — still not a guarantee of future performance."
            )
        elif ordered[0][0] != "HIGH":
            lines.append(
                "Baseline result: HIGH did NOT clearly outperform the other tiers "
                "in this historical sample. Treat live HIGH ratings with caution until "
                "more evidence is collected."
            )
        if insufficient:
            lines.append(
                "Because some buckets are small, do NOT treat this as proof either way."
            )
    lines.append("")
    lines.append(
        "Important: past results do not guarantee future results. "
        "This is an educational baseline, not financial advice."
    )
    return lines


def build_backtest_report(run: BacktestRun, metrics: dict[str, Any]) -> str:
    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║     HISTORICAL BACKTEST REPORT  ·  Beginner Friendly     ║",
        "║     Existing scanner rules  ·  No look-ahead bias        ║",
        "║     Analysis only  ·  NO brokerage  ·  NO live trades    ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {run.mode}",
        f"Instruments: {', '.join(run.instruments)}",
        f"Timeframes: {', '.join(run.timeframes)}",
        f"Bars loaded (sum across series): {run.bars_scanned}",
        f"Total paper signals evaluated: {len(run.trades)}",
        "",
        "HOW A WIN IS DEFINED",
        "  1. At each historical bar (after warmup), score the market using ONLY",
        "     prices available up to that bar (no future data).",
        "  2. If the scanner says HIGH / MEDIUM / LOW with bullish or bearish",
        "     direction, open a hypothetical paper trade at that bar's CLOSE.",
        "  3. Hold for a fixed forward period, then exit at that later CLOSE:",
    ]
    for tf, bars in FORWARD_BARS.items():
        if tf in run.timeframes:
            lines.append(f"       • {tf}: hold {bars} bars")
    lines.extend(
        [
            "  4. Bullish signal wins if price rose over the hold (after costs).",
            "     Bearish signal wins if price fell over the hold (after costs).",
            "  5. Win = net return > 0 after round-trip cost/slippage assumption.",
            "  6. Signals on one market are non-overlapping (no stacking).",
            "",
            "ABOUT MAX DRAWDOWN",
            "  Max drawdown here chains every paper trade in time order as if they",
            "  were one long sequence. That is a stress view, not a real multi-market",
            "  portfolio with position sizing. Use it to compare buckets, not as a",
            "  literal account forecast.",
            "",
            "COST / SLIPPAGE ASSUMPTIONS (round-trip, educational estimates)",
        ]
    )
    for cls, c in ROUND_TRIP_COST.items():
        lines.append(f"  • {cls}: {c * 100:.2f}% of price per round trip")

    lines.extend(
        [
            "",
            "Scanner scoring rules were NOT changed to improve these results.",
            f"Minimum signals for a 'reliable' bucket: {MIN_SIGNALS_FOR_CONCLUSION}",
            "",
            "═" * 60,
            "RESULTS BY CONFIDENCE (HIGH / MEDIUM / LOW)",
            "═" * 60,
        ]
    )
    for conf in ("HIGH", "MEDIUM", "LOW"):
        bag = metrics["by_confidence"][conf]
        lines.append(f"\n[{conf}]")
        lines.extend(_bag_block(bag))

    lines.extend(["", "═" * 60, "RESULTS BY ASSET CLASS", "═" * 60])
    for cls, bag in metrics["by_asset_class"].items():
        lines.append(f"\n[{cls.upper()}] overall")
        lines.extend(_bag_block(bag))
        for conf in ("HIGH", "MEDIUM", "LOW"):
            sub = metrics["by_asset_class_confidence"][cls][conf]
            if sub.signals == 0:
                continue
            lines.append(f"  — {cls}/{conf}: n={sub.signals}, win={_pct(sub.win_rate)}, "
                         f"avg={_pct(sub.avg_return)}, PF={_pf(sub.profit_factor)}, "
                         f"DD={_pct(sub.max_drawdown)}")

    lines.extend(["", "═" * 60, "RESULTS BY TIMEFRAME", "═" * 60])
    for tf, bag in metrics["by_timeframe"].items():
        lines.append(f"\n[{tf}] overall")
        lines.extend(_bag_block(bag))
        for conf in ("HIGH", "MEDIUM", "LOW"):
            sub = metrics["by_timeframe_confidence"][tf][conf]
            if sub.signals == 0:
                continue
            lines.append(f"  — {tf}/{conf}: n={sub.signals}, win={_pct(sub.win_rate)}, "
                         f"avg={_pct(sub.avg_return)}, PF={_pf(sub.profit_factor)}, "
                         f"DD={_pct(sub.max_drawdown)}")

    lines.extend(["", "═" * 60, "BEGINNER CONCLUSION", "═" * 60, ""])
    lines.extend(_ranking_conclusion(metrics["by_confidence"]))

    if run.errors:
        lines.extend(["", "DATA NOTES"])
        for err in run.errors:
            lines.append(f"  ! {err}")

    lines.extend(
        [
            "",
            "═" * 60,
            "End of backtest · Analysis only · Not financial advice",
            "═" * 60,
            "",
        ]
    )
    return "\n".join(lines)
