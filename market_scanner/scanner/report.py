"""Beginner-friendly daily summary formatting (no trading)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from scanner.levels import format_level
from scanner.opportunity import Opportunity, format_price, rank_opportunities


DIR_ICON = {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "● NEUTRAL"}


def _section_header(title: str) -> str:
    bar = "─" * 58
    return f"\n{bar}\n{title}\n{bar}"


def format_opportunity_card(opp: Opportunity) -> str:
    lines = [
        f"  {opp.instrument}  ({opp.name})",
        f"  Class: {opp.asset_class}   |   Timeframe: {opp.timeframe}",
        f"  Price: {format_price(opp.price)}",
        f"  Direction: {DIR_ICON.get(opp.direction, opp.direction)}",
        f"  Confidence: {opp.confidence}  (score {opp.score}/100)",
        f"  Why: {opp.reason}",
        f"  RSI: {opp.rsi if opp.rsi is not None else 'n/a'}",
        f"  SMA20 / SMA50: {opp.sma_relation}",
        f"  MACD: {opp.macd_condition}",
        f"  Volatility: {opp.volatility_note}",
        (
            f"  Support: {format_level(opp.support)}"
            + (f" / {format_level(opp.support_2)}" if opp.support_2 else "")
        ),
        (
            f"  Resistance: {format_level(opp.resistance)}"
            + (f" / {format_level(opp.resistance_2)}" if opp.resistance_2 else "")
        ),
    ]
    return "\n".join(lines)


def build_daily_summary(
    opportunities: Iterable[Opportunity],
    *,
    mode_label: str,
    errors: list[str] | None = None,
) -> str:
    """Plain-English daily report a beginner can read without Python knowledge."""
    ranked = rank_opportunities(list(opportunities))
    by_conf: dict[str, list[Opportunity]] = defaultdict(list)
    for opp in ranked:
        by_conf[opp.confidence].append(opp)

    high = by_conf.get("HIGH", [])
    medium = by_conf.get("MEDIUM", [])
    low = by_conf.get("LOW", [])
    none = by_conf.get("NO STRONG SETUP", [])

    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║          DAILY MARKET SCANNER  ·  Beginner Report        ║",
        "║     Forex · Gold/Silver/Oil  ·  Alerts Only              ║",
        "║     Crypto disabled by default  ·  NO auto-trading       ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {mode_label}",
        f"Markets reviewed: {len(ranked)}",
        f"Strong setups (HIGH): {len(high)}   |   MEDIUM: {len(medium)}   |   LOW: {len(low)}",
        f"NO STRONG SETUP: {len(none)}",
        "",
        "How to read this:",
        "  • HIGH / MEDIUM / LOW = how clearly indicators agree right now",
        "  • NO STRONG SETUP = wait — do not force a trade idea",
        "  • This is educational analysis only, not financial advice",
        "  • Active universe = Forex + commodities (crypto off unless re-enabled)",
    ]

    def dump(title: str, items: list[Opportunity], empty_msg: str) -> None:
        lines.append(_section_header(title))
        if not items:
            lines.append(f"  {empty_msg}")
            return
        for i, opp in enumerate(items, 1):
            lines.append(f"\n#{i}")
            lines.append(format_opportunity_card(opp))

    dump("HIGH CONFIDENCE — strongest opportunities first", high, "None today.")
    dump("MEDIUM CONFIDENCE", medium, "None today.")
    dump("LOW CONFIDENCE — weaker / early ideas only", low, "None today.")
    dump(
        "NO STRONG SETUP — markets with nothing clear enough",
        none,
        "Every market had at least a weak idea (unusual).",
    )

    # Quick top picks blurb
    lines.append(_section_header("BEGINNER QUICK TAKE"))
    actionable = [o for o in ranked if o.confidence in ("HIGH", "MEDIUM")]
    if not actionable:
        lines.append(
            "  Nothing stands out as a strong setup across Forex and commodities."
        )
        lines.append("  Sitting in cash / waiting is a valid choice.")
    else:
        lines.append("  Top ranked ideas to study (not automatic trades):")
        for opp in actionable[:5]:
            lines.append(
                f"  • {opp.instrument} {opp.timeframe}: {opp.direction.upper()} "
                f"[{opp.confidence} {opp.score}] — {opp.reason[:110]}"
            )

    # Per asset-class one-liners (only classes present in this scan)
    lines.append(_section_header("BY MARKET TYPE"))
    present = sorted({o.asset_class for o in ranked})
    for cls in present:
        subset = [o for o in ranked if o.asset_class == cls]
        best = next((o for o in subset if o.confidence != "NO STRONG SETUP"), None)
        if best is None:
            lines.append(f"  {cls.upper()}: NO STRONG SETUP across scanned instruments")
        else:
            lines.append(
                f"  {cls.upper()}: best watch = {best.instrument} {best.timeframe} "
                f"{best.direction} [{best.confidence} {best.score}] @ {format_price(best.price)}"
            )
    if "crypto" not in present:
        lines.append("  CRYPTO: disabled in active universe (code still available)")

    if errors:
        lines.append(_section_header("DATA NOTES"))
        for err in errors:
            lines.append(f"  ! {err}")

    lines.extend(
        [
            "",
            "═" * 60,
            "End of daily report · Alerts & analysis only · Not financial advice",
            "═" * 60,
            "",
        ]
    )
    return "\n".join(lines)
