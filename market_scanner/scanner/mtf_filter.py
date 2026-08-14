"""Multi-timeframe confirmation filter (analysis only).

For MEDIUM/HIGH signals, require 1d and 1wk directional agreement.
Does NOT change scoring thresholds. Live default is OFF until approved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from scanner.opportunity import Opportunity

MTF_PAIR = ("1d", "1wk")


def directions_agree(a: str, b: str) -> bool:
    """True only when both sides are the same actionable direction."""
    if a not in ("bullish", "bearish") or b not in ("bullish", "bearish"):
        return False
    return a == b


def apply_mtf_filter(
    opportunities: Iterable[Opportunity],
    *,
    enabled: bool = True,
    downgrade: bool = True,
) -> tuple[list[Opportunity], dict[str, int]]:
    """Filter/downgrade MEDIUM/HIGH opportunities that lack 1d↔1wk agreement.

    Returns (filtered_list, stats).
    stats keys:
      - medium_high_total
      - suppressed_disagree
      - suppressed_missing_pair
      - kept_agree
      - low_untouched
    """
    opps = list(opportunities)
    stats = {
        "medium_high_total": 0,
        "suppressed_disagree": 0,
        "suppressed_missing_pair": 0,
        "kept_agree": 0,
        "low_untouched": 0,
    }
    if not enabled:
        return opps, stats

    # Index latest opp per instrument+timeframe (scan already one per pair)
    by_key: dict[tuple[str, str], Opportunity] = {
        (o.instrument, o.timeframe): o for o in opps
    }

    out: list[Opportunity] = []
    for opp in opps:
        if opp.confidence == "LOW":
            stats["low_untouched"] += 1
            out.append(opp)
            continue
        if opp.confidence not in ("HIGH", "MEDIUM"):
            out.append(opp)
            continue
        if opp.timeframe not in MTF_PAIR:
            # Other TFs (1h/4h): leave unchanged when pair not defined
            out.append(opp)
            continue

        stats["medium_high_total"] += 1
        other_tf = "1wk" if opp.timeframe == "1d" else "1d"
        other = by_key.get((opp.instrument, other_tf))

        if other is None or other.direction not in ("bullish", "bearish"):
            stats["suppressed_missing_pair"] += 1
            out.append(_suppress_or_downgrade(opp, downgrade, reason_missing=True, other=other))
            continue

        if directions_agree(opp.direction, other.direction):
            stats["kept_agree"] += 1
            # Annotate agreement in reason (does not change score/thresholds)
            noted = replace(
                opp,
                reason=(
                    opp.reason
                    + f" MTF confirm: {opp.timeframe} and {other_tf} both {opp.direction}."
                ),
            )
            out.append(noted)
        else:
            stats["suppressed_disagree"] += 1
            out.append(
                _suppress_or_downgrade(
                    opp,
                    downgrade,
                    reason_missing=False,
                    other=other,
                )
            )

    return out, stats


def _suppress_or_downgrade(
    opp: Opportunity,
    downgrade: bool,
    *,
    reason_missing: bool,
    other: Optional[Opportunity],
) -> Opportunity:
    if reason_missing:
        detail = (
            f"MTF filter: no clear {('1wk' if opp.timeframe == '1d' else '1d')} "
            "direction to confirm this MEDIUM/HIGH signal."
        )
    else:
        assert other is not None
        detail = (
            f"MTF filter: {opp.timeframe} {opp.direction} disagrees with "
            f"{other.timeframe} {other.direction} — suppressed as stronger opportunity."
        )

    if not downgrade:
        # Hard suppress: mark as non-actionable
        return replace(
            opp,
            confidence="NO STRONG SETUP",
            direction="neutral",
            reason=detail + " " + opp.reason,
        )

    # Downgrade path: demote to NO STRONG SETUP (not LOW — avoids fake LOW sample)
    return replace(
        opp,
        confidence="NO STRONG SETUP",
        direction="neutral",
        reason=detail + " " + opp.reason,
    )
