"""Beginner-friendly ATR regime A vs B report."""

from __future__ import annotations

from typing import Any, Optional

from config import MIN_SIGNALS_FOR_CONCLUSION
from backtest.metrics import MetricBag


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _line(bag: MetricBag) -> str:
    return (
        f"n={bag.signals} win={_pct(bag.win_rate)} avg={_pct(bag.avg_return)} "
        f"med={_pct(bag.median_return)} cum={_pct(bag.cumulative_return)} "
        f"dd={_pct(bag.max_drawdown)} reliable={bag.reliable}"
    )


def build_atr_regime_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  ATR / VOLATILITY REGIME STUDY  ·  Analysis only         ║",
        "║  A = Original  ·  B = Suppress MEDIUM/HIGH if high_atr   ║",
        "║  Scoring thresholds UNCHANGED · Crypto disabled          ║",
        "║  Live scanner NOT changed · DO NOT enable/merge yet      ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {result['mode']}",
        f"Train/Test: {result['train_fraction']:.0%} / {1 - result['train_fraction']:.0%}",
        f"Instruments: {', '.join(result['instruments'])}",
        f"Timeframes: {', '.join(result['timeframes'])}",
        f"Bars loaded: {result['bars_loaded']}",
        f"Note: {result['note']}",
        "",
        "TEST signal counts:",
        f"  Total original signals:     {result['counts']['test_total']}",
        f"  MEDIUM/HIGH with high_atr:  {result['counts']['test_mh_high_atr_removed']}  (removed by B)",
        f"  Kept under A:               {result['counts']['test_a_kept']}",
        f"  Kept under B:               {result['counts']['test_b_kept']}",
        "",
        "════════════════════════════════════════════════════════════",
        "POLICY COMPARISON ON CHRONOLOGICAL TEST",
        "════════════════════════════════════════════════════════════",
    ]

    for key, title in (
        ("A_original", "A — Original (no ATR gate)"),
        ("B_suppress_high_atr_mh", "B — Suppress MEDIUM/HIGH when high_atr"),
    ):
        pol = result["policies_test"][key]
        lines.append("")
        lines.append(f"[{title}]")
        lines.append(f"  OVERALL:     {_line(pol['overall'])}")
        lines.append(f"  MEDIUM+HIGH: {_line(pol['medium_high'])}")
        for conf in ("HIGH", "MEDIUM", "LOW"):
            lines.append(f"  {conf}: {_line(pol['by_confidence'][conf])}")
        lines.append("  By asset class (all confidences):")
        for cls, bag in pol["by_asset_class"].items():
            lines.append(f"    {cls}: {_line(bag)}")
        lines.append("  By asset class (MEDIUM+HIGH only):")
        for cls, bag in pol["by_asset_class_mh"].items():
            lines.append(f"    {cls}: {_line(bag)}")

    br = result["breadth_test"]
    rem = br["removed_mh_summary"]
    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "REMOVED MEDIUM/HIGH (high_atr) — what B drops",
            "════════════════════════════════════════════════════════════",
            f"  {_line(rem)}",
            f"  Sum of removed trade net returns: {_pct(br['removed_sum_net'])}",
            "  Removed by asset class:",
        ]
    )
    for cls, info in br["removed_by_asset_class"].items():
        share = _pct(info["share_of_removed_sum"]) if info["share_of_removed_sum"] is not None else "n/a"
        avg = _pct(info["avg_net"]) if info["avg_net"] is not None else "n/a"
        lines.append(
            f"    {cls}: n={info['n']} avg={avg} sum={_pct(info['sum_net'])} "
            f"share_of_removed_sum={share}"
        )
    lines.append("  Removed by instrument (largest |sum| first):")
    for row in br["removed_by_instrument"][:12]:
        lines.append(
            f"    {row['instrument']} ({row['asset_class']}): n={row['n']} "
            f"avg={_pct(row['avg_net'])} sum={_pct(row['sum_net'])}"
        )
    conc = br["concentration"]
    lines.extend(
        [
            "  Concentration of removed PnL:",
            f"    Worst 1 trade share of removed sum: {_pct(conc['worst_1_share_of_removed_sum'])}",
            f"    Worst 3 trades share of removed sum: {_pct(conc['worst_3_share_of_removed_sum'])}",
            f"    Best 1 trade share of removed sum:  {_pct(conc['best_1_share_of_removed_sum'])}",
            f"    Best 3 trades share of removed sum:  {_pct(conc['best_3_share_of_removed_sum'])}",
            f"  Kept MH avg vs removed MH avg: "
            f"{_pct(br['kept_vs_removed_mh']['kept_avg'])} vs "
            f"{_pct(br['kept_vs_removed_mh']['removed_avg'])} "
            f"(n_kept={br['kept_vs_removed_mh']['kept_n']}, "
            f"n_removed={br['kept_vs_removed_mh']['removed_n']})",
        ]
    )

    a_mh = result["policies_test"]["A_original"]["medium_high"]
    b_mh = result["policies_test"]["B_suppress_high_atr_mh"]["medium_high"]
    a_all = result["policies_test"]["A_original"]["overall"]
    b_all = result["policies_test"]["B_suppress_high_atr_mh"]["overall"]

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "BEGINNER CONCLUSION",
            "════════════════════════════════════════════════════════════",
            f"  A MEDIUM+HIGH: {_line(a_mh)}",
            f"  B MEDIUM+HIGH: {_line(b_mh)}",
            f"  A overall:     {_line(a_all)}",
            f"  B overall:     {_line(b_all)}",
            "",
        ]
    )

    improved_mh_avg = (
        a_mh.avg_return is not None
        and b_mh.avg_return is not None
        and b_mh.avg_return > a_mh.avg_return
    )
    improved_mh_cum = (
        a_mh.cumulative_return is not None
        and b_mh.cumulative_return is not None
        and b_mh.cumulative_return > a_mh.cumulative_return
    )

    # Breadth flags
    a_cls = result["policies_test"]["A_original"]["by_asset_class_mh"]
    b_cls = result["policies_test"]["B_suppress_high_atr_mh"]["by_asset_class_mh"]
    class_lifts = []
    for cls in sorted(set(a_cls) | set(b_cls)):
        aa = a_cls.get(cls)
        bb = b_cls.get(cls)
        if aa and bb and aa.avg_return is not None and bb.avg_return is not None:
            class_lifts.append((cls, bb.avg_return - aa.avg_return, aa.signals, bb.signals))

    broad = False
    if class_lifts:
        positive_classes = [c for c, d, _, _ in class_lifts if d > 0]
        broad = len(positive_classes) >= 2 and improved_mh_avg

    rem_n = br["removed_mh_count"]
    worst3 = conc["worst_3_share_of_removed_sum"]
    concentrated = rem_n > 0 and worst3 is not None and abs(worst3) >= 0.5

    if rem_n < MIN_SIGNALS_FOR_CONCLUSION:
        lines.append(
            f"  Removed sample is small (n={rem_n}; need ≥ {MIN_SIGNALS_FOR_CONCLUSION}). "
            "Treat any B vs A difference cautiously."
        )
    else:
        lines.append(f"  Removed MEDIUM+HIGH sample size n={rem_n} meets the minimum threshold.")

    lines.append(
        f"  MH average return: A={_pct(a_mh.avg_return)} → B={_pct(b_mh.avg_return)} "
        f"({'improved' if improved_mh_avg else 'not improved'})."
    )
    lines.append(
        f"  MH cumulative return: A={_pct(a_mh.cumulative_return)} → "
        f"B={_pct(b_mh.cumulative_return)} "
        f"({'improved' if improved_mh_cum else 'not improved'})."
    )
    kept_avg = br["kept_vs_removed_mh"]["kept_avg"]
    rem_avg = br["kept_vs_removed_mh"]["removed_avg"]
    if kept_avg is not None and rem_avg is not None:
        if rem_avg > kept_avg:
            lines.append(
                "  Caution: removed high_atr MEDIUM/HIGH had a HIGHER average return than "
                "kept signals — suppressing them can discard useful setups."
            )
        else:
            lines.append(
                "  Removed high_atr MEDIUM/HIGH had a lower average return than kept signals."
            )

    if class_lifts:
        bits = ", ".join(
            f"{c}: Δavg={d * 100:+.2f}% (n {na}→{nb})" for c, d, na, nb in class_lifts
        )
        lines.append(f"  Asset-class MH avg change A→B: {bits}")
    if broad:
        lines.append("  MH avg lift appears in more than one asset class (broad-based hint).")
    else:
        lines.append("  No broad-based MH average-return lift across asset classes.")

    # Mixed-sign sum makes "share" percentages misleading — flag that.
    if rem_n > 0 and br["removed_sum_net"] is not None:
        abs_sum = sum(abs(x["sum_net"]) for x in br["removed_by_instrument"]) or 0.0
        if abs_sum > 0 and abs(br["removed_sum_net"]) < 0.25 * abs_sum:
            lines.append(
                "  Removed trades mix large winners and losers (net sum near a wash); "
                "single-trade 'share of sum' figures can exceed 100% and are not stable."
            )
    if concentrated and rem_n >= 3:
        lines.append(
            "  Path/drawdown effects may be driven by a few extreme removed trades "
            f"(worst-3 vs removed-sum ratio ≈ {_pct(worst3)})."
        )

    lines.extend(
        [
            "",
            "Recommendation posture: analysis only. Do not enable an ATR filter live.",
            "Do not change scoring thresholds. Wait for explicit approval.",
            "",
        ]
    )
    return "\n".join(lines)
