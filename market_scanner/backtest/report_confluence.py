"""Beginner-friendly confluence A/B/C/D report."""

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


def build_confluence_report(result: dict[str, Any]) -> str:
    titles = {
        "A_original": "A — Original (baseline)",
        "B_require_macd_strong": "B — Require macd_strong (MEDIUM/HIGH)",
        "C_require_directional_sr": "C — Require directional S/R confluence",
        "D_macd_strong_and_sr": "D — macd_strong AND directional S/R",
    }
    fr = result["feature_rates_test_mh"]
    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  CONFLUENCE STUDY  ·  MACD / S/R  ·  Analysis only       ║",
        "║  A original · B macd_strong · C S/R · D both             ║",
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
        "TEST MEDIUM+HIGH feature hits (before filters):",
        f"  MH total:            {fr['mh_total']}",
        f"  macd_strong:         {fr['macd_strong']}",
        f"  directional S/R:     {fr['directional_sr']}",
        f"  both (macd+S/R):     {fr['both']}",
        "",
        "Signals removed vs original (MEDIUM+HIGH only; LOW always kept):",
    ]
    for key, title in titles.items():
        rc = result["removal_counts"][key]
        lines.append(
            f"  {title}: removed_MH={rc['removed_mh']} kept_MH={rc['kept_mh']} "
            f"kept_total={rc['kept_total']}"
        )

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "POLICY COMPARISON ON CHRONOLOGICAL TEST",
            "════════════════════════════════════════════════════════════",
        ]
    )

    for key, title in titles.items():
        pol = result["policies_test"][key]
        lines.append("")
        lines.append(f"[{title}]")
        lines.append(f"  OVERALL:     {_line(pol['overall'])}")
        lines.append(f"  MEDIUM+HIGH: {_line(pol['medium_high'])}")
        for conf in ("HIGH", "MEDIUM", "LOW"):
            lines.append(f"  {conf}: {_line(pol['by_confidence'][conf])}")
        lines.append("  By asset class (all):")
        for cls, bag in pol["by_asset_class"].items():
            lines.append(f"    {cls}: {_line(bag)}")
        lines.append("  By asset class (MEDIUM+HIGH):")
        for cls, bag in pol["by_asset_class_mh"].items():
            lines.append(f"    {cls}: {_line(bag)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "BREADTH — what each filter removes / concentration",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for key, title in titles.items():
        if key == "A_original":
            continue
        br = result["breadth_test"][key]
        rem = br["removed_summary"]
        kept = br["kept_summary"]
        lines.append("")
        lines.append(f"[{title}]")
        lines.append(f"  Removed MH: {_line(rem)}")
        lines.append(f"  Kept MH:    {_line(kept)}")
        lines.append(
            f"  Sum of removed net returns: {_pct(br['removed_sum_net'])}"
        )
        lines.append("  Removed by asset class:")
        for cls, info in br["removed_by_asset_class"].items():
            lines.append(
                f"    {cls}: n={info['n']} avg={_pct(info['avg_net'])} "
                f"sum={_pct(info['sum_net'])}"
            )
        lines.append("  Removed by instrument (largest |sum| first):")
        for row in br["removed_by_instrument"][:10]:
            lines.append(
                f"    {row['instrument']} ({row['asset_class']}): n={row['n']} "
                f"avg={_pct(row['avg_net'])} sum={_pct(row['sum_net'])}"
            )
        conc = br["concentration"]
        lines.append(
            f"  Concentration (vs removed sum): worst1={_pct(conc['worst_1_share_of_removed_sum'])} "
            f"worst3={_pct(conc['worst_3_share_of_removed_sum'])} "
            f"best1={_pct(conc['best_1_share_of_removed_sum'])} "
            f"best3={_pct(conc['best_3_share_of_removed_sum'])}"
        )
        if br["leave_one_instrument_out_kept_mh"]:
            lines.append("  Leave-one-instrument-out (kept MH avg − A MH avg):")
            for row in br["leave_one_instrument_out_kept_mh"][:6]:
                lines.append(
                    f"    drop {row['dropped_instrument']}: n={row['n']} "
                    f"avg={_pct(row['avg'])} ΔvsA={_pct(row['delta_vs_A_mh_avg'])}"
                )

    # Conclusion
    a_mh = result["policies_test"]["A_original"]["medium_high"]
    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "BEGINNER CONCLUSION",
            "════════════════════════════════════════════════════════════",
            f"  A MEDIUM+HIGH: {_line(a_mh)}",
        ]
    )

    scores: list[tuple[str, str, MetricBag, int]] = []
    for key, title in titles.items():
        if key == "A_original":
            continue
        bag = result["policies_test"][key]["medium_high"]
        removed = result["removal_counts"][key]["removed_mh"]
        scores.append((key, title, bag, removed))
        lines.append(f"  {title}: {_line(bag)}  (removed {removed} MH)")

    def avg_ok(bag: MetricBag) -> bool:
        return (
            a_mh.avg_return is not None
            and bag.avg_return is not None
            and bag.avg_return > a_mh.avg_return
        )

    def cum_ok(bag: MetricBag) -> bool:
        return (
            a_mh.cumulative_return is not None
            and bag.cumulative_return is not None
            and bag.cumulative_return > a_mh.cumulative_return
        )

    # Breadth: MH avg lift in both forex and commodity vs A
    def class_broad(key: str) -> bool:
        a_cls = result["policies_test"]["A_original"]["by_asset_class_mh"]
        b_cls = result["policies_test"][key]["by_asset_class_mh"]
        lifts = 0
        for cls in ("forex", "commodity"):
            aa = a_cls.get(cls)
            bb = b_cls.get(cls)
            if (
                aa
                and bb
                and aa.avg_return is not None
                and bb.avg_return is not None
                and bb.avg_return > aa.avg_return
                and bb.signals >= 10
            ):
                lifts += 1
        return lifts >= 2

    lines.append("")
    candidates: list[str] = []
    for key, title, bag, removed in scores:
        rem_bag = result["breadth_test"][key]["removed_summary"]
        discards_good = (
            rem_bag.avg_return is not None
            and bag.avg_return is not None
            and rem_bag.avg_return > bag.avg_return
        )
        reliable = bag.signals >= MIN_SIGNALS_FOR_CONCLUSION
        broad = class_broad(key)
        improves = avg_ok(bag) and (cum_ok(bag) or avg_ok(bag))
        note_bits = []
        if not reliable:
            note_bits.append(f"sample small (n={bag.signals})")
        if avg_ok(bag):
            note_bits.append("MH avg ↑ vs A")
        else:
            note_bits.append("MH avg not ↑ vs A")
        if cum_ok(bag):
            note_bits.append("MH cum ↑ vs A")
        if broad:
            note_bits.append("lift in both asset classes")
        else:
            note_bits.append("not broad across asset classes")
        if discards_good:
            note_bits.append("removed trades had higher avg than kept (discards good setups)")
        lines.append(f"  Verdict {title}: {'; '.join(note_bits)}.")

        if reliable and avg_ok(bag) and broad and not discards_good:
            candidates.append(key)

    lines.append("")
    if not candidates:
        # Soft recommendation: best MH avg among reliable policies that beat A
        reliable_beaters = [
            (key, title, bag)
            for key, title, bag, _ in scores
            if bag.signals >= MIN_SIGNALS_FOR_CONCLUSION and avg_ok(bag)
        ]
        if reliable_beaters:
            best = max(reliable_beaters, key=lambda x: x[2].avg_return or -999)
            if class_broad(best[0]):
                rec = f"Lean toward {best[1]} — but confirm breadth before enabling."
            else:
                rec = (
                    "KEEP ORIGINAL — some filters beat A on MH avg, but improvement "
                    "is not clearly broad-based; do not enable live yet."
                )
        else:
            rec = (
                "KEEP ORIGINAL — none of B/C/D showed a reliable, broad MH average "
                "improvement without concerning sample or discard issues."
            )
        # D special note
        d_bag = result["policies_test"]["D_macd_strong_and_sr"]["medium_high"]
        if d_bag.signals < MIN_SIGNALS_FOR_CONCLUSION:
            lines.append(
                f"  Policy D sample (n={d_bag.signals}) is below "
                f"{MIN_SIGNALS_FOR_CONCLUSION} — treat D as exploratory only."
            )
    else:
        # Prefer least aggressive among candidates
        order = [
            "B_require_macd_strong",
            "C_require_directional_sr",
            "D_macd_strong_and_sr",
        ]
        pick = next(k for k in order if k in candidates)
        rec = f"Candidate for further study: {titles[pick]} (not live yet)."

    lines.extend(
        [
            f"  Recommendation: {rec}",
            "",
            "Analysis only. Live scanner unchanged. Do not merge/enable without approval.",
            "",
        ]
    )
    return "\n".join(lines)
