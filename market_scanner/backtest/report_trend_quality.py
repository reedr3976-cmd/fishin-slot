"""Beginner report for trend-quality study."""

from __future__ import annotations

from typing import Any, Optional


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x * 100:.2f}%"


def _pf(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


def _row(m: dict[str, Any]) -> str:
    return (
        f"n={m.get('signals', 0)} win={_pct(m.get('win_rate'))} "
        f"avg={_pct(m.get('avg_return'))} PF={_pf(m.get('profit_factor'))} "
        f"dd={_pct(m.get('max_drawdown'))} cum={_pct(m.get('cumulative_return'))}"
    )


def build_trend_quality_report(result: dict[str, Any]) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  TREND-QUALITY FILTER STUDY  ·  Analysis only            ║",
        "║  4H ORIGINAL control  ·  Walk-forward folds  ·  2× costs ║",
        "║  Live scanner UNCHANGED                                  ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Mode: {result['mode']}",
        f"Note: {result['note']}",
        f"Train thresholds: {result['train_thresholds']}",
        f"ORIGINAL TEST n={result['original_test_n']} "
        f"avg={_pct(result['original_test_avg'])} dd={_pct(result['original_test_dd'])}",
        "",
        "Robustness gate: TEST avg > ORIGINAL, ≥3/4 folds avg>0, 2× cost avg>0,",
        f"n≥{result['min_signals']}, DD ≤ ORIGINAL+5pp, ≥2 symbols with positive avg.",
        "",
    ]

    for name, block in result["candidates"].items():
        lines.append(f"[{name}] ({block['scope']}) {block['description']}")
        lines.append(
            f"  TEST 1× {_row(block['test_1x']['all'])}  "
            f"removed={block['removed_vs_original']}  "
            f"syms+={block['test_1x']['n_symbols_positive_avg']}  "
            f"PASS={block['passes_robustness']}"
        )
        lines.append(f"  TEST 2× {_row(block['test_2x']['all'])}")
        for cls in ("commodity", "forex", "stocks"):
            lines.append(f"    {cls}: {_row(block['test_1x']['by_asset_class'][cls])}")
        lines.append(f"  Folds positive: {block['folds_positive']}/4")
        for fname, pack in block["folds"].items():
            lines.append(
                f"    {fname}: {_row(pack['all'])} | "
                f"C {_row(pack['by_asset_class']['commodity'])} | "
                f"F {_row(pack['by_asset_class']['forex'])} | "
                f"S {_row(pack['by_asset_class']['stocks'])}"
            )
        syms = sorted(
            block["test_1x"]["by_symbol"].items(),
            key=lambda kv: (kv[1].get("avg_return") is not None, kv[1].get("avg_return") or -9),
            reverse=True,
        )
        lines.append("  Symbols:")
        for s, m in syms:
            lines.append(f"    {s}: {_row(m)}")
        lines.append("")

    passed = [n for n, b in result["candidates"].items() if b.get("passes_robustness")]
    near = []
    for n, b in result["candidates"].items():
        if n == "ORIGINAL" or b.get("passes_robustness"):
            continue
        t = b["test_1x"]["all"]
        if (
            (t.get("avg_return") or -1) > (result["original_test_avg"] or 0)
            and b["folds_positive"] >= 2
            and t.get("signals", 0) >= result["min_signals"]
        ):
            near.append(n)

    # Architecture recommendation
    univ_pass = [n for n in passed if result["candidates"][n]["scope"] == "universal"]
    class_pass = [n for n in passed if result["candidates"][n]["scope"] != "universal"]
    if univ_pass:
        arch = (
            "A) Evidence supports exploring ONE universal trend-quality filter "
            f"({', '.join(univ_pass)}), but only after explicit approval."
        )
    elif class_pass:
        arch = (
            "B) Only class-scoped filters passed gates — suggests separate "
            "commodity/FX/stocks trend-quality rules may be needed, not one universal rule."
        )
    elif near:
        arch = (
            "C) No candidate fully passed robustness. Near-misses "
            f"({', '.join(near)}) still fail fold/cost/DD gates — the current "
            "signal architecture (score + sma_stack + fixed hold) likely needs "
            "redesign rather than a thin quality overlay."
        )
    else:
        arch = (
            "C) No candidate improved ORIGINAL with multi-fold robustness. "
            "Trend-quality overlays on the current architecture are not enough; "
            "a broader redesign (entries/exits/regime handling) is more appropriate."
        )

    lines.extend(
        [
            "════════════════════════════════════════════════════════════",
            "ARCHITECTURE READ",
            "════════════════════════════════════════════════════════════",
            f"  {arch}",
            "",
            "════════════════════════════════════════════════════════════",
            "VERDICT",
            "════════════════════════════════════════════════════════════",
            "",
            "1) KEEP AS-IS",
            "  • Live ORIGINAL scanner defaults unchanged.",
            "  • Crypto off; no experimental filters enabled.",
            "",
            "2) STRONG CANDIDATE",
        ]
    )
    if passed:
        for n in passed:
            lines.append(f"  • {n}: {result['candidates'][n]['description']}")
    else:
        lines.append("  • None. No filter passed the robustness requirements.")

    lines.append("")
    lines.append("3) STUDY FURTHER")
    if near:
        for n in near:
            lines.append(f"  • {n} (near-miss; failed full gate)")
    else:
        lines.append("  • Broader redesign of trend definition / exit logic.")
    lines.append("  • Longer history / true rolling walk-forward if more 4H data becomes available.")

    lines.extend(
        [
            "",
            "4) REJECT",
            "  • Enabling any trend-quality filter live without approval.",
            "  • Per-symbol parameter fitting or deleting symbols from train losses.",
        ]
    )
    rejected = [
        n
        for n, b in result["candidates"].items()
        if n != "ORIGINAL" and n not in passed and n not in near
    ]
    for n in rejected[:10]:
        lines.append(f"  • {n} as a live rule.")

    lines.extend(
        [
            "",
            "SAFEST NEXT LIVE CHANGE:",
            "  NO LIVE CHANGE."
            if not passed
            else (
                "  Still wait for explicit approval before enabling "
                + ", ".join(passed)
                + "."
            ),
            "",
        ]
    )
    return "\n".join(lines)
