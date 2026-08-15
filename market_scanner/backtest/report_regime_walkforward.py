"""Report for regime / walk-forward study."""

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


def build_regime_walkforward_report(result: dict[str, Any]) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  REGIME / WALK-FORWARD STUDY  ·  Analysis only           ║",
        "║  Why commodities work in recent TEST but not all periods ║",
        "║  Live scanner UNCHANGED                                  ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Mode: {result['mode']}",
        f"Note: {result['note']}",
        f"Train medians: {result['train_medians']}",
        "",
        "════════════════════════════════════════════════════════════",
        "1) COMMODITY REGIMES BY CHRONOLOGICAL FOLD",
        "════════════════════════════════════════════════════════════",
    ]
    for fold, block in result["fold_regime_diagnostics"].items():
        lines.append(f"  [{fold}] window={block['window']}")
        lines.append(f"    ALL {_row(block['all']['all'])}")
        lines.append(
            f"    commodity {_row(block['all']['by_asset_class']['commodity'])}"
        )
        lines.append(f"    oil+silver {_row(block['all']['oil_silver'])}")
        lines.append("    Commodity regimes:")
        for lab, m in block["commodity_regimes"].items():
            lines.append(f"      {lab}: {_row(m)}")
        lines.append("    Oil/silver regimes:")
        for lab, m in block["oil_silver_regimes"].items():
            lines.append(f"      {lab}: {_row(m)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "2) CANDIDATES VS ORIGINAL (TEST + fold consistency + costs)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for name, block in result["candidates"].items():
        lines.append("")
        lines.append(f"[{name}] {block['description']}")
        lines.append(
            f"  TEST ALL {_row(block['test']['all'])}  "
            f"(removed vs ORIG {block['removed_vs_original_test']})"
        )
        for cls in ("commodity", "forex", "stocks"):
            lines.append(f"    {cls}: {_row(block['test']['by_asset_class'][cls])}")
        lines.append(f"    oil+silver: {_row(block['test']['oil_silver'])}")
        lines.append(
            f"  Folds positive: commodity={block['folds_commodity_positive']}/4 "
            f"oil+silver={block.get('folds_oil_silver_positive', '?')}/4 "
            f"all={block['folds_all_positive']}/4"
        )
        for fname, pack in block["folds"].items():
            lines.append(
                f"    {fname}: ALL {_row(pack['all'])} | "
                f"comm {_row(pack['by_asset_class']['commodity'])} | "
                f"OS {_row(pack['oil_silver'])}"
            )
        for clab, pack in block["cost_stress_test"].items():
            lines.append(
                f"  cost {clab}: ALL {_row(pack['all'])} | "
                f"comm {_row(pack['by_asset_class']['commodity'])}"
            )
        # top/bottom symbols on TEST
        syms = sorted(
            block["test"]["by_symbol"].items(),
            key=lambda kv: (kv[1].get("avg_return") is not None, kv[1].get("avg_return") or -9),
            reverse=True,
        )
        lines.append("  TEST symbols (best→worst):")
        for s, m in syms:
            lines.append(f"    {s}: {_row(m)}")

    # Verdict
    orig = result["candidates"]["ORIGINAL"]
    strong = []
    study = []
    reject = []

    for name, block in result["candidates"].items():
        if name == "ORIGINAL":
            continue
        test_avg = block["test"]["all"].get("avg_return")
        orig_avg = orig["test"]["all"].get("avg_return")
        test_n = block["test"]["all"].get("signals", 0)
        folds_ok = block["folds_all_positive"] >= 3
        comm_folds = block["folds_commodity_positive"] >= 3
        oil_folds = block.get("folds_oil_silver_positive", 0) >= 3
        cost2 = block["cost_stress_test"]["2x"]["all"].get("avg_return")
        beats = (
            test_avg is not None
            and orig_avg is not None
            and test_avg > orig_avg
            and test_n >= result["min_signals"]
        )
        cost_ok = cost2 is not None and cost2 > 0
        if beats and folds_ok and cost_ok and (comm_folds or oil_folds or "FX_" in name or "STK_" in name):
            # require class-relevant fold consistency
            if name.startswith("COMM_") and not (comm_folds or oil_folds):
                study.append(name)
            elif name.startswith("FX_") and block["folds"]["F4"]["by_asset_class"]["forex"].get("avg_return", -1) <= 0:
                study.append(name)
            else:
                strong.append(name)
        elif beats or oil_folds:
            study.append(name)
        else:
            reject.append(name)

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "VERDICT",
            "════════════════════════════════════════════════════════════",
            "",
            "KEEP AS-IS",
            "  • Live ORIGINAL scanner defaults (unchanged).",
            "  • Crypto off; no ATR take-profit; no train-based symbol deletion.",
            "",
            "STRONG CANDIDATE",
        ]
    )
    if strong:
        for n in strong:
            lines.append(f"  • {n}: {result['candidates'][n]['description']}")
    else:
        lines.append(
            "  • None. No candidate beat ORIGINAL on TEST avg, stayed mostly "
            "positive across ≥3/4 folds, and survived 2× costs."
        )

    lines.append("")
    lines.append("STUDY FURTHER")
    if study:
        for n in study:
            lines.append(f"  • {n}")
    else:
        lines.append("  • (see commodity regime diagnostics — edge is regime/period specific)")
    lines.append(
        "  • Oil/silver edge appears concentrated in high-vol and/or late folds; "
        "confirm with more history before live emphasis."
    )

    lines.append("")
    lines.append("REJECT")
    lines.append("  • Enabling any candidate live without approval.")
    for n in reject[:8]:
        lines.append(f"  • {n} as a live rule (failed consistency and/or did not beat ORIGINAL).")
    if len(reject) > 8:
        lines.append(f"  • …and {len(reject) - 8} other weak candidates.")

    lines.extend(
        [
            "",
            "SAFEST NEXT LIVE CHANGE:",
            "  Make NO live change until a candidate is positive on TEST, "
            "positive in ≥3/4 chronological folds, and still positive at 2× costs.",
            "",
        ]
    )
    return "\n".join(lines)
