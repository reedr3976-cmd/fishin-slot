"""Report for asset-class / symbol validation study."""

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
        f"avg={_pct(m.get('avg_return'))} med={_pct(m.get('median_return'))} "
        f"cum={_pct(m.get('cumulative_return'))} "
        f"avgW={_pct(m.get('avg_winner'))} avgL={_pct(m.get('avg_loser'))} "
        f"PF={_pf(m.get('profit_factor'))} dd={_pct(m.get('max_drawdown'))} "
        f"rel={m.get('reliable')}"
    )


def build_asset_class_validation_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  ASSET-CLASS / SYMBOL VALIDATION  ·  Analysis only       ║",
        "║  Control = ORIGINAL · 4H trending MH · fixed hold        ║",
        "║  Live scanner UNCHANGED · No TP enable · No merges       ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Mode: {result['mode']}",
        f"Control: {result['control']}",
        f"Train/Test: {result['train_fraction']:.0%} / {1 - result['train_fraction']:.0%}",
        f"Instruments: {', '.join(result['instruments'])}",
        f"Bars: {result['bars_loaded']}",
        "",
        "════════════════════════════════════════════════════════════",
        "1) TEST BY ASSET CLASS (ORIGINAL fixed hold)",
        "════════════════════════════════════════════════════════════",
        f"  ALL: {_row(result['baseline_test']['all'])}",
    ]
    for cls, m in result["baseline_test"]["by_asset_class"].items():
        lines.append(f"  {cls}: {_row(m)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "2) TEST BY SYMBOL",
            "════════════════════════════════════════════════════════════",
        ]
    )
    syms = sorted(
        result["baseline_test"]["by_symbol"].items(),
        key=lambda kv: (kv[1].get("avg_return") is not None, kv[1].get("avg_return") or -9),
        reverse=True,
    )
    for inst, m in syms:
        lines.append(f"  {inst} ({m['asset_class']}): {_row(m)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "3) WHY STRONG (USOIL/XAGUSD) VS WEAK (EURUSD/QQQ/…)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for side in ("strong", "weak"):
        p = result["why_strong_vs_weak"][side]
        lines.append(f"  [{side.upper()}] symbols={p.get('symbols')} n={p.get('n')}")
        if p.get("n", 0) == 0:
            continue
        lines.append(f"    metrics: {_row(p['metrics'])}")
        lines.append(
            f"    avg_score={p['avg_score']:.1f} atr%={_pct(p['avg_atr_pct'])} "
            f"bullish={_pct(p['bullish_share'])}"
        )
        fr = p["feature_rates"]
        lines.append(
            "    features: "
            + ", ".join(f"{k}={_pct(v)}" for k, v in fr.items())
        )

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "4) COMMODITY EDGE ACROSS CHRONOLOGICAL PERIODS",
            "════════════════════════════════════════════════════════════",
            "  Full-sample thirds (not parameter-fit):",
        ]
    )
    for name, block in result["period_thirds"].items():
        lines.append(f"  [{name}] ALL {_row(block['all'])}")
        for cls in ("commodity", "forex", "stocks"):
            lines.append(f"    {cls}: {_row(block['by_asset_class'][cls])}")
        for s, m in block["commodity_symbols"].items():
            lines.append(f"    {s}: {_row(m)}")

    lines.extend(
        [
            "",
            "  Inside TEST only — early vs late half:",
        ]
    )
    for half in ("early", "late"):
        block = result["test_early_vs_late"][half]
        lines.append(f"  [TEST {half}] ALL {_row(block['all'])}")
        for cls in ("commodity", "forex", "stocks"):
            lines.append(f"    {cls}: {_row(block['by_asset_class'][cls])}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "5) ATR STOPS VS ORIGINAL FIXED HOLD (same TEST entries)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for pol, block in result["stops_test"].items():
        lines.append(f"  [{pol}] ALL {_row(block['all'])}")
        for cls in ("commodity", "forex", "stocks"):
            lines.append(f"    {cls}: {_row(block['by_asset_class'][cls])}")
        # Symbols with n >= 15
        big = [
            (s, m)
            for s, m in block["by_symbol"].items()
            if m.get("signals", 0) >= 15
        ]
        big.sort(key=lambda kv: kv[0])
        if big:
            lines.append("    Symbols with n≥15:")
            for s, m in big:
                lines.append(f"      {s}: {_row(m)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "6) COMMODITY COST / SPREAD STRESS (fixed hold, same entries)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for label, block in result["commodity_cost_stress"].items():
        lines.append(
            f"  [{label}] cost={_pct(block['round_trip_cost'])}  "
            f"commodity {_row(block['commodity'])}"
        )
        lines.append(f"    USOIL {_row(block['USOIL'])}")
        lines.append(f"    XAGUSD {_row(block['XAGUSD'])}")
        lines.append(f"    XAUUSD {_row(block['XAUUSD'])}")

    excl = result["symbol_exclusion_study"]
    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "7) SYMBOL EXCLUSION STUDY (NOT applied live)",
            "════════════════════════════════════════════════════════════",
            f"  Rule: {excl['rule']}",
            "",
            f"  A) Exclude TRAIN-negative (n≥15): {excl['exclude_negative_train']['symbols']}",
            f"     TEST baseline: {_row(excl['exclude_negative_train']['test_baseline'])}",
            f"     TEST kept:     {_row(excl['exclude_negative_train']['test_kept'])}",
            f"     TEST removed:  {_row(excl['exclude_negative_train']['test_removed'])}",
            "",
            f"  B) Exclude TRAIN worst-3 (n≥20): {excl['exclude_worst3_train']['symbols']}",
            f"     TRAIN ranks: {excl['exclude_worst3_train']['train_ranks']}",
            f"     TEST baseline: {_row(excl['exclude_worst3_train']['test_baseline'])}",
            f"     TEST kept:     {_row(excl['exclude_worst3_train']['test_kept'])}",
            f"     TEST removed:  {_row(excl['exclude_worst3_train']['test_removed'])}",
        ]
    )

    # Verdict buckets
    base_cls = result["baseline_test"]["by_asset_class"]
    stops = result["stops_test"]
    fh_c = base_cls["commodity"]["avg_return"]
    s15_c = stops["stop_1.5atr"]["by_asset_class"]["commodity"]["avg_return"]
    fh_f = base_cls["forex"]["avg_return"]
    s15_f = stops["stop_1.5atr"]["by_asset_class"]["forex"]["avg_return"]
    fh_s = base_cls["stocks"]["avg_return"]
    s15_s = stops["stop_1.5atr"]["by_asset_class"]["stocks"]["avg_return"]

    # Commodity period consistency
    comm_avgs = [
        result["period_thirds"][p]["by_asset_class"]["commodity"].get("avg_return")
        for p in ("P1_early", "P2_mid", "P3_late")
    ]
    comm_positive_periods = sum(1 for a in comm_avgs if a is not None and a > 0)

    cost_ok = True
    for label in ("stress_20bps", "stress_30bps"):
        m = result["commodity_cost_stress"][label]["commodity"]
        if m.get("avg_return") is None or m["avg_return"] <= 0:
            cost_ok = False

    excl_neg = excl["exclude_negative_train"]
    excl_helps = (
        excl_neg["test_kept"].get("avg_return") is not None
        and excl_neg["test_baseline"].get("avg_return") is not None
        and excl_neg["test_kept"]["avg_return"] > excl_neg["test_baseline"]["avg_return"]
        and excl_neg["test_kept"].get("signals", 0) >= result["min_signals"]
    )

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "VERDICT TABLE",
            "════════════════════════════════════════════════════════════",
            "",
            "KEEP AS-IS",
            "  • ORIGINAL scoring thresholds and entry logic (control).",
            "  • Crypto disabled.",
            "  • No ATR take-profit (already rejected).",
            "  • Live scanner defaults unchanged until explicit approval.",
            "",
            "STRONG CANDIDATE",
        ]
    )
    if (
        fh_c is not None
        and fh_c > 0
        and comm_positive_periods >= 2
        and cost_ok
        and base_cls["commodity"].get("signals", 0) >= result["min_signals"]
    ):
        lines.append(
            "  • Commodity focus on 4H trending MH (USOIL/XAGUSD drive edge; "
            f"positive in {comm_positive_periods}/3 chronological thirds; "
            "survives 2–3× cost stress)."
        )
    else:
        lines.append(
            "  • (None automatic) Commodity edge needs more caution — see STUDY FURTHER."
        )

    if (
        s15_c is not None
        and fh_c is not None
        and s15_c >= fh_c
        and stops["stop_1.5atr"]["by_asset_class"]["commodity"].get("max_drawdown")
        is not None
        and base_cls["commodity"].get("max_drawdown") is not None
        and stops["stop_1.5atr"]["by_asset_class"]["commodity"]["max_drawdown"]
        <= base_cls["commodity"]["max_drawdown"] + 0.01
    ):
        lines.append(
            "  • 1.5 ATR stop for COMMODITIES only (same entries; risk-aware vs fixed hold)."
        )
    else:
        lines.append("  • (No broad strong stop candidate auto-tagged.)")

    lines.extend(["", "STUDY FURTHER"])
    lines.append(
        f"  • 1.5 ATR stop by class: commodity Δavg="
        f"{_pct(None if s15_c is None or fh_c is None else s15_c - fh_c)}; "
        f"forex Δavg={_pct(None if s15_f is None or fh_f is None else s15_f - fh_f)}; "
        f"stocks Δavg={_pct(None if s15_s is None or fh_s is None else s15_s - fh_s)}."
    )
    lines.append(
        "  • Symbol exclusion rules (train-selected): "
        + (
            "TEST avg improved — still do not remove live without approval."
            if excl_helps
            else "TEST improvement not clearly robust — do not exclude yet."
        )
    )
    lines.append(
        "  • Gold (XAUUSD) vs oil/silver split inside commodities."
    )
    lines.append(
        "  • Whether forex/stocks should stay in alerts-only / lower priority."
    )

    lines.extend(
        [
            "",
            "REJECT",
            "  • Enabling ATR take-profit.",
            "  • Enabling 1.5 ATR stop on forex/stocks from this evidence alone.",
            "  • Per-symbol stop/parameter optimization.",
            "  • Auto-removing symbols from the live catalog without approval.",
            "  • Merging any change into the live scanner in this step.",
            "",
            "SAFEST NEXT LIVE CHANGE (if any) — awaiting your approval:",
        ]
    )

    if (
        fh_c is not None
        and fh_c > 0
        and comm_positive_periods >= 2
        and cost_ok
    ):
        lines.append(
            "  Prefer a narrow, reversible product change: keep ORIGINAL rules, "
            "keep crypto off, and if you enable anything next, make 4H trending "
            "alerts emphasize commodities (oil/silver) without deleting other "
            "symbols yet. Do NOT enable ATR stops or TP live until you say so."
        )
    else:
        lines.append(
            "  Safest path: make NO live change. Continue analysis-only until "
            "commodity robustness and stop behavior are clearer."
        )

    lines.append("")
    return "\n".join(lines)
