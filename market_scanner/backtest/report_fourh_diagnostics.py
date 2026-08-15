"""Beginner-friendly 4H trending diagnostics report."""

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


def _block(title: str, metrics: dict[str, Any]) -> list[str]:
    lines = [f"[{title}]", f"  ALL MH: {_line(metrics['all'])}"]
    lines.append(f"  HIGH:   {_line(metrics['HIGH'])}")
    lines.append(f"  MEDIUM: {_line(metrics['MEDIUM'])}")
    if metrics.get("avg_r_multiple") is not None:
        lines.append(f"  Avg R-multiple: {metrics['avg_r_multiple']:.2f}")
    if metrics.get("exit_reasons"):
        bits = ", ".join(f"{k}={v}" for k, v in sorted(metrics["exit_reasons"].items()))
        lines.append(f"  Exit reasons: {bits}")
    lines.append("  By asset class:")
    for cls, bag in metrics["by_asset_class"].items():
        lines.append(f"    {cls}: {_line(bag)}")
    lines.append("  By symbol:")
    # Sort by avg return desc for readability
    syms = sorted(
        metrics["by_symbol"].items(),
        key=lambda kv: (kv[1].avg_return is not None, kv[1].avg_return or -999),
        reverse=True,
    )
    for inst, bag in syms:
        lines.append(f"    {inst}: {_line(bag)}")
    return lines


def build_fourh_diagnostics_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  4H TRENDING DIAGNOSTICS  ·  ORIGINAL scanner            ║",
        "║  Forex + Stocks + Commodities  ·  No crypto              ║",
        "║  Analysis only  ·  Live scanner UNCHANGED                ║",
        "║  No MACD / directional S/R hard filters as candidates    ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"Data mode: {result['mode']}",
        f"Train/Test: {result['train_fraction']:.0%} / {1 - result['train_fraction']:.0%}",
        f"Timeframe: {result['timeframe']}",
        f"Asset classes: {', '.join(result['asset_classes'])}",
        f"Instruments: {', '.join(result['instruments'])}",
        f"Bars loaded: {result['bars_loaded']}",
        f"Note: {result['note']}",
        "",
        "════════════════════════════════════════════════════════════",
        "1) BASELINE — MEDIUM+HIGH on 4H (fixed hold)",
        "════════════════════════════════════════════════════════════",
        "",
        "TEST — all MEDIUM/HIGH (any setup the original scorer fires):",
    ]
    lines.extend(_block("TEST all MH", result["baseline"]["test_all_mh"]))
    lines.append("")
    lines.append("TEST — trending-only MEDIUM/HIGH (require sma_stack):")
    lines.extend(_block("TEST trending MH", result["baseline"]["test_trending_mh"]))

    wl = result["winner_loser_train_trending_mh"]
    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "2) WHAT SEPARATES WINNERS FROM LOSERS? (TRAIN trending MH)",
            "════════════════════════════════════════════════════════════",
            f"  Wins={wl['n_wins']}  Losses={wl['n_losses']}",
            (
                f"  Avg score wins/losses: "
                f"{wl['avg_score_wins'] if wl['avg_score_wins'] is not None else 'n/a'} / "
                f"{wl['avg_score_losses'] if wl['avg_score_losses'] is not None else 'n/a'}"
            ),
            f"  Avg ATR% wins/losses: {_pct(wl['avg_atr_pct_wins'])} / {_pct(wl['avg_atr_pct_losses'])}",
            "  Feature presence: rate among winners minus rate among losers (TRAIN):",
        ]
    )
    for row in result["top_train_feature_deltas"]:
        lines.append(
            f"    {row['feature']}: Δ={_pct(row['delta_win_minus_loss'])} "
            f"(win_flag={_pct(row['win_rate_with_flag'])}, "
            f"loss_flag={_pct(row['loss_rate_with_flag'])}, hits={row['hits_total']})"
        )

    wlt = result["winner_loser_test_trending_mh"]
    lines.extend(
        [
            "",
            "  TEST trending MH winner/loser snapshot (not used for proposal):",
            f"    Wins={wlt['n_wins']} Losses={wlt['n_losses']} "
            f"score {wlt['avg_score_wins']}/{wlt['avg_score_losses']} "
            f"ATR% {_pct(wlt['avg_atr_pct_wins'])}/{_pct(wlt['avg_atr_pct_losses'])}",
        ]
    )

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "3) ENTRY QUALITY — OOS TEST (fixed hold; MH only)",
            "    (MACD / directional S/R hard filters intentionally excluded)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for gate, block in result["entry_quality_oos"].items():
        m = block["metrics"]
        lines.append("")
        lines.append(
            f"[entry:{gate}] removed_MH={block['removed_mh']}  {_line(m['all'])}"
        )
        lines.append(f"  HIGH: {_line(m['HIGH'])}  MEDIUM: {_line(m['MEDIUM'])}")
        for cls, bag in m["by_asset_class"].items():
            lines.append(f"    {cls}: {_line(bag)}")

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "4) STOP-LOSS — OOS TEST (trending MH entries; ORIGINAL scores)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for name, m in result["stop_loss_oos"].items():
        lines.append("")
        lines.extend(_block(f"stop:{name}", m))

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "5) TAKE-PROFIT — OOS TEST (trending MH entries; ORIGINAL scores)",
            "════════════════════════════════════════════════════════════",
        ]
    )
    for name, m in result["take_profit_oos"].items():
        lines.append("")
        lines.extend(_block(f"tp:{name}", m))

    # Recommendation logic
    base = result["baseline"]["test_trending_mh"]["all"]
    entry_best = None
    entry_best_avg = base.avg_return
    for gate, block in result["entry_quality_oos"].items():
        if gate == "none":
            continue
        bag = block["metrics"]["all"]
        if (
            bag.avg_return is not None
            and bag.signals >= MIN_SIGNALS_FOR_CONCLUSION
            and (entry_best_avg is None or bag.avg_return > entry_best_avg)
        ):
            # Prefer broad: need >=2 asset classes with n>=5 and non-worse avg
            classes = block["metrics"]["by_asset_class"]
            ok_classes = 0
            for cls, cb in classes.items():
                base_cls = result["baseline"]["test_trending_mh"]["by_asset_class"].get(cls)
                if cb.signals >= 5 and cb.avg_return is not None:
                    if base_cls is None or base_cls.avg_return is None or cb.avg_return >= base_cls.avg_return - 1e-9:
                        ok_classes += 1
            if bag.avg_return > (base.avg_return or -999) and ok_classes >= 1:
                entry_best = gate
                entry_best_avg = bag.avg_return

    def best_policy(oos: dict[str, Any], baseline_name: str = "fixed_hold") -> tuple[Optional[str], MetricBag]:
        base_bag = oos[baseline_name]["all"]
        best_name = None
        best_bag = base_bag
        for name, m in oos.items():
            if name == baseline_name:
                continue
            bag = m["all"]
            if bag.signals < MIN_SIGNALS_FOR_CONCLUSION or bag.avg_return is None:
                continue
            if base_bag.avg_return is None or bag.avg_return > base_bag.avg_return:
                # drawdown not worse by >5pp OR avg clearly better
                dd_ok = True
                if (
                    bag.max_drawdown is not None
                    and base_bag.max_drawdown is not None
                    and bag.max_drawdown > base_bag.max_drawdown + 0.05
                ):
                    dd_ok = bag.avg_return > (base_bag.avg_return or 0) + 0.001
                if dd_ok:
                    best_name = name
                    best_bag = bag
        return best_name, best_bag

    stop_best, stop_bag = best_policy(result["stop_loss_oos"])
    tp_best, tp_bag = best_policy(result["take_profit_oos"])

    lines.extend(
        [
            "",
            "════════════════════════════════════════════════════════════",
            "BEGINNER RECOMMENDATION (analysis only — wait for approval)",
            "════════════════════════════════════════════════════════════",
            f"  Baseline TEST trending MH: {_line(base)}",
            "",
            "  KEEP:",
            "    • Original scoring thresholds (HIGH≥60 / MED≥40 / LOW≥25).",
            "    • Crypto disabled.",
            "    • No MACD-strong or directional S/R hard filters (already rejected).",
            "    • Continue focusing on genuinely trending markets (sma_stack).",
            "",
            "  CHANGE CANDIDATES (not live yet):",
        ]
    )

    if entry_best:
        lines.append(
            f"    • Entry gate '{entry_best}' beat trending baseline on TEST avg "
            f"with n≥{MIN_SIGNALS_FOR_CONCLUSION} — consider for a follow-up confirm."
        )
    else:
        lines.append(
            "    • No entry-quality gate clearly beat the trending baseline on TEST "
            "with a reliable sample — do not tighten entries yet."
        )

    if stop_best:
        lines.append(
            f"    • Stop-loss candidate: {stop_best} "
            f"(TEST {_line(stop_bag)}) vs fixed_hold."
        )
    else:
        lines.append(
            "    • No ATR stop-loss clearly improved TEST avg vs fixed hold "
            "with a reliable sample — keep studying exits; do not enable SL yet."
        )

    if tp_best:
        lines.append(
            f"    • Take-profit candidate: {tp_best} "
            f"(TEST {_line(tp_bag)}) vs fixed_hold."
        )
    else:
        lines.append(
            "    • No ATR take-profit clearly improved TEST avg vs fixed hold "
            "with a reliable sample — do not enable TP yet."
        )

    lines.extend(
        [
            "",
            "  REJECT (for now):",
            "    • Enabling MACD or directional S/R as hard filters.",
            "    • Auto-merging any exit/entry rule into the live scanner.",
            "    • Changing scoring thresholds without a fresh OOS win.",
            "",
            "  PRODUCT DIRECTION (awaiting approval before live edits):",
            "    • Target universe: forex + stocks + commodities (no crypto).",
            "    • Target main scan timeframe: 4H trending setups.",
            "    • Live today remains unchanged until you explicitly approve.",
            "",
        ]
    )
    return "\n".join(lines)
