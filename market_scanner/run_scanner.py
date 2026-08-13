#!/usr/bin/env python3
"""Daily Market Scanner CLI — ranked alerts & analysis only (no trading).

Examples:
  python3 run_scanner.py --live
  python3 run_scanner.py --demo
  python3 run_scanner.py --live --symbols EURUSD,BTCUSD,XAUUSD --tf 1d
  python3 run_scanner.py --live --tf 1h,1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DAILY_TIMEFRAMES, INSTRUMENTS, TIMEFRAMES
from scanner import scan_opportunities, write_outputs
from scanner.report import build_daily_summary
from scanner.setups import SetupAlert


BANNER = """
╔══════════════════════════════════════════════════════════╗
║   DAILY MARKET SCANNER  ·  Alerts & Analysis Only        ║
║   Forex · Crypto · Commodities                           ║
║   NO brokerage connection · NO order placement           ║
╚══════════════════════════════════════════════════════════╝
"""

STRENGTH_RANK = {"low": 1, "medium": 2, "high": 3}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ranked daily market scanner (alerts only, no trading)."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Use cached/synthetic historical data (offline-friendly).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Fetch public Yahoo Finance chart data (no API key).",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=f"Comma list of instruments. Default: all. Options: {','.join(INSTRUMENTS)}",
    )
    p.add_argument(
        "--tf",
        type=str,
        default=None,
        help=(
            f"Comma list of timeframes. Default (daily report): {','.join(DAILY_TIMEFRAMES)}. "
            f"Options: {','.join(TIMEFRAMES)}"
        ),
    )
    p.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma list of asset classes: forex,crypto,commodity",
    )
    p.add_argument(
        "--min-strength",
        choices=["low", "medium", "high"],
        default="low",
        help="Only include actionable opportunities at/above this confidence.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    demo = bool(args.demo)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in symbols if s not in INSTRUMENTS]
        if bad:
            print(f"Unknown symbols: {', '.join(bad)}")
            print(f"Known: {', '.join(INSTRUMENTS)}")
            return 2

    timeframes = (
        [t.strip() for t in args.tf.split(",") if t.strip()]
        if args.tf
        else list(DAILY_TIMEFRAMES)
    )
    bad_tf = [t for t in timeframes if t not in TIMEFRAMES]
    if bad_tf:
        print(f"Unknown timeframes: {', '.join(bad_tf)}")
        print(f"Known: {', '.join(TIMEFRAMES)}")
        return 2

    assets = None
    if args.assets:
        assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]

    print(BANNER)
    mode_label = (
        "DEMO (cached/synthetic historical)"
        if demo
        else "LIVE PUBLIC DATA (Yahoo Finance, no API key)"
    )
    print(f"Mode:        {mode_label}")
    print(
        f"Instruments: {', '.join(symbols) if symbols else 'ALL (' + str(len(INSTRUMENTS)) + ')'}"
    )
    print(f"Timeframes:  {', '.join(timeframes)}")
    print(f"Assets:      {', '.join(assets) if assets else 'forex, crypto, commodity'}")
    print("-" * 60)

    opportunities, snapshots, errors = scan_opportunities(
        symbols, timeframes, demo=demo, asset_classes=assets
    )

    min_rank = STRENGTH_RANK[args.min_strength]
    alerts: list[SetupAlert] = []
    for opp in opportunities:
        if opp.confidence == "NO STRONG SETUP":
            continue
        strength = opp.confidence.lower()
        if STRENGTH_RANK.get(strength, 0) < min_rank:
            continue
        alerts.append(
            SetupAlert(
                instrument=opp.instrument,
                name=opp.name,
                asset_class=opp.asset_class,
                timeframe=opp.timeframe,
                setup=f"score_{opp.score}",
                side=opp.direction,
                strength=strength,
                price=opp.price,
                message=opp.reason,
                metrics={
                    "rsi": opp.rsi,
                    "sma_fast": opp.sma20,
                    "sma_slow": opp.sma50,
                    "atr": opp.atr,
                    "score": opp.score,
                    "support": opp.support,
                    "resistance": opp.resistance,
                },
                scanned_at=opp.scanned_at,
            )
        )

    out_path = write_outputs(
        alerts, snapshots, errors, opportunities, mode_label=mode_label
    )

    summary = build_daily_summary(opportunities, mode_label=mode_label, errors=errors)
    print(summary)
    print(f"Saved JSON: {out_path}")
    print("Saved beginner text: output/daily_summary.txt")
    print("Saved CSV: output/latest_alerts.csv")
    print("Reminder: alerts only. No trades placed. Not financial advice.")
    return 0 if snapshots else 1


if __name__ == "__main__":
    raise SystemExit(main())
