#!/usr/bin/env python3
"""Market Scanner CLI — alerts & analysis only (no trading).

Examples:
  python run_scanner.py --demo
  python run_scanner.py --live
  python run_scanner.py --live --assets forex,crypto,commodity --tf 1d
  python run_scanner.py --live --symbols EURUSD,BTCUSD,XAUUSD --tf 1h,1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local imports resolve when run as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_TIMEFRAMES, INSTRUMENTS, TIMEFRAMES
from scanner import scan_markets, write_outputs


BANNER = """
╔══════════════════════════════════════════════════════════╗
║   MARKET SCANNER  ·  Alerts & Analysis Only              ║
║   Forex · Crypto · Commodities                           ║
║   NO brokerage connection · NO order placement           ║
╚══════════════════════════════════════════════════════════╝
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan markets for technical setups (alerts only, no trading)."
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
        default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Comma list of timeframes. Options: {','.join(TIMEFRAMES)}",
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
        help="Filter alerts below this strength (default: low = show all).",
    )
    return p


STRENGTH_RANK = {"low": 1, "medium": 2, "high": 3}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Default: public Yahoo data (no key). Use --demo for offline/synthetic.
    demo = bool(args.demo)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in symbols if s not in INSTRUMENTS]
        if bad:
            print(f"Unknown symbols: {', '.join(bad)}")
            print(f"Known: {', '.join(INSTRUMENTS)}")
            return 2

    timeframes = [t.strip() for t in args.tf.split(",") if t.strip()]
    bad_tf = [t for t in timeframes if t not in TIMEFRAMES]
    if bad_tf:
        print(f"Unknown timeframes: {', '.join(bad_tf)}")
        print(f"Known: {', '.join(TIMEFRAMES)}")
        return 2

    assets = None
    if args.assets:
        assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]

    print(BANNER)
    mode_label = "DEMO (cached/synthetic historical)" if demo else "LIVE PUBLIC DATA (Yahoo Finance, no API key)"
    print(f"Mode:        {mode_label}")
    print(f"Instruments: {', '.join(symbols) if symbols else 'ALL (' + str(len(INSTRUMENTS)) + ')'}")
    print(f"Timeframes:  {', '.join(timeframes)}")
    print(f"Assets:      {', '.join(assets) if assets else 'forex, crypto, commodity'}")
    print("-" * 60)

    alerts, snapshots, errors = scan_markets(
        symbols, timeframes, demo=demo, asset_classes=assets
    )

    min_rank = STRENGTH_RANK[args.min_strength]
    alerts = [a for a in alerts if STRENGTH_RANK.get(a.strength, 0) >= min_rank]

    out_path = write_outputs(alerts, snapshots, errors)

    # Snapshots table
    print("\nPRICE SNAPSHOTS")
    print(f"{'Instrument':<10} {'TF':<4} {'Class':<10} {'Last':>14} {'RSI':>7} {'Bars':>6}")
    print("-" * 60)
    for s in snapshots:
        rsi = s.get("rsi")
        rsi_s = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "-"
        print(
            f"{s['instrument']:<10} {s['timeframe']:<4} {s['asset_class']:<10} "
            f"{s['last_close']:>14.6g} {rsi_s:>7} {s['bars']:>6}"
        )

    # Alerts
    print("\nSETUP ALERTS (analysis only — do not auto-trade)")
    if not alerts:
        print("  (no setups matched current thresholds)")
    else:
        # Prefer actionable (medium/high) first
        ordered = sorted(
            alerts,
            key=lambda a: (-STRENGTH_RANK.get(a.strength, 0), a.instrument, a.timeframe),
        )
        for a in ordered:
            flag = {"bullish": "▲", "bearish": "▼", "neutral": "●"}.get(a.side, "●")
            print(
                f"  {flag} [{a.strength.upper():<6}] {a.instrument:<8} {a.timeframe:<4} "
                f"{a.setup:<22} @ {a.price}"
            )
            print(f"      {a.message}")

    if errors:
        print("\nFETCH NOTES")
        for e in errors:
            print(f"  ! {e}")

    print("\n" + "=" * 60)
    print(f"SCAN COMPLETE — {len(snapshots)} snapshots, {len(alerts)} alerts")
    print(f"Saved: {out_path}")
    print("Reminder: alerts only. No trades placed. Not financial advice.")
    print("=" * 60)
    return 0 if snapshots else 1


if __name__ == "__main__":
    raise SystemExit(main())
