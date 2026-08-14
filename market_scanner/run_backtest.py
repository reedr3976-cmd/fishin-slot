#!/usr/bin/env python3
"""Run historical backtest of scanner confidence ratings (analysis only).

Examples:
  python3 run_backtest.py --live
  python3 run_backtest.py --demo
  python3 run_backtest.py --live --tf 1d,1wk
  python3 run_backtest.py --live --symbols EURUSD,BTCUSD,XAUUSD --tf 1d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    BACKTEST_DEFAULT_TIMEFRAMES,
    BACKTEST_REPORT_JSON,
    BACKTEST_REPORT_TXT,
    INSTRUMENTS,
    OUTPUT_DIR,
    TIMEFRAMES,
)
from backtest.engine import run_backtest_with_metrics
from backtest.report import build_backtest_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest scanner HIGH/MEDIUM/LOW ratings (no live trading)."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic/offline history")
    mode.add_argument(
        "--live",
        action="store_true",
        help="Public Yahoo historical OHLC (no API key)",
    )
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument(
        "--tf",
        type=str,
        default=",".join(BACKTEST_DEFAULT_TIMEFRAMES),
        help=f"Timeframes (default {','.join(BACKTEST_DEFAULT_TIMEFRAMES)})",
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
            return 2

    timeframes = [t.strip() for t in args.tf.split(",") if t.strip()]
    bad_tf = [t for t in timeframes if t not in TIMEFRAMES]
    if bad_tf:
        print(f"Unknown timeframes: {', '.join(bad_tf)}")
        return 2

    print("Running historical backtest (analysis only, no brokerage)...")
    print(f"Mode: {'demo' if demo else 'public historical Yahoo data'}")
    print(f"Symbols: {symbols or 'ALL'}")
    print(f"Timeframes: {timeframes}")

    run, metrics = run_backtest_with_metrics(symbols, timeframes, demo=demo)
    report = build_backtest_report(run, metrics)
    print(report)

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    Path(BACKTEST_REPORT_TXT).write_text(report, encoding="utf-8")

    payload = {
        "mode": run.mode,
        "disclaimer": (
            "Educational backtest of scanner confidence ratings. "
            "No orders placed. Not financial advice."
        ),
        "instruments": run.instruments,
        "timeframes": run.timeframes,
        "bars_scanned": run.bars_scanned,
        "trade_count": len(run.trades),
        "errors": run.errors,
        "metrics": {
            "overall": metrics["overall"].to_dict(),
            "by_confidence": {k: v.to_dict() for k, v in metrics["by_confidence"].items()},
            "by_asset_class": {k: v.to_dict() for k, v in metrics["by_asset_class"].items()},
            "by_timeframe": {k: v.to_dict() for k, v in metrics["by_timeframe"].items()},
            "by_asset_class_confidence": {
                cls: {c: bag.to_dict() for c, bag in confs.items()}
                for cls, confs in metrics["by_asset_class_confidence"].items()
            },
            "by_timeframe_confidence": {
                tf: {c: bag.to_dict() for c, bag in confs.items()}
                for tf, confs in metrics["by_timeframe_confidence"].items()
            },
        },
        "trades": [t.to_dict() for t in run.trades],
    }
    # JSON cannot serialize inf — convert
    text = json.dumps(payload, indent=2, default=str)
    text = text.replace("Infinity", "null")
    Path(BACKTEST_REPORT_JSON).write_text(text, encoding="utf-8")

    print(f"Saved: {BACKTEST_REPORT_TXT}")
    print(f"Saved: {BACKTEST_REPORT_JSON}")
    return 0 if not (run.errors and not run.trades) else 1


if __name__ == "__main__":
    raise SystemExit(main())
