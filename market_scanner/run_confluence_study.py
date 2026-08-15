#!/usr/bin/env python3
"""Confluence A/B/C/D study (analysis only).

Does not change live scanner defaults or scoring thresholds.

Examples:
  python3 run_confluence_study.py --live
  python3 run_confluence_study.py --demo
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
    CONFLUENCE_REPORT_JSON,
    CONFLUENCE_REPORT_TXT,
    INSTRUMENTS,
    OUTPUT_DIR,
    TIMEFRAMES,
    VALIDATION_TRAIN_FRACTION,
    active_instruments,
)
from backtest.confluence import run_confluence_study
from backtest.metrics import MetricBag
from backtest.report_confluence import build_confluence_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Confluence A/B/C/D OOS study")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--tf", type=str, default=",".join(BACKTEST_DEFAULT_TIMEFRAMES))
    p.add_argument(
        "--train-frac",
        type=float,
        default=VALIDATION_TRAIN_FRACTION,
        help="Chronological train fraction per series (default 0.70)",
    )
    return p


def _to_jsonable(obj):
    if isinstance(obj, MetricBag):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


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

    print("Confluence study (analysis only; live scanner unchanged)...")
    print(f"Mode: {'demo' if demo else 'public historical'}")
    print(f"Train fraction: {args.train_frac:.0%}")
    print(
        f"Universe: {symbols or 'ACTIVE forex+commodities (' + str(len(active_instruments())) + ')'}"
    )
    print("Crypto off · original scoring · no live confluence filter")

    result = run_confluence_study(
        demo=demo,
        instruments=symbols,
        timeframes=timeframes,
        train_fraction=args.train_frac,
    )
    report = build_confluence_report(result)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CONFLUENCE_REPORT_TXT).write_text(report, encoding="utf-8")
    Path(CONFLUENCE_REPORT_JSON).write_text(
        json.dumps(_to_jsonable(result), indent=2), encoding="utf-8"
    )
    print(report)
    print(f"Saved: {CONFLUENCE_REPORT_TXT}")
    print(f"Saved: {CONFLUENCE_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
