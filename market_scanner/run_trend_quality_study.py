#!/usr/bin/env python3
"""Trend-quality filter study (analysis only). Live scanner unchanged.

  python3 run_trend_quality_study.py --live
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
    INSTRUMENTS,
    OUTPUT_DIR,
    TREND_QUALITY_JSON,
    TREND_QUALITY_TXT,
    VALIDATION_TRAIN_FRACTION,
    study_instruments,
)
from backtest.trend_quality import run_trend_quality_study
from backtest.report_trend_quality import build_trend_quality_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trend-quality OOS study")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=VALIDATION_TRAIN_FRACTION)
    return p


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj == float("inf"):
        return "inf"
    return obj


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    demo = bool(args.demo)
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in symbols if s not in INSTRUMENTS]
        if bad:
            print(f"Unknown: {bad}")
            return 2

    print("Trend-quality study (analysis only; live unchanged)...")
    univ = symbols or list(study_instruments().keys())
    print(f"Universe ({len(univ)}): {', '.join(univ)}")

    result = run_trend_quality_study(
        demo=demo, instruments=symbols, train_fraction=args.train_frac
    )
    report = build_trend_quality_report(result)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TREND_QUALITY_TXT).write_text(report, encoding="utf-8")
    Path(TREND_QUALITY_JSON).write_text(
        json.dumps(_to_jsonable(result), indent=2), encoding="utf-8"
    )
    print(report)
    print(f"Saved: {TREND_QUALITY_TXT}")
    print(f"Saved: {TREND_QUALITY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
