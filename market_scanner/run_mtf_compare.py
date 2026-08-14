#!/usr/bin/env python3
"""Compare ORIGINAL vs multi-timeframe-filtered scanner (analysis only).

Does not change live defaults. Scoring thresholds unchanged.
Crypto remains disabled by default.

  python3 run_mtf_compare.py --live
  python3 run_mtf_compare.py --demo
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
    MTF_REPORT_JSON,
    MTF_REPORT_TXT,
    OUTPUT_DIR,
    TIMEFRAMES,
    VALIDATION_TRAIN_FRACTION,
    active_instruments,
)
from backtest.mtf_compare import run_mtf_comparison
from backtest.report_mtf import build_mtf_comparison_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ORIGINAL vs MTF filter OOS comparison")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=VALIDATION_TRAIN_FRACTION)
    args = p.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in symbols if s not in INSTRUMENTS]
        if bad:
            print(f"Unknown symbols: {bad}")
            return 2

    print("Comparing ORIGINAL vs MTF-filtered scanner (analysis only)...")
    print(f"Mode: {'demo' if args.demo else 'public historical'}")
    print(
        f"Universe: {symbols or list(active_instruments().keys())} "
        "(crypto disabled by default)"
    )
    print("Scoring thresholds: UNCHANGED (original)")
    print("Live scanner: NOT modified / MTF remains OFF by default")

    result = run_mtf_comparison(
        demo=bool(args.demo),
        instruments=symbols,
        timeframes=["1d", "1wk"],
        train_fraction=args.train_frac,
    )
    report = build_mtf_comparison_report(result)
    print(report)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MTF_REPORT_TXT).write_text(report, encoding="utf-8")

    def bag(b):
        return b.to_dict()

    payload = {
        "mode": result["mode"],
        "train_fraction": result["train_fraction"],
        "instruments": result["instruments"],
        "mtf_test_stats": result["mtf_test_stats"],
        "mtf_train_stats": result["mtf_train_stats"],
        "metrics": {
            k: {
                "overall": bag(v["overall"]),
                "by_confidence": {c: bag(x) for c, x in v["by_confidence"].items()},
                "by_asset_class": {c: bag(x) for c, x in v["by_asset_class"].items()},
                "by_timeframe": {c: bag(x) for c, x in v["by_timeframe"].items()},
            }
            for k, v in result["metrics"].items()
        },
        "high_medium_test": {
            "original": bag(result["high_medium_test"]["original"]),
            "mtf": bag(result["high_medium_test"]["mtf"]),
        },
        "errors": result["errors"],
        "disclaimer": (
            "Educational comparison only. Live scanner unchanged. "
            "Do not merge without explicit approval."
        ),
    }
    text = json.dumps(payload, indent=2, default=str).replace("Infinity", "null")
    Path(MTF_REPORT_JSON).write_text(text, encoding="utf-8")
    print(f"Saved: {MTF_REPORT_TXT}")
    print(f"Saved: {MTF_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
