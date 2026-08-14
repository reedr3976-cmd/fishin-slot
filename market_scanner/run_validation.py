#!/usr/bin/env python3
"""Out-of-sample validation for scanner confidence ratings (analysis only).

Examples:
  python3 run_validation.py --live
  python3 run_validation.py --demo
  python3 run_validation.py --live --tf 1d,1wk
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
    INSTRUMENTS,
    OUTPUT_DIR,
    TIMEFRAMES,
    VALIDATION_REPORT_JSON,
    VALIDATION_REPORT_TXT,
    VALIDATION_TRAIN_FRACTION,
)
from backtest.report_validation import build_validation_report
from backtest.validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OOS validation of confidence scoring")
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

    print("Running chronological train/test validation (analysis only)...")
    print(f"Mode: {'demo' if demo else 'public historical'}")
    print(f"Train fraction: {args.train_frac:.0%}")

    result = run_validation(
        demo=demo,
        instruments=symbols,
        timeframes=timeframes,
        train_fraction=args.train_frac,
    )
    report = build_validation_report(result)
    print(report)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(VALIDATION_REPORT_TXT).write_text(report, encoding="utf-8")

    def bag_dict(metrics_block: dict) -> dict:
        out = {
            "overall": metrics_block["overall"].to_dict(),
            "by_confidence": {k: v.to_dict() for k, v in metrics_block["by_confidence"].items()},
            "by_asset_class": {k: v.to_dict() for k, v in metrics_block["by_asset_class"].items()},
            "by_timeframe": {k: v.to_dict() for k, v in metrics_block["by_timeframe"].items()},
        }
        return out

    payload = {
        "mode": result["mode"],
        "train_fraction": result["train_fraction"],
        "recommendation": result["recommendation"],
        "rationale": result["rationale"],
        "original_rules": result["original_rules"].to_dict(),
        "revised_rules": result["revised_rules"].to_dict(),
        "feature_edges_train": [e.to_dict() for e in result["feature_edges_train"]],
        "metrics": {k: bag_dict(v) for k, v in result["metrics"].items()},
        "errors": result["errors"],
        "disclaimer": (
            "Educational validation only. No orders placed. Not financial advice. "
            "Do not merge rule changes without explicit approval."
        ),
    }
    text = json.dumps(payload, indent=2, default=str).replace("Infinity", "null")
    Path(VALIDATION_REPORT_JSON).write_text(text, encoding="utf-8")
    print(f"Saved: {VALIDATION_REPORT_TXT}")
    print(f"Saved: {VALIDATION_REPORT_JSON}")
    print(f"Recommendation: {result['recommendation']['decision']}")
    print("Live scanner still uses ORIGINAL rules. PR should NOT be merged without approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
