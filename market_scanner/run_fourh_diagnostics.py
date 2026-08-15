#!/usr/bin/env python3
"""4H trending diagnostics for the ORIGINAL scanner (analysis only).

Live scanner defaults are not modified.

Examples:
  python3 run_fourh_diagnostics.py --live
  python3 run_fourh_diagnostics.py --demo
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
    FOURH_DIAG_REPORT_JSON,
    FOURH_DIAG_REPORT_TXT,
    INSTRUMENTS,
    OUTPUT_DIR,
    VALIDATION_TRAIN_FRACTION,
    study_instruments,
)
from backtest.fourh_diagnostics import run_fourh_diagnostics
from backtest.metrics import MetricBag
from backtest.report_fourh_diagnostics import build_fourh_diagnostics_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="4H trending diagnostics (analysis only)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument(
        "--train-frac",
        type=float,
        default=VALIDATION_TRAIN_FRACTION,
        help="Chronological train fraction (default 0.70)",
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
        crypto = [s for s in symbols if INSTRUMENTS[s]["asset_class"] == "crypto"]
        if crypto:
            print(f"Crypto not allowed in this study: {', '.join(crypto)}")
            return 2

    print("4H trending diagnostics (analysis only; live scanner unchanged)...")
    print(f"Mode: {'demo' if demo else 'public historical'}")
    print(f"Train fraction: {args.train_frac:.0%}")
    univ = symbols or list(study_instruments().keys())
    print(f"Universe ({len(univ)}): {', '.join(univ)}")
    print("Crypto excluded · original scoring · no live enable")

    result = run_fourh_diagnostics(
        demo=demo, instruments=symbols, train_fraction=args.train_frac
    )
    report = build_fourh_diagnostics_report(result)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FOURH_DIAG_REPORT_TXT).write_text(report, encoding="utf-8")
    Path(FOURH_DIAG_REPORT_JSON).write_text(
        json.dumps(_to_jsonable(result), indent=2), encoding="utf-8"
    )
    print(report)
    print(f"Saved: {FOURH_DIAG_REPORT_TXT}")
    print(f"Saved: {FOURH_DIAG_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
