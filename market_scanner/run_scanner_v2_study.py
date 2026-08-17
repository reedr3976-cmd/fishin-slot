#!/usr/bin/env python3
"""Run Scanner V2 staged research study (analysis only — no live changes).

Usage:
  python run_scanner_v2_study.py
  python run_scanner_v2_study.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V2_REPORT_JSON,
    SCANNER_V2_REPORT_TXT,
    study_instruments,
)
from backtest.engine import load_series_map
from backtest.report_scanner_v2 import build_study_payload, write_reports


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V2 research study (4H)")
    p.add_argument("--demo", action="store_true", help="Use synthetic data")
    p.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        help="Optional instrument keys (default = study universe)",
    )
    args = p.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    keys = list(args.instruments) if args.instruments else list(study_instruments().keys())
    print(f"Loading 4H series for {len(keys)} instruments (demo={args.demo})...", flush=True)
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=args.demo)
    # Keep only 4h
    series_map = {k: v for k, v in series_map.items() if k[1] == "4h"}
    print(f"Loaded {len(series_map)} series, {bars} bars; {len(errors)} errors", flush=True)
    for e in errors:
        print(f"  ERR: {e}", flush=True)

    if not series_map:
        print("No series loaded — aborting.", flush=True)
        return 1

    print("Running staged walk-forward study...", flush=True)
    payload = build_study_payload(series_map)
    if errors:
        payload["load_errors"] = errors
    text = write_reports(payload, SCANNER_V2_REPORT_TXT, SCANNER_V2_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V2_REPORT_TXT}", flush=True)
    print(f"Wrote {SCANNER_V2_REPORT_JSON}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
