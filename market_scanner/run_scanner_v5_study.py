#!/usr/bin/env python3
"""Run Scanner V5 independent robustness validation (no live/paper enablement)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V5_REPORT_JSON,
    SCANNER_V5_REPORT_TXT,
    V5_COMMODITIES,
    V5_HELD_OUT_STOCKS,
    V5_V4_STOCKS,
)
from backtest.engine import load_series_map
from backtest.report_scanner_v5 import build_v5_payload, write_v5_reports


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V5 robustness validation")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    keys = list(V5_V4_STOCKS) + list(V5_HELD_OUT_STOCKS) + list(V5_COMMODITIES)
    print(f"Loading 4H series for {len(keys)} instruments...", flush=True)
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=args.demo)
    series_map = {k: v for k, v in series_map.items() if k[1] == "4h"}
    print(f"Loaded {len(series_map)} series, {bars} bars; {len(errors)} errors", flush=True)
    for e in errors:
        print(f"  ERR: {e}", flush=True)
    if not series_map:
        return 1

    payload = build_v5_payload(series_map)
    if errors:
        payload["load_errors"] = errors
    text = write_v5_reports(payload, SCANNER_V5_REPORT_TXT, SCANNER_V5_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V5_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
