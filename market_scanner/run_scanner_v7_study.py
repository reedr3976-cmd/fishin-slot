#!/usr/bin/env python3
"""Run Scanner V7 robustness research (research only; live scanner untouched)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V7_REPORT_JSON,
    SCANNER_V7_REPORT_TXT,
    V7_COMMODITY_DISCOVERY,
    V7_COMMODITY_HELDOUT,
    V7_FX_DISCOVERY,
    V7_FX_HELDOUT,
    V7_STOCK_DISCOVERY,
    V7_STOCK_HELDOUT,
    active_instruments,
)
from backtest.engine import load_series_map
from backtest.report_scanner_v7 import build_v7_payload, write_v7_reports


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V7 robustness research")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Safety: research-only commodities must not appear in live active universe
    active = active_instruments()
    for key in ("NATGAS", "COPPER", "CORN"):
        if key in active:
            print(f"FATAL: research_only symbol {key} leaked into active_instruments()", flush=True)
            return 2

    keys = sorted(
        set(V7_STOCK_DISCOVERY)
        | set(V7_STOCK_HELDOUT)
        | set(V7_COMMODITY_DISCOVERY)
        | set(V7_COMMODITY_HELDOUT)
        | set(V7_FX_DISCOVERY)
        | set(V7_FX_HELDOUT)
    )
    print(f"Loading 4H + 1d for {len(keys)} instruments...", flush=True)
    series_4h, err4, bars4 = load_series_map(keys, ["4h"], demo=args.demo)
    series_1d, err1, bars1 = load_series_map(keys, ["1d"], demo=args.demo)
    series_4h = {k: v for k, v in series_4h.items() if k[1] == "4h"}
    daily_map = {k: v for (k, tf), v in series_1d.items() if tf == "1d"}
    errors = err4 + err1
    print(
        f"Loaded 4H={len(series_4h)} ({bars4} bars), 1d={len(daily_map)} ({bars1} bars); "
        f"{len(errors)} errors",
        flush=True,
    )
    for e in errors:
        print(f"  ERR: {e}", flush=True)
    if not series_4h:
        return 1

    payload = build_v7_payload(series_4h, daily_map)
    if errors:
        payload["load_errors"] = errors
    text = write_v7_reports(payload, SCANNER_V7_REPORT_TXT, SCANNER_V7_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V7_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
