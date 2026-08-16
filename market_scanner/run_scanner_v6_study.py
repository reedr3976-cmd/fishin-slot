#!/usr/bin/env python3
"""Run Scanner V6 clean strategy-family reset (research only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V6_REPORT_JSON,
    SCANNER_V6_REPORT_TXT,
    V6_COMMODITY_DISCOVERY,
    V6_COMMODITY_HELDOUT,
    V6_FX_DISCOVERY,
    V6_FX_HELDOUT,
    V6_STOCK_DISCOVERY,
    V6_STOCK_HELDOUT,
)
from backtest.engine import load_series_map
from backtest.report_scanner_v6 import build_v6_payload, write_v6_reports


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V6 strategy-family reset")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    keys = sorted(
        set(V6_STOCK_DISCOVERY)
        | set(V6_STOCK_HELDOUT)
        | set(V6_COMMODITY_DISCOVERY)
        | set(V6_COMMODITY_HELDOUT)
        | set(V6_FX_DISCOVERY)
        | set(V6_FX_HELDOUT)
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

    payload = build_v6_payload(series_4h, daily_map)
    if errors:
        payload["load_errors"] = errors
    text = write_v6_reports(payload, SCANNER_V6_REPORT_TXT, SCANNER_V6_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V6_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
