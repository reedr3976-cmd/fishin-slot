#!/usr/bin/env python3
"""Run Scanner V11 near-miss refinement research (research only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V11_REPORT_JSON,
    SCANNER_V11_REPORT_TXT,
    V11_COMM_DEV,
    V11_COMM_FINAL_INST,
    V11_FX_DEV,
    V11_FX_FINAL_INST,
    V11_MACRO_KEYS,
    V11_STOCK_DEV,
    V11_STOCK_FINAL_INST,
    active_instruments,
)
from backtest.data_quality_audit import full_data_audit
from backtest.engine import load_series_map
from backtest.macro_features import build_macro_context
from backtest.report_scanner_v11 import build_v11_payload, write_v11_reports
from providers.macro_calendar import load_macro_bundle


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V11 near-miss research")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    active = active_instruments()
    for key in ("NATGAS", "COPPER", "CORN", "DXY", "US10Y", "US3M", "TIP"):
        if key in active:
            print(f"FATAL: research_only symbol {key} leaked into active_instruments()", flush=True)
            return 2

    keys = sorted(
        set(V11_STOCK_DEV)
        | set(V11_STOCK_FINAL_INST)
        | set(V11_COMM_DEV)
        | set(V11_COMM_FINAL_INST)
        | set(V11_FX_DEV)
        | set(V11_FX_FINAL_INST)
        | set(V11_MACRO_KEYS)
    )
    print(f"Loading 4H + 1D + 1WK for {len(keys)} instruments...", flush=True)
    series_4h, err4, _ = load_series_map(keys, ["4h"], demo=args.demo)
    series_1d, err1, _ = load_series_map(keys, ["1d"], demo=args.demo)
    series_1wk, errw, _ = load_series_map(keys, ["1wk"], demo=args.demo)
    series_4h = {k: v for k, v in series_4h.items() if k[1] == "4h"}
    daily_map = {k: v for (k, tf), v in series_1d.items() if tf == "1d"}
    weekly_map = {k: v for (k, tf), v in series_1wk.items() if tf == "1wk"}
    errors = err4 + err1 + errw
    if not series_4h:
        return 1

    print("Running data quality audit...", flush=True)
    data_audit = full_data_audit(series_4h, daily_map, weekly_map)
    print(f"  median 4H span days: {data_audit['loaded_coverage']['median_4h_span_days']}", flush=True)

    print("Loading macro bundle (V9 layer retained)...", flush=True)
    bundle = load_macro_bundle()
    ctx = build_macro_context(bundle, series_4h, daily_map)

    payload = build_v11_payload(series_4h, daily_map, weekly_map, bundle, ctx, data_audit)
    if errors:
        payload["load_errors"] = errors
    text = write_v11_reports(payload, SCANNER_V11_REPORT_TXT, SCANNER_V11_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V11_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
