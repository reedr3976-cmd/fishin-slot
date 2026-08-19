#!/usr/bin/env python3
"""Run Scanner V12 data expansion + frozen V11_S_FVG_SWEEP replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V12_REPORT_JSON,
    SCANNER_V12_REPORT_TXT,
    V12_COMM_DEV,
    V12_COMM_FINAL_INST,
    V12_FX_DEV,
    V12_FX_FINAL_INST,
    V12_MACRO_KEYS,
    V12_STOCK_DEV,
    V12_STOCK_FINAL_INST,
    active_instruments,
)
from backtest.data_integrity import validate_panel
from backtest.macro_features import build_macro_context
from backtest.report_scanner_v12 import build_v12_payload, write_v12_reports
from providers.data_source_audit import build_data_source_audit
from providers.macro_calendar import load_macro_bundle
from providers.research_data_loader import load_extended_panel


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V12 extended data replication")
    p.add_argument("--demo", action="store_true", help="Not supported for V12 — requires extended fetch")
    args = p.parse_args()
    if args.demo:
        print("V12 requires extended Dukascopy data; --demo disabled.", flush=True)
        return 2

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    active = active_instruments()
    for key in ("NATGAS", "COPPER", "CORN", "DXY", "US10Y", "US3M", "TIP"):
        if key in active:
            print(f"FATAL: research_only symbol {key} leaked into active_instruments()", flush=True)
            return 2

    print("=" * 72, flush=True)
    print("PHASE 1 — DATA SOURCE AUDIT", flush=True)
    audit = build_data_source_audit()
    print(audit["integration_policy"], flush=True)

    keys = sorted(
        set(V12_STOCK_DEV)
        | set(V12_STOCK_FINAL_INST)
        | set(V12_COMM_DEV)
        | set(V12_COMM_FINAL_INST)
        | set(V12_FX_DEV)
        | set(V12_FX_FINAL_INST)
        | set(V12_MACRO_KEYS)
    )

    print("=" * 72, flush=True)
    print("PHASE 1/2 — LOAD EXTENDED PANEL (Dukascopy primary)", flush=True)
    series_4h, daily_map, weekly_map, errors, provenance = load_extended_panel(keys)
    if errors:
        for e in errors:
            print(f"  LOAD ERR: {e}", flush=True)
    if not series_4h:
        print("FATAL: no 4H series loaded", flush=True)
        return 1

    print("PHASE 2 — DATA INTEGRITY", flush=True)
    integrity = validate_panel(series_4h)

    print("Loading macro context (Yahoo daily proxies)...", flush=True)
    bundle = load_macro_bundle()
    ctx = build_macro_context(bundle, series_4h, daily_map)

    print("=" * 72, flush=True)
    print("PHASE 3–8 — FROZEN REPLICATION EXPERIMENTS", flush=True)
    payload = build_v12_payload(series_4h, daily_map, weekly_map, audit, integrity, provenance, ctx)
    payload["load_errors"] = errors
    text = write_v12_reports(payload, SCANNER_V12_REPORT_TXT, SCANNER_V12_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V12_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
