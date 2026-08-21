#!/usr/bin/env python3
"""Run Scanner V13 confirmation research for frozen E3 (no optimisation)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_DIR,
    SCANNER_V13_REPORT_JSON,
    SCANNER_V13_REPORT_TXT,
    V13_STOCK_ALL,
    V13_STOCK_DEV,
    V13_STOCK_FINAL_INST,
    active_instruments,
)
from backtest.data_integrity import validate_panel
from backtest.frozen_e3_spec import FROZEN_E3_VERSION, frozen_e3_hash
from backtest.macro_features import build_macro_context
from backtest.report_scanner_v13 import build_v13_payload, write_v13_reports
from providers.macro_calendar import load_macro_bundle
from providers.research_data_loader import load_extended_panel


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT.parent), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Scanner V13 frozen E3 confirmation")
    p.add_argument("--demo", action="store_true", help="Disabled — V13 requires extended data")
    args = p.parse_args()
    if args.demo:
        print("V13 requires extended Dukascopy data; --demo disabled.", flush=True)
        return 2

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    active = active_instruments()
    for key in ("NATGAS", "COPPER", "CORN", "DXY", "US10Y", "US3M", "TIP"):
        if key in active:
            print(f"FATAL: research_only symbol {key} leaked into active_instruments()", flush=True)
            return 2

    print("=" * 72, flush=True)
    print("V13 CONFIRMATION — FROZEN E3 ONLY (NO OPTIMISATION)", flush=True)
    print(f"Spec version: {FROZEN_E3_VERSION}", flush=True)
    print(f"Spec hash: {frozen_e3_hash()}", flush=True)
    print(f"Git commit: {_git_commit()}", flush=True)

    keys = sorted(set(V13_STOCK_ALL) | {"DXY", "US10Y", "US3M", "TIP"})
    print("=" * 72, flush=True)
    print("LOAD EXTENDED PANEL (Dukascopy primary, cached)", flush=True)
    series_4h, daily_map, weekly_map, errors, provenance = load_extended_panel(keys)
    if errors:
        for e in errors:
            print(f"  LOAD ERR: {e}", flush=True)
    # Keep only stock 4h series for E3
    stock_keys = set(V13_STOCK_DEV) | set(V13_STOCK_FINAL_INST)
    series_4h = {k: v for k, v in series_4h.items() if k[0] in stock_keys}
    if not series_4h:
        print("FATAL: no stock 4H series loaded", flush=True)
        return 1

    print("DATA INTEGRITY", flush=True)
    integrity = validate_panel(series_4h)
    print(f"  all_ok={integrity.get('all_ok')} issues={integrity.get('instruments_with_issues')}", flush=True)

    print("Macro context...", flush=True)
    bundle = load_macro_bundle()
    ctx = build_macro_context(bundle, series_4h, daily_map)

    print("=" * 72, flush=True)
    print("PHASES 1–12 — CONFIRMATION BATTERY", flush=True)
    payload = build_v13_payload(
        series_4h, daily_map, weekly_map, integrity, provenance, ctx, git_commit=_git_commit()
    )
    payload["load_errors"] = errors
    text = write_v13_reports(payload, SCANNER_V13_REPORT_TXT, SCANNER_V13_REPORT_JSON)
    print(text)
    print(f"Wrote {SCANNER_V13_REPORT_TXT}", flush=True)
    print(f"VERDICT: {payload.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
