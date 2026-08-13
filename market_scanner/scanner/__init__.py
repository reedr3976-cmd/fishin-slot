"""Multi-instrument / multi-timeframe market scanner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional

import requests

from config import DEFAULT_TIMEFRAMES, INSTRUMENTS, OUTPUT_DIR
from models import CandleSeries
from providers.yahoo import DataFetchError, fetch_instrument
from scanner.setups import SetupAlert, analyze_series


def scan_markets(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
    asset_classes: Optional[Iterable[str]] = None,
) -> tuple[list[SetupAlert], list[dict], list[str]]:
    """Scan instruments. Returns (alerts, snapshots, errors)."""
    keys = list(instruments) if instruments else list(INSTRUMENTS.keys())
    tfs = list(timeframes) if timeframes else list(DEFAULT_TIMEFRAMES)
    classes = set(asset_classes) if asset_classes else None

    alerts: list[SetupAlert] = []
    snapshots: list[dict] = []
    errors: list[str] = []

    cache_dir = Path(__file__).resolve().parent.parent / "demo_data"
    session = requests.Session()

    for key in keys:
        meta = INSTRUMENTS.get(key)
        if not meta:
            errors.append(f"Unknown instrument: {key}")
            continue
        if classes and meta["asset_class"] not in classes:
            continue
        for tf in tfs:
            try:
                series: CandleSeries = fetch_instrument(
                    key, tf, demo=demo, session=session, cache_dir=cache_dir if demo else None
                )
                snap = series.to_summary()
                snap["name"] = meta["name"]
                # Attach latest indicator highlights without requiring an alert
                from indicators import compute_all
                from scanner.setups import _last

                if len(series) >= 50:
                    ind = compute_all(series.close, series.high, series.low)
                    snap["rsi"] = round(_last(ind["rsi"]) or float("nan"), 2)
                    snap["sma20"] = round(_last(ind["sma_fast"]) or float("nan"), 6)
                    snap["sma50"] = round(_last(ind["sma_slow"]) or float("nan"), 6)
                snapshots.append(snap)
                alerts.extend(analyze_series(series, meta["name"]))
            except DataFetchError as exc:
                errors.append(f"{key} {tf}: {exc}")
            except Exception as exc:  # noqa: BLE001 — surface in report
                errors.append(f"{key} {tf}: unexpected {type(exc).__name__}: {exc}")

    return alerts, snapshots, errors


def write_outputs(alerts: list[SetupAlert], snapshots: list[dict], errors: list[str]) -> Path:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "mode": "alerts_only",
        "disclaimer": (
            "Educational market scanner. No orders are placed. "
            "Not financial advice. Public market data may be delayed."
        ),
        "alert_count": len(alerts),
        "snapshot_count": len(snapshots),
        "errors": errors,
        "alerts": [a.to_dict() for a in alerts],
        "snapshots": snapshots,
    }
    json_path = out / "latest_alerts.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out / "latest_alerts.csv"
    fields = [
        "instrument",
        "name",
        "asset_class",
        "timeframe",
        "setup",
        "side",
        "strength",
        "price",
        "message",
        "scanned_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for a in alerts:
            row = {k: a.to_dict().get(k) for k in fields}
            writer.writerow(row)

    return json_path
