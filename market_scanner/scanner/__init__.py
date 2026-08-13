"""Multi-instrument / multi-timeframe market scanner (alerts & ranking only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional

import requests

from config import (
    DAILY_SUMMARY_JSON,
    DAILY_SUMMARY_TXT,
    DEFAULT_TIMEFRAMES,
    INSTRUMENTS,
    OUTPUT_DIR,
)
from models import CandleSeries
from providers.yahoo import DataFetchError, fetch_instrument
from scanner.opportunity import Opportunity, evaluate_opportunity, rank_opportunities
from scanner.report import build_daily_summary
from scanner.setups import SetupAlert, analyze_series


def scan_markets(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
    asset_classes: Optional[Iterable[str]] = None,
) -> tuple[list[SetupAlert], list[dict], list[str]]:
    """Legacy scan API: returns (alerts, snapshots, errors)."""
    opportunities, snapshots, errors = scan_opportunities(
        instruments, timeframes, demo=demo, asset_classes=asset_classes
    )
    # Derive simple alerts for backward-compatible JSON/CSV consumers
    alerts: list[SetupAlert] = []
    for opp in opportunities:
        if opp.confidence == "NO STRONG SETUP":
            continue
        alerts.append(
            SetupAlert(
                instrument=opp.instrument,
                name=opp.name,
                asset_class=opp.asset_class,
                timeframe=opp.timeframe,
                setup=opp.confidence.lower() + "_opportunity",
                side=opp.direction,
                strength=opp.confidence.lower(),
                price=opp.price,
                message=opp.reason,
                metrics={
                    "rsi": opp.rsi,
                    "sma_fast": opp.sma20,
                    "sma_slow": opp.sma50,
                    "macd_condition": opp.macd_condition,
                    "atr": opp.atr,
                    "score": opp.score,
                    "support": opp.support,
                    "resistance": opp.resistance,
                },
                scanned_at=opp.scanned_at,
            )
        )
    return alerts, snapshots, errors


def scan_opportunities(
    instruments: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    *,
    demo: bool = False,
    asset_classes: Optional[Iterable[str]] = None,
) -> tuple[list[Opportunity], list[dict], list[str]]:
    """Scan markets and return ranked opportunity cards + snapshots + errors."""
    keys = list(instruments) if instruments else list(INSTRUMENTS.keys())
    tfs = list(timeframes) if timeframes else list(DEFAULT_TIMEFRAMES)
    classes = set(asset_classes) if asset_classes else None

    opportunities: list[Opportunity] = []
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
                    key,
                    tf,
                    demo=demo,
                    session=session,
                    cache_dir=cache_dir if demo else None,
                )
                snap = series.to_summary()
                snap["name"] = meta["name"]
                opp = evaluate_opportunity(series, meta["name"])
                snap["rsi"] = opp.rsi
                snap["sma20"] = opp.sma20
                snap["sma50"] = opp.sma50
                snap["score"] = opp.score
                snap["confidence"] = opp.confidence
                snap["direction"] = opp.direction
                snapshots.append(snap)
                opportunities.append(opp)
            except DataFetchError as exc:
                errors.append(f"{key} {tf}: {exc}")
            except Exception as exc:  # noqa: BLE001 — surface in report
                errors.append(f"{key} {tf}: unexpected {type(exc).__name__}: {exc}")

    return rank_opportunities(opportunities), snapshots, errors


def write_outputs(
    alerts: list[SetupAlert],
    snapshots: list[dict],
    errors: list[str],
    opportunities: Optional[list[Opportunity]] = None,
    *,
    mode_label: str = "public/historical",
) -> Path:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    ranked = rank_opportunities(opportunities or [])
    payload = {
        "mode": "alerts_only",
        "disclaimer": (
            "Educational market scanner. No orders are placed. "
            "Not financial advice. Public market data may be delayed."
        ),
        "alert_count": len(alerts),
        "opportunity_count": len(ranked),
        "snapshot_count": len(snapshots),
        "errors": errors,
        "opportunities": [o.to_dict() for o in ranked],
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
        "direction",
        "confidence",
        "score",
        "price",
        "reason",
        "rsi",
        "sma20",
        "sma50",
        "macd_condition",
        "atr",
        "support",
        "resistance",
        "scanned_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for o in ranked:
            writer.writerow(
                {
                    "instrument": o.instrument,
                    "name": o.name,
                    "asset_class": o.asset_class,
                    "timeframe": o.timeframe,
                    "direction": o.direction,
                    "confidence": o.confidence,
                    "score": o.score,
                    "price": o.price,
                    "reason": o.reason,
                    "rsi": o.rsi,
                    "sma20": o.sma20,
                    "sma50": o.sma50,
                    "macd_condition": o.macd_condition,
                    "atr": o.atr,
                    "support": o.support,
                    "resistance": o.resistance,
                    "scanned_at": o.scanned_at,
                }
            )

    # Beginner daily summary
    summary_text = build_daily_summary(ranked, mode_label=mode_label, errors=errors)
    Path(DAILY_SUMMARY_TXT).write_text(summary_text, encoding="utf-8")
    Path(DAILY_SUMMARY_JSON).write_text(
        json.dumps(
            {
                "mode": "alerts_only",
                "mode_label": mode_label,
                "opportunities": [o.to_dict() for o in ranked],
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return json_path
