"""Load extended research data for V12 (Dukascopy primary, Yahoo fallback)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

import requests

from config import INSTRUMENTS, V12_DUKA_START_ISO
from models import CandleSeries
from providers.dukascopy_data import (
    DUKASCOPY_INSTRUMENTS,
    DUKASCOPY_UNAVAILABLE,
    aggregate_daily,
    aggregate_weekly,
    fetch_4h,
)
from providers.yahoo import fetch_instrument


def _parse_start(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def load_extended_panel(
    instruments: Iterable[str],
    *,
    start_iso: Optional[str] = None,
    end: Optional[datetime] = None,
    include_macro_yahoo: bool = True,
) -> tuple[dict[tuple[str, str], CandleSeries], dict[str, CandleSeries], dict[str, CandleSeries], list[str], dict]:
    """Return (series_4h map, daily_map, weekly_map, errors, provenance)."""
    start = _parse_start(start_iso or V12_DUKA_START_ISO)
    end = end or datetime.now(timezone.utc)
    series_4h: dict[tuple[str, str], CandleSeries] = {}
    daily_map: dict[str, CandleSeries] = {}
    weekly_map: dict[str, CandleSeries] = {}
    errors: list[str] = []
    provenance: dict[str, dict] = {}

    session = requests.Session()
    for key in instruments:
        if key not in INSTRUMENTS:
            errors.append(f"Unknown instrument {key}")
            continue
        try:
            if key in DUKASCOPY_INSTRUMENTS:
                print(f"  extended load {key} 4h (Dukascopy)...", flush=True)
                s4 = fetch_4h(key, start, end)
                note = DUKASCOPY_INSTRUMENTS[key][1]
                src = "dukascopy_bid_cfd"
            elif key in DUKASCOPY_UNAVAILABLE:
                print(f"  extended load {key} 4h (Yahoo fallback — {DUKASCOPY_UNAVAILABLE[key]})...", flush=True)
                s4 = fetch_instrument(key, "4h", session=session, for_backtest=True)
                note = DUKASCOPY_UNAVAILABLE[key]
                src = "yahoo_short_1h_agg"
            elif include_macro_yahoo and INSTRUMENTS[key].get("asset_class") == "macro":
                print(f"  macro load {key} 1d (Yahoo)...", flush=True)
                s1d = fetch_instrument(key, "1d", session=session, for_backtest=True)
                daily_map[key] = s1d
                provenance[key] = {"source": "yahoo_1d", "note": "macro context only"}
                continue
            else:
                print(f"  extended load {key} 4h (Yahoo fallback)...", flush=True)
                s4 = fetch_instrument(key, "4h", session=session, for_backtest=True)
                note = "No Dukascopy mapping"
                src = "yahoo_short_1h_agg"

            series_4h[(key, "4h")] = s4
            daily_map[key] = aggregate_daily(s4)
            weekly_map[key] = aggregate_weekly(s4)
            provenance[key] = {
                "source": src,
                "note": note,
                "bars_4h": len(s4),
                "first_ts": int(s4.timestamps[0]),
                "last_ts": int(s4.timestamps[-1]),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    return series_4h, daily_map, weekly_map, errors, provenance
