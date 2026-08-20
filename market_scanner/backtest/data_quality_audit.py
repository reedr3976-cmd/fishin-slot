"""Research data-quality audit for V11 (documents sources; no silent changes)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config import BACKTEST_TIMEFRAMES, INSTRUMENTS
from models import CandleSeries


def _span_days(series: CandleSeries) -> float:
    if len(series) < 2:
        return 0.0
    return (int(series.timestamps[-1]) - int(series.timestamps[0])) / 86400.0


def audit_loaded_series(
    series_4h: dict[tuple[str, str], CandleSeries],
    daily_map: dict[str, CandleSeries],
    weekly_map: dict[str, CandleSeries],
) -> dict[str, Any]:
    """Summarise point-in-time bar coverage actually used in the study."""
    per_inst: dict[str, Any] = {}
    for (key, tf), s in series_4h.items():
        if tf != "4h":
            continue
        d = daily_map.get(key)
        w = weekly_map.get(key)
        per_inst[key] = {
            "bars_4h": len(s),
            "span_days_4h": round(_span_days(s), 1),
            "first_4h": datetime.fromtimestamp(int(s.timestamps[0]), tz=timezone.utc).isoformat()
            if len(s)
            else None,
            "last_4h": datetime.fromtimestamp(int(s.timestamps[-1]), tz=timezone.utc).isoformat()
            if len(s)
            else None,
            "bars_1d": len(d) if d is not None else 0,
            "span_days_1d": round(_span_days(d), 1) if d is not None else 0,
            "bars_1wk": len(w) if w is not None else 0,
            "span_days_1wk": round(_span_days(w), 1) if w is not None else 0,
        }
    spans = [v["span_days_4h"] for v in per_inst.values() if v["bars_4h"]]
    bars = [v["bars_4h"] for v in per_inst.values() if v["bars_4h"]]
    return {
        "instruments": per_inst,
        "median_4h_bars": int(sorted(bars)[len(bars) // 2]) if bars else 0,
        "min_4h_bars": min(bars) if bars else 0,
        "max_4h_bars": max(bars) if bars else 0,
        "median_4h_span_days": round(sorted(spans)[len(spans) // 2], 1) if spans else 0,
        "configured_4h_source": {
            "provider": "Yahoo Finance chart API",
            "raw_interval": BACKTEST_TIMEFRAMES["4h"]["interval"],
            "raw_range": BACKTEST_TIMEFRAMES["4h"]["range"],
            "aggregation": "1h bars aggregated to 4h UTC buckets (aggregate_to_4h)",
        },
    }


def probe_yahoo_1h_limits(symbol: str = "EURUSD=X") -> dict[str, Any]:
    """Probe Yahoo intraday depth without changing the live fetch path."""
    import time

    ua = "Mozilla/5.0 (compatible; MarketScanner/1.0)"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    probes = []
    now = int(time.time())
    for label, params in (
        ("730d", {"interval": "60m", "range": "730d"}),
        ("2y_period", {"interval": "60m", "period1": now - 2 * 365 * 86400, "period2": now}),
        ("3y_period", {"interval": "60m", "period1": now - 3 * 365 * 86400, "period2": now}),
        ("max_range", {"interval": "60m", "range": "max"}),
    ):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": ua}, timeout=25)
            if r.status_code != 200:
                probes.append({"label": label, "status": "error", "http": r.status_code})
                continue
            ts = (r.json().get("chart") or {}).get("result", [{}])[0].get("timestamp") or []
            probes.append(
                {
                    "label": label,
                    "status": "ok",
                    "bars_1h": len(ts),
                    "first": datetime.fromtimestamp(ts[0], tz=timezone.utc).date().isoformat() if ts else None,
                    "last": datetime.fromtimestamp(ts[-1], tz=timezone.utc).date().isoformat() if ts else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            probes.append({"label": label, "status": "exception", "detail": str(exc)})

    return {
        "symbol_tested": symbol,
        "probes": probes,
        "conclusion": (
            "Yahoo 1h FX/liquid symbols support ~730 days (~2 years) of dense intraday history. "
            "Requests beyond ~2 years return HTTP 422. The `max` range returns sparse legacy bars "
            "unsuitable for 4H research. No alternative intraday source was integrated in V11 "
            "because Stooq intraday fetch failed in environment probe and longer Yahoo history "
            "is unavailable without a paid/vendor feed."
        ),
        "recommendation": (
            "Treat V11 results as valid only within the audited span (~2y 4H). "
            "Do not over-fit sparse history. If a licensed OHLC vendor is approved later, "
            "add an explicit research-only loader and re-document before changing studies."
        ),
    }


def full_data_audit(
    series_4h: dict[tuple[str, str], CandleSeries],
    daily_map: dict[str, CandleSeries],
    weekly_map: dict[str, CandleSeries],
    *,
    probe_symbol: Optional[str] = None,
) -> dict[str, Any]:
    sym = probe_symbol or INSTRUMENTS.get("EURUSD", {}).get("symbol", "EURUSD=X")
    loaded = audit_loaded_series(series_4h, daily_map, weekly_map)
    probe = probe_yahoo_1h_limits(sym)
    return {
        "loaded_coverage": loaded,
        "yahoo_intraday_probe": probe,
        "daily_1d_source": {"provider": "Yahoo", "range": BACKTEST_TIMEFRAMES["1d"]["range"]},
        "weekly_source": {"provider": "Yahoo", "range": BACKTEST_TIMEFRAMES["1wk"]["range"]},
        "data_changed_from_v10": False,
        "note": "V11 uses the same Yahoo loader as V10; audit documents limitations only.",
    }
