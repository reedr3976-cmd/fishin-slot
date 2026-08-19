"""V12 data-source comparison audit (documented; no silent substitution)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from providers.dukascopy_data import DUKASCOPY_INSTRUMENTS, DUKASCOPY_UNAVAILABLE


def _yahoo_probe(symbol: str = "EURUSD=X") -> dict[str, Any]:
    import time

    ua = "Mozilla/5.0 (compatible; MarketScanner/1.0)"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    rows = []
    for label, params in (
        ("730d_1h", {"interval": "60m", "range": "730d"}),
        ("2y_period", {"interval": "60m", "period1": int(time.time()) - 2 * 365 * 86400, "period2": int(time.time())}),
        ("5y_1d", {"interval": "1d", "range": "5y"}),
    ):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": ua}, timeout=25)
            if r.status_code != 200:
                rows.append({"label": label, "status": "error", "http": r.status_code})
                continue
            ts = (r.json().get("chart") or {}).get("result", [{}])[0].get("timestamp") or []
            rows.append(
                {
                    "label": label,
                    "bars": len(ts),
                    "first": datetime.fromtimestamp(ts[0], tz=timezone.utc).date().isoformat() if ts else None,
                    "last": datetime.fromtimestamp(ts[-1], tz=timezone.utc).date().isoformat() if ts else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"label": label, "status": "exception", "detail": str(exc)})
    return {"symbol": symbol, "probes": rows}


def _dukascopy_probe() -> dict[str, Any]:
    try:
        import dukascopy_python as dc
        import dukascopy_python.instruments as inst
        from datetime import datetime, timezone

        start = datetime(2010, 1, 1, tzinfo=timezone.utc)
        end = datetime(2011, 1, 1, tzinfo=timezone.utc)
        df = dc.fetch(inst.INSTRUMENT_FX_MAJORS_EUR_USD, dc.INTERVAL_HOUR_1, dc.OFFER_SIDE_BID, start, end)
        return {
            "status": "ok",
            "sample": "EURUSD 2010",
            "bars_1h": len(df),
            "cost": "free",
            "api_key_required": False,
            "history_from": "2003+ major FX (per Dukascopy docs)",
            "timezone": "UTC",
            "adjustment": "bid-side CFD/spot feed; no dividend adjustment on FX",
            "mapped_instruments": len(DUKASCOPY_INSTRUMENTS),
            "unavailable": DUKASCOPY_UNAVAILABLE,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


def build_data_source_audit(*, yahoo_symbol: str = "EURUSD=X") -> dict[str, Any]:
    """Compare viable sources; paid/keyed options documented but NOT integrated."""
    return {
        "audit_date": datetime.now(timezone.utc).isoformat(),
        "v12_selected_primary": "Dukascopy (dukascopy-python)",
        "v12_selected_secondary": "Yahoo Finance (short 1h fallback for symbols absent on Dukascopy)",
        "integration_policy": (
            "No paid subscriptions purchased. No API keys required for Dukascopy. "
            "HF Data Library and EODHD offer longer US equity intraday but require free registration/API keys — "
            "NOT integrated; listed for user decision."
        ),
        "sources": [
            {
                "name": "Yahoo Finance (existing V6–V11)",
                "cost": "free",
                "api_key": False,
                "depth_1h": "~730 days dense (422 beyond ~2y)",
                "depth_1d": "~5 years",
                "asset_coverage": "FX, commodities (futures proxies), stocks, macro proxies",
                "adjustment": "vendor-adjusted; stock splits reflected in vendor series",
                "timezone": "exchange/vendor dependent; aggregated to UTC 4H buckets",
                "suitability": "adequate for HTF/daily; insufficient for long 4H research",
                "integrated_v12": "fallback only",
                "probe": _yahoo_probe(yahoo_symbol),
            },
            {
                "name": "Dukascopy public datafeed",
                "cost": "free",
                "api_key": False,
                "depth_1h": "~2003–present (FX majors); multi-year for metals/energy/US CFDs",
                "depth_4h": "causal aggregation from 1h",
                "asset_coverage": "FX, metals, energy CFDs, US equity/ETF CFDs",
                "adjustment": "bid-side CFD; US stocks are CFD not cash; commodities are CFD not exchange futures",
                "timezone": "UTC",
                "session": "24h FX; US CFDs follow vendor session; weekends omitted",
                "roll_handling": "continuous CFD — no explicit futures roll calendar",
                "suitability": "PRIMARY for V12 extended intraday research",
                "integrated_v12": True,
                "probe": _dukascopy_probe(),
                "instrument_map_size": len(DUKASCOPY_INSTRUMENTS),
            },
            {
                "name": "HF Data Library",
                "cost": "free (CC BY 4.0)",
                "api_key": True,
                "registration": "free account + email verification",
                "depth_1min": "Dec 2002–present US equities/ETFs",
                "asset_coverage": "1,391 US stocks/ETFs",
                "suitability": "best free long US equity intraday IF user registers",
                "integrated_v12": False,
                "reason_not_integrated": "requires API key/account — user must approve registration first",
                "url": "https://hfdatalibrary.com/",
            },
            {
                "name": "EODHD Intraday API",
                "cost": "freemium / paid tiers",
                "api_key": True,
                "depth_1h": "up to 7200 days per request (paid/free tier limits apply)",
                "asset_coverage": "global stocks, FX, crypto",
                "integrated_v12": False,
                "reason_not_integrated": "requires API key; paid tiers for production volume",
                "url": "https://eodhd.com/financial-apis/intraday-historical-data-api",
            },
            {
                "name": "Polygon.io / IQFeed / Norgate / Bloomberg",
                "cost": "paid",
                "integrated_v12": False,
                "reason_not_integrated": "paid subscription — not purchased per V12 instructions",
            },
        ],
        "comparison_summary": {
            "best_free_no_key": "Dukascopy for FX + CFD commodities + US equity CFDs",
            "best_free_us_stocks_long_intraday": "HF Data Library (requires free API key)",
            "v11_yahoo_limitation_confirmed": True,
        },
    }
