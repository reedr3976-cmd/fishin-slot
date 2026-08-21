"""Dukascopy historical OHLC loader for V12 extended research (free, no API key).

Data are Dukascopy bid-side CFD/spot feed bars (UTC). US names are equity CFDs,
not cash equities. Commodity symbols are CFD/futures proxies — see instrument notes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from models import CandleSeries

try:
    import dukascopy_python as dc
    import dukascopy_python.instruments as dc_inst
except ImportError as exc:  # pragma: no cover
    raise ImportError("dukascopy-python required for V12 extended data") from exc

from providers.yahoo import aggregate_to_4h

CACHE_DIR = Path(__file__).resolve().parent.parent / "research_cache" / "dukascopy"

# Research instrument key -> (dukascopy constant name, human note)
DUKASCOPY_INSTRUMENTS: dict[str, tuple[str, str]] = {
    "EURUSD": ("INSTRUMENT_FX_MAJORS_EUR_USD", "FX spot CFD bid"),
    "GBPUSD": ("INSTRUMENT_FX_MAJORS_GBP_USD", "FX spot CFD bid"),
    "USDJPY": ("INSTRUMENT_FX_MAJORS_USD_JPY", "FX spot CFD bid"),
    "AUDUSD": ("INSTRUMENT_FX_MAJORS_AUD_USD", "FX spot CFD bid"),
    "USDCAD": ("INSTRUMENT_FX_MAJORS_USD_CAD", "FX spot CFD bid"),
    "USDCHF": ("INSTRUMENT_FX_MAJORS_USD_CHF", "FX spot CFD bid"),
    "XAUUSD": ("INSTRUMENT_FX_METALS_XAU_USD", "Metal CFD bid"),
    "XAGUSD": ("INSTRUMENT_FX_METALS_XAG_USD", "Metal CFD bid"),
    "USOIL": ("INSTRUMENT_CMD_ENERGY_E_LIGHT", "WTI energy CFD bid"),
    "NATGAS": ("INSTRUMENT_CMD_ENERGY_GAS_CMD_USD", "Natural gas CFD bid"),
    "COPPER": ("INSTRUMENT_CMD_METALS_COPPER_CMD_USD", "Copper CFD bid"),
    "SPY": ("INSTRUMENT_ETF_CFD_US_SPY_US_USD", "US ETF CFD bid (not cash ETF)"),
    "QQQ": ("INSTRUMENT_ETF_CFD_US_QQQ_US_USD", "US ETF CFD bid"),
    "AAPL": ("INSTRUMENT_US_AAPL_US_USD", "US equity CFD bid"),
    "MSFT": ("INSTRUMENT_US_MSFT_US_USD", "US equity CFD bid"),
    "XOM": ("INSTRUMENT_US_XOM_US_USD", "US equity CFD bid"),
    "AMZN": ("INSTRUMENT_US_AMZN_US_USD", "US equity CFD bid"),
    "GOOGL": ("INSTRUMENT_US_GOOGL_US_USD", "US equity CFD bid"),
    "META": ("INSTRUMENT_US_FB_US_USD", "US equity CFD bid (FB ticker on Dukascopy)"),
    "NVDA": ("INSTRUMENT_US_NVDA_US_USD", "US equity CFD bid"),
    "JPM": ("INSTRUMENT_US_JPM_US_USD", "US equity CFD bid"),
    "JNJ": ("INSTRUMENT_US_JNJ_US_USD", "US equity CFD bid"),
    "WMT": ("INSTRUMENT_US_WMT_US_USD", "US equity CFD bid"),
    "BA": ("INSTRUMENT_US_BA_US_USD", "US equity CFD bid"),
    "DIS": ("INSTRUMENT_US_DIS_US_USD", "US equity CFD bid"),
}

# Not available on Dukascopy — must use Yahoo short history if needed
DUKASCOPY_UNAVAILABLE = {"CORN": "No Dukascopy symbol; Yahoo 1h proxy only (~730d)"}


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fetch_hourly_range(
    instrument_key: str,
    start: datetime,
    end: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    const_name, _ = DUKASCOPY_INSTRUMENTS[instrument_key]
    instrument = getattr(dc_inst, const_name)
    df = dc.fetch(instrument, dc.INTERVAL_HOUR_1, dc.OFFER_SIDE_BID, start, end)
    if df is None or len(df) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    idx = df.index
    if idx.tzinfo is None:
        idx = idx.tz_localize(timezone.utc)
    else:
        idx = idx.tz_convert(timezone.utc)
    # dukascopy-python returns datetime64[ms]; convert to Unix seconds
    raw = idx.astype(np.int64)
    unit = getattr(idx.dtype, "unit", "ns")
    if unit == "ms":
        ts = (raw // 1_000).astype(np.int64)
    elif unit == "ns":
        ts = (raw // 1_000_000_000).astype(np.int64)
    else:
        ts = (raw // 1_000_000).astype(np.int64)  # us fallback
    return (
        ts,
        df["open"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].fillna(0.0).to_numpy(dtype=np.float64),
    )


def fetch_hourly_cached(
    instrument_key: str,
    start: datetime,
    end: datetime,
    *,
    cache_dir: Optional[Path] = None,
) -> CandleSeries:
    """Fetch 1h bars year-by-year with disk cache."""
    if instrument_key not in DUKASCOPY_INSTRUMENTS:
        raise KeyError(f"No Dukascopy mapping for {instrument_key}")

    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{instrument_key}_1h_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    all_ts, all_o, all_h, all_l, all_c, all_v = [], [], [], [], [], []
    year = start.year
    while year <= end.year:
        y_start = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
        y_end = min(end, datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc))
        cache_file = cache_dir / f"{instrument_key}_1h_{year}.npz"
        loaded = False
        if cache_file.exists() and meta.get(str(year)) in ("ok", "empty"):
            if meta.get(str(year)) == "empty":
                ts = np.array([], dtype=np.int64)
                o = h = l = c = v = ts
                loaded = True
            else:
                z = np.load(cache_file)
                ts, o, h, l, c, v = z["ts"], z["o"], z["h"], z["l"], z["c"], z["v"]
                # Invalidate legacy cache written with wrong timestamp scaling (ms treated as ns)
                if len(ts) and int(ts.max()) < 1_000_000_000:
                    meta[str(year)] = "bad_ts"
                    cache_file.unlink(missing_ok=True)
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                else:
                    loaded = True
        if not loaded:
            print(f"    duka fetch {instrument_key} {year}...", flush=True)
            ts, o, h, l, c, v = _fetch_hourly_range(instrument_key, y_start, y_end)
            np.savez_compressed(cache_file, ts=ts, o=o, h=h, l=l, c=c, v=v)
            meta[str(year)] = "ok" if len(ts) else "empty"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            time.sleep(0.2)
        if len(ts):
            all_ts.append(ts)
            all_o.append(o)
            all_h.append(h)
            all_l.append(l)
            all_c.append(c)
            all_v.append(v)
        year += 1

    if not all_ts:
        raise RuntimeError(f"Dukascopy returned no data for {instrument_key}")

    ts = np.concatenate(all_ts)
    order = np.argsort(ts)
    ts = ts[order]
    from config import INSTRUMENTS

    meta_cfg = INSTRUMENTS[instrument_key]
    return CandleSeries(
        instrument=instrument_key,
        symbol=meta_cfg["symbol"],
        asset_class=meta_cfg["asset_class"],
        timeframe="1h",
        timestamps=ts,
        open=np.concatenate(all_o)[order],
        high=np.concatenate(all_h)[order],
        low=np.concatenate(all_l)[order],
        close=np.concatenate(all_c)[order],
        volume=np.concatenate(all_v)[order],
    )


def fetch_4h(
    instrument_key: str,
    start: datetime,
    end: datetime,
    *,
    cache_dir: Optional[Path] = None,
) -> CandleSeries:
    hourly = fetch_hourly_cached(instrument_key, start, end, cache_dir=cache_dir)
    s4 = aggregate_to_4h(hourly)
    s4.timeframe = "4h"
    return s4


def aggregate_daily(series_4h: CandleSeries) -> CandleSeries:
    """Causal daily bars from 4H (UTC date buckets)."""
    day = series_4h.timestamps // 86400
    uniq = np.unique(day)
    rows = []
    for d in uniq:
        m = day == d
        rows.append(
            (
                int(series_4h.timestamps[m][0]),
                float(series_4h.open[m][0]),
                float(np.max(series_4h.high[m])),
                float(np.min(series_4h.low[m])),
                float(series_4h.close[m][-1]),
                float(np.sum(series_4h.volume[m])),
            )
        )
    arr = np.array(rows, dtype=np.float64)
    return CandleSeries(
        instrument=series_4h.instrument,
        symbol=series_4h.symbol,
        asset_class=series_4h.asset_class,
        timeframe="1d",
        timestamps=arr[:, 0].astype(np.int64),
        open=arr[:, 1],
        high=arr[:, 2],
        low=arr[:, 3],
        close=arr[:, 4],
        volume=arr[:, 5],
    )


def aggregate_weekly(series_4h: CandleSeries) -> CandleSeries:
    """Causal weekly bars from 4H (ISO week buckets, UTC)."""
    ts = series_4h.timestamps.astype(np.int64)
    # week id = timestamp // (7*86400)
    week = ts // (7 * 86400)
    uniq = np.unique(week)
    rows = []
    for w in uniq:
        m = week == w
        rows.append(
            (
                int(ts[m][0]),
                float(series_4h.open[m][0]),
                float(np.max(series_4h.high[m])),
                float(np.min(series_4h.low[m])),
                float(series_4h.close[m][-1]),
                float(np.sum(series_4h.volume[m])),
            )
        )
    arr = np.array(rows, dtype=np.float64)
    return CandleSeries(
        instrument=series_4h.instrument,
        symbol=series_4h.symbol,
        asset_class=series_4h.asset_class,
        timeframe="1wk",
        timestamps=arr[:, 0].astype(np.int64),
        open=arr[:, 1],
        high=arr[:, 2],
        low=arr[:, 3],
        close=arr[:, 4],
        volume=arr[:, 5],
    )
