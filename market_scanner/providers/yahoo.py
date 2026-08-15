"""Market data providers — public sources only, no brokerage login."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote as url_quote

import numpy as np
import requests

from config import INSTRUMENTS, REQUEST_DELAY_SEC, TIMEFRAMES
from models import CandleSeries

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = (
    "Mozilla/5.0 (compatible; MarketScanner/1.0; "
    "+https://github.com/reedr3976-cmd/fishin-slot; educational-alerts-only)"
)


class DataFetchError(RuntimeError):
    pass


def _clean_arrays(
    ts: list, o: list, h: list, l: list, c: list, v: list
) -> tuple[np.ndarray, ...]:
    """Drop bars with missing OHLC."""
    rows = []
    for i in range(len(ts)):
        if None in (o[i], h[i], l[i], c[i]):
            continue
        vol = 0.0 if v[i] is None else float(v[i])
        rows.append(
            (int(ts[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i]), vol)
        )
    if not rows:
        raise DataFetchError("No valid OHLC bars after cleaning")
    arr = np.array(rows, dtype=np.float64)
    return (
        arr[:, 0].astype(np.int64),
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],
        arr[:, 4],
        arr[:, 5],
    )


def aggregate_to_4h(series: CandleSeries) -> CandleSeries:
    """Aggregate 1h bars into 4h bars."""
    if len(series) < 4:
        raise DataFetchError("Not enough 1h bars to build 4h series")

    # Align to 4-hour UTC buckets
    bucket = series.timestamps // 14400
    uniq = np.unique(bucket)
    outs = []
    for b in uniq:
        mask = bucket == b
        outs.append(
            (
                int(series.timestamps[mask][0]),
                float(series.open[mask][0]),
                float(np.max(series.high[mask])),
                float(np.min(series.low[mask])),
                float(series.close[mask][-1]),
                float(np.sum(series.volume[mask])),
            )
        )
    arr = np.array(outs, dtype=np.float64)
    return CandleSeries(
        instrument=series.instrument,
        symbol=series.symbol,
        asset_class=series.asset_class,
        timeframe="4h",
        timestamps=arr[:, 0].astype(np.int64),
        open=arr[:, 1],
        high=arr[:, 2],
        low=arr[:, 3],
        close=arr[:, 4],
        volume=arr[:, 5],
    )


def fetch_yahoo_ohlcv(
    instrument_key: str,
    timeframe: str,
    session: Optional[requests.Session] = None,
    *,
    range_override: Optional[str] = None,
    interval_override: Optional[str] = None,
) -> CandleSeries:
    if instrument_key not in INSTRUMENTS:
        raise DataFetchError(f"Unknown instrument: {instrument_key}")
    if timeframe not in TIMEFRAMES:
        raise DataFetchError(f"Unknown timeframe: {timeframe}")

    meta = INSTRUMENTS[instrument_key]
    tf = TIMEFRAMES[timeframe]
    y_interval = interval_override or tf["interval"]
    y_range = range_override or tf["range"]
    symbol = meta["symbol"]

    sess = session or requests.Session()
    url = YAHOO_CHART.format(symbol=url_quote(symbol, safe="="))
    params = {"interval": y_interval, "range": y_range}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    resp = sess.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code == 429:
        raise DataFetchError(
            f"Yahoo rate-limited for {symbol}. Wait a minute and retry."
        )
    if resp.status_code != 200:
        raise DataFetchError(f"Yahoo HTTP {resp.status_code} for {symbol}")

    payload = resp.json()
    result = (payload.get("chart") or {}).get("result")
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise DataFetchError(f"Yahoo empty result for {symbol}: {err}")

    chart = result[0]
    ts = chart.get("timestamp") or []
    quotes = (chart.get("indicators") or {}).get("quote") or [{}]
    q0 = quotes[0]
    o, h, l, c, v = (
        q0.get("open") or [],
        q0.get("high") or [],
        q0.get("low") or [],
        q0.get("close") or [],
        q0.get("volume") or [],
    )
    timestamps, open_, high, low, close, volume = _clean_arrays(ts, o, h, l, c, v)

    series = CandleSeries(
        instrument=instrument_key,
        symbol=symbol,
        asset_class=meta["asset_class"],
        timeframe=timeframe if timeframe != "4h" else "1h",
        timestamps=timestamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    if timeframe == "4h":
        series = aggregate_to_4h(series)
    return series


def fetch_instrument(
    instrument_key: str,
    timeframe: str,
    *,
    demo: bool = False,
    session: Optional[requests.Session] = None,
    cache_dir: Optional[Path] = None,
    for_backtest: bool = False,
) -> CandleSeries:
    """Fetch live public data, or load/generate demo historical series."""
    if demo:
        # Longer synthetic history for offline backtests
        n = 400 if for_backtest else 200
        return load_or_build_demo(instrument_key, timeframe, cache_dir, n_bars=n)

    if for_backtest:
        from config import BACKTEST_TIMEFRAMES

        bt = BACKTEST_TIMEFRAMES.get(timeframe, {})
        series = fetch_yahoo_ohlcv(
            instrument_key,
            timeframe,
            session=session,
            range_override=bt.get("range"),
            interval_override=bt.get("interval"),
        )
    else:
        series = fetch_yahoo_ohlcv(instrument_key, timeframe, session=session)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{instrument_key}_{timeframe}.json"
        save_series_json(series, path)
    time.sleep(REQUEST_DELAY_SEC)
    return series


def save_series_json(series: CandleSeries, path: Path) -> None:
    data = {
        "instrument": series.instrument,
        "symbol": series.symbol,
        "asset_class": series.asset_class,
        "timeframe": series.timeframe,
        "timestamps": series.timestamps.tolist(),
        "open": series.open.tolist(),
        "high": series.high.tolist(),
        "low": series.low.tolist(),
        "close": series.close.tolist(),
        "volume": series.volume.tolist(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def load_series_json(path: Path) -> CandleSeries:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CandleSeries(
        instrument=data["instrument"],
        symbol=data["symbol"],
        asset_class=data["asset_class"],
        timeframe=data["timeframe"],
        timestamps=np.asarray(data["timestamps"], dtype=np.int64),
        open=np.asarray(data["open"], dtype=np.float64),
        high=np.asarray(data["high"], dtype=np.float64),
        low=np.asarray(data["low"], dtype=np.float64),
        close=np.asarray(data["close"], dtype=np.float64),
        volume=np.asarray(data["volume"], dtype=np.float64),
    )


def _synthetic_series(instrument_key: str, timeframe: str, n: int = 200) -> CandleSeries:
    """Deterministic synthetic OHLCV for offline unit tests."""
    meta = INSTRUMENTS[instrument_key]
    seed = sum(ord(ch) for ch in instrument_key + timeframe) % 10_000
    rng = np.random.default_rng(seed)
    # Rough starting levels by asset class
    base = {
        "forex": 1.10,
        "crypto": 50_000.0 if instrument_key.startswith("BTC") else 2_000.0,
        "commodity": 2_000.0 if "XAU" in instrument_key else 70.0,
        "stocks": 100.0,
    }.get(meta["asset_class"], 100.0)
    if instrument_key == "USDJPY":
        base = 150.0
    if instrument_key == "XAGUSD":
        base = 30.0
    if instrument_key == "SOLUSD":
        base = 140.0
    if instrument_key == "ETHUSD":
        base = 3_000.0

    step = {"1h": 3600, "4h": 14400, "1d": 86400, "1wk": 604800}[timeframe]
    now = int(time.time())
    # Align to step
    now = now - (now % step)
    timestamps = np.arange(now - (n - 1) * step, now + 1, step, dtype=np.int64)

    rets = rng.normal(0.0002, 0.01, size=n)
    # Inject a mild trend + a few swings so indicators can fire
    rets[n // 3 : n // 3 + 10] += 0.015
    rets[2 * n // 3 : 2 * n // 3 + 10] -= 0.012
    close = base * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.008, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.008, n))
    volume = rng.uniform(1e3, 1e5, n)

    return CandleSeries(
        instrument=instrument_key,
        symbol=meta["symbol"],
        asset_class=meta["asset_class"],
        timeframe=timeframe,
        timestamps=timestamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def load_or_build_demo(
    instrument_key: str,
    timeframe: str,
    cache_dir: Optional[Path] = None,
    n_bars: int = 200,
) -> CandleSeries:
    cache_dir = cache_dir or Path(__file__).resolve().parent.parent / "demo_data"
    path = cache_dir / f"{instrument_key}_{timeframe}.json"
    if path.exists():
        series = load_series_json(path)
        if len(series) >= n_bars:
            return series
    series = _synthetic_series(instrument_key, timeframe, n=n_bars)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_series_json(series, path)
    return series
