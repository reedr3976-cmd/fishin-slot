"""Technical indicators implemented with NumPy (no paid libraries)."""

from __future__ import annotations

import numpy as np


def sma(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if period <= 0 or len(values) < period:
        return out
    csum = np.cumsum(values, dtype=np.float64)
    out[period - 1 :] = (csum[period - 1 :] - np.concatenate([[0.0], csum[:-period]])) / period
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if period <= 0 or len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = sma(gain, period)
    avg_loss = sma(loss, period)
    # Wilder smoothing after seed
    for i in range(period, len(close)):
        if i == period:
            continue
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss > 0)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[:period] = np.nan
    # When avg_loss == 0 and avg_gain > 0 => RSI 100
    zero_loss = (avg_loss == 0) & (avg_gain > 0)
    out[zero_loss] = 100.0
    return out


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    # Signal EMA only over valid MACD values — compute from first non-nan
    signal_line = np.full_like(line, np.nan)
    valid_idx = np.where(~np.isnan(line))[0]
    if len(valid_idx) >= signal:
        start = int(valid_idx[0])
        segment = line[start:]
        sig_seg = ema(segment, signal)
        signal_line[start:] = sig_seg
    hist = line - signal_line
    return line, signal_line, hist


def bollinger(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, period)
    out_std = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period - 1, len(close)):
        out_std[i] = np.std(close[i - period + 1 : i + 1], ddof=0)
    upper = mid + num_std * out_std
    lower = mid - num_std * out_std
    return upper, mid, lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < 2:
        return out
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return sma(tr, period)


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder ADX / +DI / -DI. Values at i use only bars ≤ i."""
    n = len(close)
    adx_out = np.full(n, np.nan, dtype=np.float64)
    pdi = np.full(n, np.nan, dtype=np.float64)
    mdi = np.full(n, np.nan, dtype=np.float64)
    if n < period + 2:
        return adx_out, pdi, mdi

    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close[:-1]
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )

    # Wilder smooth seeded with SMA
    atr_w = np.full(n, np.nan)
    pdm_w = np.full(n, np.nan)
    mdm_w = np.full(n, np.nan)
    atr_w[period] = np.sum(tr[:period])
    pdm_w[period] = np.sum(plus_dm[:period])
    mdm_w[period] = np.sum(minus_dm[:period])
    for i in range(period + 1, n):
        atr_w[i] = atr_w[i - 1] - (atr_w[i - 1] / period) + tr[i - 1]
        pdm_w[i] = pdm_w[i - 1] - (pdm_w[i - 1] / period) + plus_dm[i - 1]
        mdm_w[i] = mdm_w[i - 1] - (mdm_w[i - 1] / period) + minus_dm[i - 1]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr_w[i] and atr_w[i] > 0:
            pdi[i] = 100.0 * pdm_w[i] / atr_w[i]
            mdi[i] = 100.0 * mdm_w[i] / atr_w[i]
            denom = pdi[i] + mdi[i]
            if denom > 0:
                dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / denom

    # ADX = Wilder smooth of DX
    first = period * 2 - 1
    if first < n and not np.isnan(dx[period:first + 1]).any():
        adx_out[first] = np.nanmean(dx[period : first + 1])
        for i in range(first + 1, n):
            if np.isnan(dx[i]) or np.isnan(adx_out[i - 1]):
                continue
            adx_out[i] = ((adx_out[i - 1] * (period - 1)) + dx[i]) / period
    return adx_out, pdi, mdi


def swing_structure_dir(
    high: np.ndarray, low: np.ndarray, i: int, *, pivot: int = 2, max_look: int = 60
) -> int:
    """Return +1 HH+HL, -1 LH+LL, 0 otherwise. Uses only bars ≤ i."""
    if i < pivot * 4 + 2:
        return 0
    start = max(pivot, i - max_look)
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    # Confirmed pivots need `pivot` bars on the right → last confirmable index is i-pivot
    last = i - pivot
    for j in range(start + pivot, last + 1):
        window_h = high[j - pivot : j + pivot + 1]
        window_l = low[j - pivot : j + pivot + 1]
        if high[j] >= np.max(window_h):
            swing_highs.append(float(high[j]))
        if low[j] <= np.min(window_l):
            swing_lows.append(float(low[j]))
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0
    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1] < swing_lows[-2]
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def compute_all(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict[str, np.ndarray]:
    from config import (
        ATR_PERIOD,
        BB_PERIOD,
        BB_STD,
        EMA_FAST,
        EMA_SLOW,
        MACD_SIGNAL,
        RSI_PERIOD,
        SMA_FAST,
        SMA_SLOW,
    )

    macd_line, macd_signal, macd_hist = macd(close, EMA_FAST, EMA_SLOW, MACD_SIGNAL)
    bb_upper, bb_mid, bb_lower = bollinger(close, BB_PERIOD, BB_STD)
    adx_line, plus_di, minus_di = adx(high, low, close, 14)
    return {
        "sma_fast": sma(close, SMA_FAST),
        "sma_slow": sma(close, SMA_SLOW),
        "ema_fast": ema(close, EMA_FAST),
        "ema_slow": ema(close, EMA_SLOW),
        "rsi": rsi(close, RSI_PERIOD),
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "atr": atr(high, low, close, ATR_PERIOD),
        "adx": adx_line,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }
