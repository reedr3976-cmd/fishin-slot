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
    }
