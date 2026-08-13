"""OHLCV candle container (numpy-based, no pandas required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CandleSeries:
    instrument: str
    symbol: str
    asset_class: str
    timeframe: str
    timestamps: np.ndarray  # unix seconds, int64
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(len(self.close))

    @property
    def last_close(self) -> float:
        return float(self.close[-1])

    def to_summary(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
            "bars": len(self),
            "last_close": self.last_close,
            "last_ts": int(self.timestamps[-1]) if len(self) else None,
        }
