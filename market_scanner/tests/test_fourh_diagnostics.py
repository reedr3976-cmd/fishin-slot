"""Unit tests for 4H exit policies and trending gate (analysis only)."""

from __future__ import annotations

import unittest

import numpy as np

from backtest.fourh_exits import (
    FIXED_HOLD,
    STOP_1_5_ATR,
    TP_2_0_ATR,
    simulate_exit,
)
from backtest.fourh_diagnostics import _entry_gate, is_trending
from backtest.metrics import TradeResult
from models import CandleSeries


def _series(n: int = 20) -> CandleSeries:
    ts = np.arange(n, dtype=np.int64) * 14400
    close = np.linspace(100.0, 110.0, n)
    high = close + 1.0
    low = close - 1.0
    return CandleSeries(
        instrument="SPY",
        symbol="SPY",
        asset_class="stocks",
        timeframe="4h",
        timestamps=ts,
        open=close.copy(),
        high=high,
        low=low,
        close=close,
        volume=np.ones(n),
    )


def _trade(**flags) -> TradeResult:
    return TradeResult(
        instrument="EURUSD",
        asset_class="forex",
        timeframe="4h",
        confidence="MEDIUM",
        direction="bullish",
        score=45,
        entry_idx=1,
        exit_idx=5,
        entry_ts=1,
        exit_ts=2,
        entry_price=1.0,
        exit_price=1.01,
        gross_return=0.01,
        cost=0.0,
        net_return=0.01,
        win=True,
        feature_flags=dict(flags),
    )


class FourHExitTests(unittest.TestCase):
    def test_fixed_hold(self):
        s = _series()
        idx, px, reason, r = simulate_exit(s, 5, "bullish", 1.0, FIXED_HOLD)
        self.assertEqual(reason, "fixed_hold")
        self.assertEqual(idx, 9)  # 5+4
        self.assertIsNone(r)

    def test_stop_hits(self):
        s = _series()
        # Force a dip on bar 6
        s.low[6] = 90.0
        idx, px, reason, r = simulate_exit(s, 5, "bullish", atr=2.0, policy=STOP_1_5_ATR)
        self.assertEqual(reason, "stop")
        self.assertEqual(idx, 6)
        self.assertLess(px, s.close[5])

    def test_tp_hits(self):
        s = _series()
        s.high[7] = 200.0
        idx, px, reason, r = simulate_exit(s, 5, "bullish", atr=2.0, policy=TP_2_0_ATR)
        self.assertEqual(reason, "target")
        self.assertEqual(idx, 7)


class EntryGateTests(unittest.TestCase):
    def test_trending_and_gates(self):
        t = _trade(sma_stack=1, high_atr=0)
        self.assertTrue(is_trending(t))
        self.assertTrue(_entry_gate(t, "trending_only"))
        self.assertTrue(_entry_gate(t, "trending_avoid_high_atr"))
        t2 = _trade(sma_stack=1, high_atr=1)
        self.assertFalse(_entry_gate(t2, "trending_avoid_high_atr"))
        self.assertFalse(_entry_gate(_trade(sma_stack=0), "trending_only"))


if __name__ == "__main__":
    unittest.main()
