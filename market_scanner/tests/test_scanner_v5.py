"""Offline tests for Scanner V5 validation modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.scanner_v5 import FROZEN, AUDIT_FINDINGS, backtest_frozen, monte_carlo
from models import CandleSeries
from providers.yahoo import _synthetic_series


class V5FrozenTests(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(FROZEN.lookback, 20)
        self.assertEqual(FROZEN.atr_stop_mult, 1.5)
        self.assertFalse(FROZEN.require_adx)
        self.assertEqual(FROZEN.filter_name, "none")

    def test_audit_has_universe_warning(self):
        ids = {f["id"] for f in AUDIT_FINDINGS}
        self.assertIn("A3", ids)

    def test_backtest_runs(self):
        s = _synthetic_series("AAPL", "4h", n=400)
        s2 = CandleSeries(
            instrument="AAPL",
            symbol="AAPL",
            asset_class="stock",
            timeframe="4h",
            timestamps=s.timestamps,
            open=s.open,
            high=s.high,
            low=s.low,
            close=s.close,
            volume=s.volume,
        )
        trades = backtest_frozen(s2)
        self.assertIsInstance(trades, list)

    def test_monte_carlo_empty(self):
        mc = monte_carlo([])
        self.assertEqual(mc["n_runs"], 0)


if __name__ == "__main__":
    unittest.main()
