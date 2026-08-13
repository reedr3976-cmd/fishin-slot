"""Unit tests — offline only (no network required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from indicators import bollinger, ema, macd, rsi, sma
from providers.yahoo import load_or_build_demo
from scanner.setups import analyze_series
from scanner import scan_markets, write_outputs


class IndicatorTests(unittest.TestCase):
    def test_sma_known(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = sma(x, 3)
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[1]))
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_moves_toward_price(self):
        x = np.linspace(1, 20, 20)
        out = ema(x, 5)
        self.assertFalse(np.isnan(out[-1]))
        self.assertGreater(out[-1], out[4])

    def test_rsi_bounds(self):
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, 100))
        out = rsi(close, 14)
        valid = out[~np.isnan(out)]
        self.assertTrue(np.all(valid >= 0) and np.all(valid <= 100))

    def test_macd_shapes(self):
        close = np.linspace(100, 120, 80) + np.sin(np.linspace(0, 8, 80))
        line, signal, hist = macd(close)
        self.assertEqual(len(line), len(close))
        self.assertFalse(np.isnan(line[-1]))
        self.assertFalse(np.isnan(signal[-1]))

    def test_bollinger_order(self):
        close = np.linspace(50, 60, 40)
        u, m, l = bollinger(close, 20, 2)
        self.assertGreater(u[-1], m[-1])
        self.assertGreater(m[-1], l[-1])


class DemoScanTests(unittest.TestCase):
    def test_demo_fetch(self):
        series = load_or_build_demo("EURUSD", "1d")
        self.assertGreaterEqual(len(series), 50)
        self.assertEqual(series.instrument, "EURUSD")

    def test_analyze_returns_list(self):
        series = load_or_build_demo("BTCUSD", "1d")
        alerts = analyze_series(series, "Bitcoin")
        self.assertIsInstance(alerts, list)

    def test_full_demo_scan(self):
        alerts, snapshots, errors = scan_markets(
            ["EURUSD", "BTCUSD", "XAUUSD"],
            ["1d"],
            demo=True,
        )
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(errors, [])
        path = write_outputs(alerts, snapshots, errors)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 50)


if __name__ == "__main__":
    unittest.main()
