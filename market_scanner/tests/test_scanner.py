"""Unit tests — offline only (no network required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from indicators import bollinger, ema, macd, rsi, sma
from models import CandleSeries
from providers.yahoo import load_or_build_demo
from scanner import scan_markets, scan_opportunities, write_outputs
from scanner.levels import nearest_levels
from scanner.opportunity import evaluate_opportunity, rank_opportunities
from scanner.report import build_daily_summary
from scanner.setups import analyze_series


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


class LevelsTests(unittest.TestCase):
    def test_nearest_levels_order(self):
        close = np.linspace(100, 110, 60)
        high = close + 1
        low = close - 1
        levels = nearest_levels(close, high, low, sma20=105.0, sma50=102.0)
        self.assertIsNotNone(levels["support"])
        self.assertIsNotNone(levels["resistance"])
        self.assertLess(levels["support"], close[-1])
        self.assertGreater(levels["resistance"], close[-1])


class OpportunityTests(unittest.TestCase):
    def test_evaluate_returns_card(self):
        series = load_or_build_demo("EURUSD", "1d")
        opp = evaluate_opportunity(series, "Euro / US Dollar")
        self.assertEqual(opp.instrument, "EURUSD")
        self.assertIn(opp.confidence, ("HIGH", "MEDIUM", "LOW", "NO STRONG SETUP"))
        self.assertIn(opp.direction, ("bullish", "bearish", "neutral"))
        self.assertGreaterEqual(opp.score, 0)
        self.assertLessEqual(opp.score, 100)
        self.assertTrue(opp.reason)
        self.assertTrue(opp.sma_relation)
        self.assertTrue(opp.macd_condition)

    def test_flat_market_is_no_strong_setup(self):
        n = 80
        close = np.full(n, 100.0)
        # tiny noise so indicators compute but stay mixed/flat
        rng = np.random.default_rng(1)
        close = close + rng.normal(0, 0.01, n)
        high = close + 0.05
        low = close - 0.05
        series = CandleSeries(
            instrument="FLAT",
            symbol="FLAT",
            asset_class="forex",
            timeframe="1d",
            timestamps=np.arange(n, dtype=np.int64) * 86400,
            open=close.copy(),
            high=high,
            low=low,
            close=close,
            volume=np.ones(n),
        )
        opp = evaluate_opportunity(series, "Flat")
        self.assertEqual(opp.confidence, "NO STRONG SETUP")

    def test_rank_puts_high_first(self):
        a = evaluate_opportunity(load_or_build_demo("BTCUSD", "1d"), "Bitcoin")
        b = evaluate_opportunity(load_or_build_demo("EURUSD", "1d"), "Euro")
        c = evaluate_opportunity(load_or_build_demo("XAUUSD", "1d"), "Gold")
        ranked = rank_opportunities([a, b, c])
        scores = [o.score for o in ranked]
        # confidence order respected: sort key ensures non-increasing confidence rank
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NO STRONG SETUP": 3}
        ranks = [order[o.confidence] for o in ranked]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(scores), 3)


class DemoScanTests(unittest.TestCase):
    def test_demo_fetch(self):
        series = load_or_build_demo("EURUSD", "1d")
        self.assertGreaterEqual(len(series), 50)
        self.assertEqual(series.instrument, "EURUSD")

    def test_analyze_returns_list(self):
        series = load_or_build_demo("BTCUSD", "1d")
        alerts = analyze_series(series, "Bitcoin")
        self.assertIsInstance(alerts, list)

    def test_default_universe_excludes_crypto(self):
        from config import active_instruments

        active = active_instruments()
        self.assertTrue(all(v["asset_class"] != "crypto" for v in active.values()))
        self.assertIn("EURUSD", active)
        self.assertIn("XAUUSD", active)
        self.assertNotIn("BTCUSD", active)

        opps, snapshots, errors = scan_opportunities(None, ["1d"], demo=True)
        self.assertEqual(errors, [])
        self.assertTrue(snapshots)
        self.assertTrue(all(s["asset_class"] != "crypto" for s in snapshots))
        self.assertTrue(all(o.asset_class != "crypto" for o in opps))

    def test_full_demo_scan_all_classes(self):
        opps, snapshots, errors = scan_opportunities(
            ["EURUSD", "BTCUSD", "XAUUSD"],
            ["1d"],
            demo=True,
        )
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(len(opps), 3)
        self.assertEqual(errors, [])
        classes = {o.asset_class for o in opps}
        self.assertEqual(classes, {"forex", "crypto", "commodity"})
        path = write_outputs([], snapshots, errors, opps, mode_label="demo")
        self.assertTrue(path.exists())
        summary = Path("output/daily_summary.txt")
        self.assertTrue(summary.exists())
        text = summary.read_text(encoding="utf-8")
        self.assertIn("DAILY MARKET SCANNER", text)
        self.assertIn("NO STRONG SETUP", text)
        self.assertIn("FOREX", text.upper())  # by-market section uses FOREX:

    def test_legacy_scan_markets(self):
        alerts, snapshots, errors = scan_markets(
            ["EURUSD", "BTCUSD", "XAUUSD"],
            ["1d"],
            demo=True,
        )
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(errors, [])

    def test_beginner_summary_mentions_all_classes(self):
        opps, _, _ = scan_opportunities(
            ["EURUSD", "BTCUSD", "XAUUSD"],
            ["1d"],
            demo=True,
        )
        text = build_daily_summary(opps, mode_label="demo test")
        self.assertIn("EURUSD", text)
        self.assertIn("BTCUSD", text)
        self.assertIn("XAUUSD", text)
        self.assertIn("BY MARKET TYPE", text)


if __name__ == "__main__":
    unittest.main()
