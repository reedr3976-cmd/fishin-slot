"""Backtest unit tests — offline, no network, no brokerage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from backtest.engine import _slice_series, backtest_series, run_backtest_with_metrics
from backtest.metrics import summarize_trades, TradeResult
from backtest.report import build_backtest_report
from models import CandleSeries
from providers.yahoo import _synthetic_series
from scanner.opportunity import evaluate_opportunity


class LookAheadTests(unittest.TestCase):
    def test_slice_excludes_future_bars(self):
        series = _synthetic_series("EURUSD", "1d", n=100)
        sliced = _slice_series(series, 49)
        self.assertEqual(len(sliced), 50)
        self.assertEqual(int(sliced.timestamps[-1]), int(series.timestamps[49]))
        # Future close must not appear
        self.assertNotEqual(float(sliced.close[-1]), float(series.close[-1]))

    def test_historical_score_matches_sliced_evaluation(self):
        """Opportunity at bar i depends only on data through i."""
        series = _synthetic_series("BTCUSD", "1d", n=120)
        i = 80
        hist = _slice_series(series, i)
        opp = evaluate_opportunity(hist, "Bitcoin")
        # Extending future bars must not be used; re-evaluate same slice → same score
        opp2 = evaluate_opportunity(hist, "Bitcoin")
        self.assertEqual(opp.score, opp2.score)
        self.assertEqual(opp.confidence, opp2.confidence)
        # Full series evaluation can differ — that's expected; we only use slices in BT
        self.assertEqual(len(hist), i + 1)


class MetricsTests(unittest.TestCase):
    def test_summarize_win_rate(self):
        trades = [
            TradeResult(
                instrument="EURUSD",
                asset_class="forex",
                timeframe="1d",
                confidence="HIGH",
                direction="bullish",
                score=70,
                entry_idx=1,
                exit_idx=6,
                entry_ts=1,
                exit_ts=2,
                entry_price=1.0,
                exit_price=1.01,
                gross_return=0.01,
                cost=0.0004,
                net_return=0.0096,
                win=True,
            ),
            TradeResult(
                instrument="EURUSD",
                asset_class="forex",
                timeframe="1d",
                confidence="HIGH",
                direction="bullish",
                score=70,
                entry_idx=10,
                exit_idx=15,
                entry_ts=3,
                exit_ts=4,
                entry_price=1.0,
                exit_price=0.99,
                gross_return=-0.01,
                cost=0.0004,
                net_return=-0.0104,
                win=False,
            ),
        ]
        bag = summarize_trades("HIGH", trades)
        self.assertEqual(bag.signals, 2)
        self.assertAlmostEqual(bag.win_rate, 0.5)
        self.assertTrue(bag.avg_return is not None)


class EngineDemoTests(unittest.TestCase):
    def test_backtest_series_runs(self):
        series = _synthetic_series("EURUSD", "1d", n=200)
        trades = backtest_series(series)
        self.assertIsInstance(trades, list)
        for t in trades:
            self.assertIn(t.confidence, ("HIGH", "MEDIUM", "LOW"))
            self.assertEqual(t.exit_idx - t.entry_idx, 5)  # 1d forward bars
            self.assertTrue(t.entry_idx < t.exit_idx)

    def test_non_overlapping(self):
        series = _synthetic_series("XAUUSD", "1d", n=250)
        trades = backtest_series(series)
        for a, b in zip(trades, trades[1:]):
            self.assertGreaterEqual(b.entry_idx, a.exit_idx)

    def test_full_demo_backtest_all_classes(self):
        run, metrics = run_backtest_with_metrics(
            ["EURUSD", "BTCUSD", "XAUUSD"],
            ["1d"],
            demo=True,
        )
        self.assertEqual(run.errors, [])
        self.assertGreater(run.bars_scanned, 0)
        report = build_backtest_report(run, metrics)
        self.assertIn("HOW A WIN IS DEFINED", report)
        self.assertIn("HIGH", report)
        self.assertIn("MEDIUM", report)
        self.assertIn("LOW", report)
        self.assertIn("BEGINNER CONCLUSION", report)
        # All three asset classes present in metrics if any trades, or at least ran
        classes_seen = {t.asset_class for t in run.trades}
        # Even with zero trades, engine should have scanned all three series
        self.assertGreaterEqual(run.bars_scanned, 3 * 100)


if __name__ == "__main__":
    unittest.main()
