"""Tests for multi-timeframe confirmation filter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scanner.mtf_filter import apply_mtf_filter, directions_agree
from scanner.opportunity import Opportunity
from backtest.mtf_backtest import backtest_series_mtf
from providers.yahoo import _synthetic_series
from scanner.scoring import ORIGINAL_RULES


def _opp(instrument, timeframe, direction, confidence, score=50):
    return Opportunity(
        instrument=instrument,
        name=instrument,
        asset_class="forex",
        timeframe=timeframe,
        price=1.0,
        direction=direction,
        confidence=confidence,
        score=score,
        reason="test",
        rsi=50.0,
        sma20=1.0,
        sma50=1.0,
        sma_relation="x",
        macd_condition="y",
        atr=0.01,
        atr_pct=1.0,
        volatility_note="z",
        support=0.9,
        resistance=1.1,
        support_2=None,
        resistance_2=None,
    )


class MTFFilterUnitTests(unittest.TestCase):
    def test_agree_keeps_medium(self):
        opps = [
            _opp("EURUSD", "1d", "bullish", "MEDIUM", 50),
            _opp("EURUSD", "1wk", "bullish", "LOW", 30),
        ]
        out, stats = apply_mtf_filter(opps, enabled=True)
        kept = [o for o in out if o.timeframe == "1d"][0]
        self.assertEqual(kept.confidence, "MEDIUM")
        self.assertEqual(stats["kept_agree"], 1)

    def test_disagree_suppresses_medium(self):
        opps = [
            _opp("EURUSD", "1d", "bullish", "MEDIUM", 50),
            _opp("EURUSD", "1wk", "bearish", "MEDIUM", 45),
        ]
        out, stats = apply_mtf_filter(opps, enabled=True)
        d1 = [o for o in out if o.timeframe == "1d"][0]
        self.assertEqual(d1.confidence, "NO STRONG SETUP")
        self.assertEqual(stats["suppressed_disagree"], 2)  # both MEDIUM/HIGH gated

    def test_low_untouched(self):
        opps = [
            _opp("EURUSD", "1d", "bullish", "LOW", 30),
            _opp("EURUSD", "1wk", "bearish", "LOW", 30),
        ]
        out, stats = apply_mtf_filter(opps, enabled=True)
        self.assertTrue(all(o.confidence == "LOW" for o in out))
        self.assertEqual(stats["low_untouched"], 2)

    def test_disabled_noop(self):
        opps = [_opp("EURUSD", "1d", "bullish", "MEDIUM", 50)]
        out, stats = apply_mtf_filter(opps, enabled=False)
        self.assertEqual(out[0].confidence, "MEDIUM")
        self.assertEqual(stats["medium_high_total"], 0)

    def test_directions_agree(self):
        self.assertTrue(directions_agree("bullish", "bullish"))
        self.assertFalse(directions_agree("bullish", "bearish"))
        self.assertFalse(directions_agree("bullish", "neutral"))


class MTFBacktestTests(unittest.TestCase):
    def test_mtf_backtest_runs(self):
        d1 = _synthetic_series("EURUSD", "1d", n=250)
        wk = _synthetic_series("EURUSD", "1wk", n=120)
        trades, stats = backtest_series_mtf(d1, wk, ORIGINAL_RULES)
        self.assertIsInstance(trades, list)
        self.assertIn("suppressed_disagree", stats)
        for t in trades:
            if t.confidence in ("HIGH", "MEDIUM"):
                self.assertEqual(t.mtf_status, "agreed")


if __name__ == "__main__":
    unittest.main()
