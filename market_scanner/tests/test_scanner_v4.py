"""Offline tests for Scanner V4 research modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.scanner_v4 import (
    V4_S2_SWING_CLOSE,
    V4_S2_SWING_CONFIRM,
    backtest_v4,
    with_filter,
)
from backtest.report_scanner_v4 import select_on_train, stage1_justifies_stage2
from providers.yahoo import _synthetic_series


class V4EngineTests(unittest.TestCase):
    def test_swing_modes_run(self):
        s = _synthetic_series("SPY", "4h", n=400)
        for cand in (V4_S2_SWING_CLOSE, V4_S2_SWING_CONFIRM, with_filter(V4_S2_SWING_CLOSE, "min_break_atr")):
            # Force stock class
            from models import CandleSeries
            import numpy as np

            s2 = CandleSeries(
                instrument=s.instrument,
                symbol=s.symbol,
                asset_class="stock",
                timeframe=s.timeframe,
                timestamps=s.timestamps,
                open=s.open,
                high=s.high,
                low=s.low,
                close=s.close,
                volume=s.volume,
            )
            trades = backtest_v4(s2, cand)
            self.assertIsInstance(trades, list)

    def test_train_selection_ignores_test(self):
        results = [
            {"name": "A", "train_n": 25, "train_expectancy": 0.001, "oos_gate": {"test_expectancy": -0.01}},
            {"name": "B", "train_n": 25, "train_expectancy": -0.001, "oos_gate": {"test_expectancy": 0.05}},
        ]
        pick = select_on_train(results)
        self.assertEqual(pick["name"], "A")


class V4GateTests(unittest.TestCase):
    def test_stage1_gate_requires_sc_better(self):
        def block(exp, n=40, folds=1, one=False):
            return {
                "oos_gate": {
                    "test_expectancy": exp,
                    "test_n": n,
                    "folds_positive_count": folds,
                    "concentration": {"dependent_on_one": one},
                }
            }

        attr = {
            "forex": block(-0.001),
            "stock": block(0.0005),
            "commodity": block(0.0004),
            "stock_commodity": block(0.00045),
        }
        ok, _ = stage1_justifies_stage2(attr)
        self.assertTrue(ok)

        attr2 = {
            "forex": block(0.001),
            "stock": block(-0.001),
            "commodity": block(-0.001),
            "stock_commodity": block(-0.001),
        }
        ok2, _ = stage1_justifies_stage2(attr2)
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
