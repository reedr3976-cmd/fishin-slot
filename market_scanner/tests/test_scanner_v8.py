"""Offline tests for Scanner V8 nested design and live protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    INSTRUMENTS,
    V8_STOCK_DEV,
    V8_STOCK_FINAL_INST,
    V8_TRAIN_END,
    V8_VAL_END,
    active_instruments,
)
from backtest.scanner_v8 import FAMILIES, backtest_family
from backtest.report_scanner_v8 import select_on_train
from models import CandleSeries
from providers.yahoo import _synthetic_series


class V8FamilyTests(unittest.TestCase):
    def test_all_families_run(self):
        raw = _synthetic_series("SPY", "4h", n=400)
        s = CandleSeries(
            instrument="SPY",
            symbol="SPY",
            asset_class="stock",
            timeframe="4h",
            timestamps=raw.timestamps,
            open=raw.open,
            high=raw.high,
            low=raw.low,
            close=raw.close,
            volume=raw.volume,
        )
        daily = _synthetic_series("SPY", "1d", n=200)
        d = CandleSeries(
            instrument="SPY",
            symbol="SPY",
            asset_class="stock",
            timeframe="1d",
            timestamps=daily.timestamps,
            open=daily.open,
            high=daily.high,
            low=daily.low,
            close=daily.close,
            volume=daily.volume,
        )
        for fam in FAMILIES:
            trades = backtest_family(s, fam, daily=d, spy=s)
            self.assertIsInstance(trades, list)

    def test_nested_splits_ordered(self):
        self.assertLess(0.0, V8_TRAIN_END)
        self.assertLess(V8_TRAIN_END, V8_VAL_END)
        self.assertLess(V8_VAL_END, 1.0)

    def test_final_inst_disjoint_from_dev(self):
        self.assertFalse(set(V8_STOCK_FINAL_INST) & set(V8_STOCK_DEV))

    def test_train_selection_ignores_final(self):
        rows = [
            {
                "name": "A",
                "is_baseline": False,
                "train_n": 40,
                "train_expectancy": 0.002,
                "train_diversified": True,
                "gate": {"final_inst_expectancy": -1},
            },
            {
                "name": "B",
                "is_baseline": False,
                "train_n": 40,
                "train_expectancy": 0.001,
                "train_diversified": True,
                "gate": {"final_inst_expectancy": 1},
            },
            {
                "name": "BASE",
                "is_baseline": True,
                "train_n": 100,
                "train_expectancy": 0.01,
                "train_diversified": True,
            },
        ]
        self.assertEqual(select_on_train(rows)["name"], "A")

    def test_research_commodities_excluded_from_live(self):
        active = active_instruments()
        for key in ("NATGAS", "COPPER", "CORN"):
            self.assertTrue(INSTRUMENTS[key].get("research_only"))
            self.assertNotIn(key, active)


if __name__ == "__main__":
    unittest.main()
