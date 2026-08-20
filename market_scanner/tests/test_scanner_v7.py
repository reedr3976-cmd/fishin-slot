"""Offline tests for Scanner V7 families and live-universe protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import INSTRUMENTS, active_instruments
from backtest.scanner_v7 import FAMILIES, backtest_family
from backtest.report_scanner_v7 import select_on_train
from models import CandleSeries
from providers.yahoo import _synthetic_series


class V7FamilyTests(unittest.TestCase):
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
            trades = backtest_family(s, fam, daily=d)
            self.assertIsInstance(trades, list)

    def test_train_selection_prefers_diversified(self):
        rows = [
            {
                "name": "concentrated",
                "train_n": 40,
                "train_expectancy": 0.002,
                "train_diversified": False,
            },
            {
                "name": "diversified",
                "train_n": 40,
                "train_expectancy": 0.001,
                "train_diversified": True,
            },
        ]
        self.assertEqual(select_on_train(rows)["name"], "diversified")

    def test_train_selection_ignores_oos(self):
        rows = [
            {
                "name": "A",
                "train_n": 30,
                "train_expectancy": 0.001,
                "train_diversified": True,
                "gate": {"test_expectancy": -1},
            },
            {
                "name": "B",
                "train_n": 30,
                "train_expectancy": -0.001,
                "train_diversified": True,
                "gate": {"test_expectancy": 1},
            },
        ]
        self.assertEqual(select_on_train(rows)["name"], "A")

    def test_research_commodities_excluded_from_live(self):
        for key in ("NATGAS", "COPPER", "CORN"):
            self.assertIn(key, INSTRUMENTS)
            self.assertTrue(INSTRUMENTS[key].get("research_only"))
        active = active_instruments()
        for key in ("NATGAS", "COPPER", "CORN"):
            self.assertNotIn(key, active)
        self.assertIn("XAUUSD", active)
        self.assertIn("USOIL", active)


if __name__ == "__main__":
    unittest.main()
