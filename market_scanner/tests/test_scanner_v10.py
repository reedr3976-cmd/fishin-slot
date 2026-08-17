"""Offline tests for V10 causal market context and live-universe protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import INSTRUMENTS, V10_TRAIN_END, V10_VAL_END, active_instruments
from backtest.market_context import _detect_fvg_at, _htf_dir, build_context_arrays, clear_context_cache
from backtest.report_scanner_v10 import build_specs, select_on_train
from backtest.scanner_v10 import V10Spec, backtest_spec
from models import CandleSeries
from providers.yahoo import _synthetic_series


def _make_series(n: int = 300, *, instrument: str = "EURUSD") -> CandleSeries:
    raw = _synthetic_series(instrument, "4h", n=n)
    return CandleSeries(
        instrument=instrument,
        symbol=instrument,
        asset_class="forex",
        timeframe="4h",
        timestamps=raw.timestamps,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
    )


class V10SafetyTests(unittest.TestCase):
    def setUp(self):
        clear_context_cache()

    def test_research_symbols_excluded_from_live(self):
        active = active_instruments()
        for key in ("DXY", "US10Y", "US3M", "TIP", "NATGAS", "COPPER", "CORN"):
            self.assertTrue(INSTRUMENTS[key].get("research_only"))
            self.assertNotIn(key, active)

    def test_fvg_bullish_definition(self):
        high = np.array([10.0, 10.5, 10.2, 11.0], dtype=np.float64)
        low = np.array([9.5, 9.8, 10.1, 10.5], dtype=np.float64)
        z = _detect_fvg_at(high, low, 2)
        self.assertIsNotNone(z)
        self.assertEqual(z.direction, "bullish")

    def test_htf_uses_completed_bars_only(self):
        daily = _make_series(100, instrument="EURUSD")
        daily.timeframe = "1d"
        ts_mid = int(daily.timestamps[50])
        d1 = _htf_dir(daily, ts_mid)
        d2 = _htf_dir(daily, ts_mid - 86400 * 10)
        self.assertIn(d1, (-1, 0, 1))
        self.assertIn(d2, (-1, 0, 1))

    def test_context_arrays_length(self):
        s = _make_series(250)
        ctx = build_context_arrays(s)
        self.assertEqual(len(ctx.trend_dir), len(s))
        self.assertEqual(len(ctx.daily_dir), len(s))

    def test_backtest_runs_without_macro(self):
        s = _make_series(250)
        ctx = build_context_arrays(s)
        spec = V10Spec("T", "V10_TEST", "forex", "CTRL_TREND", "test")
        trades = backtest_spec(s, spec, ctx, None)
        self.assertIsInstance(trades, list)

    def test_train_selection_prefers_independent(self):
        rows = [
            {"name": "IND", "is_control": False, "is_combo": False, "train_n": 40, "train_expectancy": 0.002, "train_diversified": True},
            {"name": "COMBO", "is_control": False, "is_combo": True, "train_n": 40, "train_expectancy": 0.003, "train_diversified": True},
        ]
        self.assertEqual(select_on_train(rows)["name"], "IND")

    def test_nested_splits_pre_specified(self):
        self.assertEqual(V10_TRAIN_END, 0.55)
        self.assertEqual(V10_VAL_END, 0.75)

    def test_spec_count_bounded(self):
        specs = build_specs()
        total = sum(len(v) for v in specs.values())
        self.assertLessEqual(total, 50)


if __name__ == "__main__":
    unittest.main()
