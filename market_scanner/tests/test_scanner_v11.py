"""Offline tests for V11 causal context and live protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import V11_TRAIN_END, V11_VAL_END, active_instruments
from backtest.market_context_v11 import build_v11_context, clear_v11_cache
from backtest.report_scanner_v11 import build_specs, select_on_train
from backtest.scanner_v11 import V11Spec, backtest_spec
from models import CandleSeries
from providers.yahoo import _synthetic_series


def _series(n: int = 260) -> CandleSeries:
    raw = _synthetic_series("EURUSD", "4h", n=n)
    return CandleSeries(
        instrument="EURUSD",
        symbol="EURUSD",
        asset_class="forex",
        timeframe="4h",
        timestamps=raw.timestamps,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
    )


class V11Tests(unittest.TestCase):
    def setUp(self):
        clear_v11_cache()

    def test_live_universe_unchanged(self):
        self.assertNotIn("DXY", active_instruments())

    def test_v11_context_builds(self):
        s = _series()
        ctx = build_v11_context(s)
        self.assertEqual(len(ctx.daily_class), len(s))

    def test_backtest_runs(self):
        s = _series()
        ctx = build_v11_context(s)
        spec = V11Spec("T", "V11_TEST", "forex", "LIQ_BASE", "t", baseline="liq")
        self.assertIsInstance(backtest_spec(s, spec, ctx, None), list)

    def test_spec_count_focused(self):
        total = sum(len(v) for v in build_specs().values())
        self.assertLessEqual(total, 100)

    def test_train_prefers_liq_family(self):
        rows = [
            {"name": "L", "is_control": False, "is_exit_variant": False, "baseline": "liq", "train_n": 30, "train_expectancy": 0.002, "train_diversified": True},
            {"name": "M", "is_control": False, "is_exit_variant": False, "baseline": "mtf", "train_n": 30, "train_expectancy": 0.003, "train_diversified": True},
        ]
        self.assertEqual(select_on_train(rows)["name"], "L")

    def test_splits(self):
        self.assertEqual(V11_TRAIN_END, 0.55)
        self.assertEqual(V11_VAL_END, 0.75)


if __name__ == "__main__":
    unittest.main()
