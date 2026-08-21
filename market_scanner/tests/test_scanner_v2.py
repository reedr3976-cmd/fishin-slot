"""Offline unit tests for Scanner V2 research modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from backtest.scanner_v2 import (
    STAGE_ORIGINAL,
    STAGE_S1,
    STAGE_S2,
    STAGE_S3,
    _regime_at,
    _precompute,
    backtest_v2_stage,
)
from backtest.report_scanner_v2 import (
    evaluate_robustness,
    select_candidate_on_train,
    verdict_from_checks,
)
from indicators import adx, swing_structure_dir
from providers.yahoo import _synthetic_series


class IndicatorTests(unittest.TestCase):
    def test_adx_causal_length(self):
        s = _synthetic_series("EURUSD", "4h", n=120)
        a, p, m = adx(s.high, s.low, s.close, 14)
        self.assertEqual(len(a), len(s))
        self.assertTrue(np.isnan(a[10]))

    def test_structure_no_lookahead_pivot(self):
        s = _synthetic_series("EURUSD", "4h", n=80)
        # Structure at i must equal structure computed on sliced series end=i
        for i in (40, 50, 60):
            full = swing_structure_dir(s.high, s.low, i)
            sliced = swing_structure_dir(s.high[: i + 1], s.low[: i + 1], i)
            self.assertEqual(full, sliced)


class V2EngineTests(unittest.TestCase):
    def test_stages_run_offline(self):
        s = _synthetic_series("EURUSD", "4h", n=400)
        for stage in (STAGE_ORIGINAL, STAGE_S1, STAGE_S2, STAGE_S3):
            trades = backtest_v2_stage(s, stage)
            self.assertIsInstance(trades, list)
            for t in trades:
                self.assertEqual(t.exit_idx > t.entry_idx, True)
                self.assertGreater(t.stop_dist_pct, 0)

    def test_no_overlap_trades(self):
        s = _synthetic_series("USOIL", "4h", n=400)
        trades = backtest_v2_stage(s, STAGE_S2)
        for a, b in zip(trades, trades[1:]):
            self.assertGreaterEqual(b.entry_idx, a.exit_idx + 1)

    def test_regime_returns_tuple(self):
        s = _synthetic_series("XAUUSD", "4h", n=200)
        feat = _precompute(s)
        d, r = _regime_at(feat, 100)
        self.assertIn(r, ("trending", "ranging"))
        self.assertTrue(d in (None, "bullish", "bearish"))


class RobustnessGateTests(unittest.TestCase):
    def test_select_ignores_original_name_preference(self):
        # Empty V2 → falls back somehow; with positive S2 train picks S2
        from backtest.scanner_v2 import V2Trade

        def _t(net, stage="V2_S2_ADAPTIVE"):
            return V2Trade(
                instrument="EURUSD",
                asset_class="forex",
                timeframe="4h",
                confidence="V2",
                direction="bullish",
                score=0,
                entry_idx=1,
                exit_idx=2,
                entry_ts=1,
                exit_ts=2,
                entry_price=1.0,
                exit_price=1.0,
                gross_return=net,
                cost=0.0,
                net_return=net,
                win=net > 0,
                stage=stage,
                net_r=net / 0.01,
            )

        train = {
            "ORIGINAL": [_t(0.001, "ORIGINAL") for _ in range(25)],
            "V2_S1_REGIME_HOLD": [_t(-0.001) for _ in range(25)],
            "V2_S2_ADAPTIVE": [_t(0.002) for _ in range(25)],
            "V2_S3_DUAL_TRIG": [_t(0.0005) for _ in range(25)],
        }
        self.assertEqual(select_candidate_on_train(train), "V2_S2_ADAPTIVE")

    def test_verdict_failed_when_weak(self):
        weak = {
            "positive_test_expectancy": False,
            "folds_positive_ge_3_of_4": False,
            "positive_after_2x_costs": False,
            "adequate_trade_count": False,
            "acceptable_max_drawdown": True,
            "not_single_symbol_dependent": False,
            "all_pass": False,
            "test_expectancy": -0.001,
            "folds_positive_count": 1,
        }
        orig = {**weak, "test_expectancy": 0.0001}
        code, label = verdict_from_checks(orig, "V2_S2_ADAPTIVE", weak, {})
        self.assertIn(code, ("ORIGINAL_BETTER", "V2_FAILED"))


if __name__ == "__main__":
    unittest.main()
