"""Offline tests for Scanner V3 research modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.scanner_v3 import (
    V3_S1_BREAKOUT_ADAPT,
    V3_S2_STRUCT_MA,
    V3_X_ATR_TRAIL,
    V3_X_FIXED,
    backtest_v3_entry,
    backtest_v3_exit_only,
    collect_original_entries,
)
from backtest.report_scanner_v3 import oos_gate, verdict_v3
from providers.yahoo import _synthetic_series


class V3EngineTests(unittest.TestCase):
    def test_breakout_and_regime_run(self):
        s = _synthetic_series("EURUSD", "4h", n=400)
        for stage in (V3_S1_BREAKOUT_ADAPT, V3_S2_STRUCT_MA):
            trades = backtest_v3_entry(s, stage)
            self.assertIsInstance(trades, list)
            for t in trades:
                self.assertGreater(t.exit_idx, t.entry_idx)
                self.assertEqual(t.trigger, "breakout")

    def test_exit_only_uses_original_entries(self):
        s = _synthetic_series("USOIL", "4h", n=400)
        entries = collect_original_entries(s)
        fixed = backtest_v3_exit_only(s, V3_X_FIXED, entries=entries)
        trail = backtest_v3_exit_only(s, V3_X_ATR_TRAIL, entries=entries)
        self.assertIsInstance(fixed, list)
        self.assertIsInstance(trail, list)
        # Same entry universe cadence (subset allowed if trail overlaps)
        if fixed and trail:
            self.assertEqual(fixed[0].entry_idx, trail[0].entry_idx)

    def test_no_overlap_exit_only(self):
        s = _synthetic_series("XAUUSD", "4h", n=400)
        trades = backtest_v3_exit_only(s, V3_X_ATR_TRAIL)
        for a, b in zip(trades, trades[1:]):
            self.assertGreaterEqual(b.entry_idx, a.exit_idx + 1)


class V3GateTests(unittest.TestCase):
    def test_verdict_fail_when_empty(self):
        code, label, promo = verdict_v3({}, {}, {})
        self.assertEqual(code, "FAIL")
        self.assertIsNone(promo)

    def test_oos_gate_requires_positive_exp(self):
        from backtest.scanner_v2 import V2Trade

        def t(net):
            return V2Trade(
                instrument="EURUSD",
                asset_class="forex",
                timeframe="4h",
                confidence="V3",
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
                net_r=net / 0.01,
            )

        trades = [t(-0.001) for _ in range(40)]
        g = oos_gate(test=trades, test_2x=trades, fold_exps=[-1, -1, -1, -1], by_symbol={})
        self.assertFalse(g["positive_oos_expectancy"])
        self.assertFalse(g["all_pass"])


if __name__ == "__main__":
    unittest.main()
