"""Unit tests for ATR regime policy A/B (analysis only)."""

from __future__ import annotations

import unittest

from backtest.atr_regime import apply_atr_policy, _is_high_atr
from backtest.metrics import TradeResult
from backtest.report_atr_regime import build_atr_regime_report


def _trade(
    conf: str,
    *,
    high_atr: int = 0,
    net: float = 0.01,
    asset_class: str = "forex",
    instrument: str = "EURUSD",
) -> TradeResult:
    return TradeResult(
        instrument=instrument,
        asset_class=asset_class,
        timeframe="1d",
        confidence=conf,
        direction="bullish",
        score=50,
        entry_idx=10,
        exit_idx=15,
        entry_ts=1,
        exit_ts=2,
        entry_price=1.0,
        exit_price=1.0 + net,
        gross_return=net,
        cost=0.0,
        net_return=net,
        win=net > 0,
        feature_flags={"high_atr": high_atr},
        rules_name="original",
    )


class AtrRegimePolicyTests(unittest.TestCase):
    def test_is_high_atr(self):
        self.assertTrue(_is_high_atr(_trade("MEDIUM", high_atr=1)))
        self.assertFalse(_is_high_atr(_trade("MEDIUM", high_atr=0)))

    def test_policy_a_keeps_all(self):
        trades = [
            _trade("HIGH", high_atr=1),
            _trade("MEDIUM", high_atr=0),
            _trade("LOW", high_atr=1),
        ]
        self.assertEqual(len(apply_atr_policy(trades, "A")), 3)

    def test_policy_b_suppresses_mh_high_atr_only(self):
        trades = [
            _trade("HIGH", high_atr=1, net=-0.02),
            _trade("MEDIUM", high_atr=0, net=0.01),
            _trade("MEDIUM", high_atr=1, net=-0.03),
            _trade("LOW", high_atr=1, net=0.005),
        ]
        kept = apply_atr_policy(trades, "B")
        self.assertEqual(len(kept), 2)
        self.assertEqual({t.confidence for t in kept}, {"MEDIUM", "LOW"})
        self.assertFalse(any(_is_high_atr(t) and t.confidence != "LOW" for t in kept))

    def test_report_builds(self):
        from backtest.metrics import summarize_trades

        empty = summarize_trades("x", [])
        result = {
            "mode": "demo",
            "train_fraction": 0.7,
            "instruments": ["EURUSD"],
            "timeframes": ["1d"],
            "bars_loaded": 10,
            "note": "test",
            "counts": {
                "test_total": 0,
                "test_mh_high_atr_removed": 0,
                "test_a_kept": 0,
                "test_b_kept": 0,
            },
            "policies_test": {
                "A_original": {
                    "overall": empty,
                    "medium_high": empty,
                    "by_confidence": {"HIGH": empty, "MEDIUM": empty, "LOW": empty},
                    "by_asset_class": {},
                    "by_asset_class_mh": {},
                },
                "B_suppress_high_atr_mh": {
                    "overall": empty,
                    "medium_high": empty,
                    "by_confidence": {"HIGH": empty, "MEDIUM": empty, "LOW": empty},
                    "by_asset_class": {},
                    "by_asset_class_mh": {},
                },
            },
            "breadth_test": {
                "removed_mh_count": 0,
                "removed_mh_summary": empty,
                "removed_sum_net": 0.0,
                "removed_by_asset_class": {},
                "removed_by_instrument": [],
                "concentration": {
                    "worst_1_share_of_removed_sum": None,
                    "worst_3_share_of_removed_sum": None,
                    "best_1_share_of_removed_sum": None,
                    "best_3_share_of_removed_sum": None,
                },
                "kept_vs_removed_mh": {
                    "kept_n": 0,
                    "removed_n": 0,
                    "kept_avg": None,
                    "removed_avg": None,
                },
            },
        }
        text = build_atr_regime_report(result)
        self.assertIn("ATR / VOLATILITY REGIME STUDY", text)
        self.assertIn("A — Original", text)


if __name__ == "__main__":
    unittest.main()
