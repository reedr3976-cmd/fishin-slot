"""Unit tests for confluence policies A/B/C/D (analysis only)."""

from __future__ import annotations

import unittest

from backtest.confluence import (
    apply_policy,
    has_directional_sr,
    has_macd_strong,
    passes_policy,
)
from backtest.metrics import TradeResult
from backtest.report_confluence import build_confluence_report


def _trade(
    conf: str,
    direction: str = "bullish",
    *,
    macd_strong: int = 0,
    near_support: int = 0,
    near_resistance: int = 0,
    net: float = 0.01,
) -> TradeResult:
    return TradeResult(
        instrument="EURUSD",
        asset_class="forex",
        timeframe="1d",
        confidence=conf,
        direction=direction,
        score=45,
        entry_idx=1,
        exit_idx=6,
        entry_ts=1,
        exit_ts=2,
        entry_price=1.0,
        exit_price=1.0 + net,
        gross_return=net,
        cost=0.0,
        net_return=net,
        win=net > 0,
        feature_flags={
            "macd_strong": macd_strong,
            "near_support": near_support,
            "near_resistance": near_resistance,
        },
        rules_name="original",
    )


class ConfluencePolicyTests(unittest.TestCase):
    def test_macd_and_sr_helpers(self):
        self.assertTrue(has_macd_strong(_trade("MEDIUM", macd_strong=1)))
        self.assertTrue(
            has_directional_sr(_trade("MEDIUM", "bullish", near_support=1))
        )
        self.assertTrue(
            has_directional_sr(_trade("MEDIUM", "bearish", near_resistance=1))
        )
        self.assertFalse(
            has_directional_sr(_trade("MEDIUM", "bullish", near_resistance=1))
        )

    def test_policies(self):
        ok = _trade("MEDIUM", macd_strong=1, near_support=1)
        macd_only = _trade("MEDIUM", macd_strong=1, near_support=0)
        sr_only = _trade("MEDIUM", macd_strong=0, near_support=1)
        neither = _trade("MEDIUM", macd_strong=0, near_support=0)
        low = _trade("LOW", macd_strong=0, near_support=0)

        self.assertTrue(passes_policy(neither, "A"))
        self.assertTrue(passes_policy(macd_only, "B"))
        self.assertFalse(passes_policy(sr_only, "B"))
        self.assertTrue(passes_policy(sr_only, "C"))
        self.assertFalse(passes_policy(macd_only, "C"))
        self.assertTrue(passes_policy(ok, "D"))
        self.assertFalse(passes_policy(macd_only, "D"))
        self.assertTrue(passes_policy(low, "B"))
        self.assertTrue(passes_policy(low, "D"))

        trades = [ok, macd_only, sr_only, neither, low]
        self.assertEqual(len(apply_policy(trades, "A")), 5)
        self.assertEqual(len(apply_policy(trades, "B")), 3)  # ok, macd_only, low
        self.assertEqual(len(apply_policy(trades, "C")), 3)  # ok, sr_only, low
        self.assertEqual(len(apply_policy(trades, "D")), 2)  # ok, low

    def test_report_builds(self):
        from backtest.metrics import summarize_trades

        empty = summarize_trades("x", [])
        pol = {
            "overall": empty,
            "medium_high": empty,
            "by_confidence": {"HIGH": empty, "MEDIUM": empty, "LOW": empty},
            "by_asset_class": {},
            "by_asset_class_mh": {},
            "by_instrument_mh": {},
        }
        br = {
            "removed_n": 0,
            "kept_mh_n": 0,
            "removed_summary": empty,
            "kept_summary": empty,
            "removed_sum_net": 0.0,
            "removed_by_asset_class": {},
            "removed_by_instrument": [],
            "leave_one_instrument_out_kept_mh": [],
            "concentration": {
                "worst_1_share_of_removed_sum": None,
                "worst_3_share_of_removed_sum": None,
                "best_1_share_of_removed_sum": None,
                "best_3_share_of_removed_sum": None,
            },
            "sample_permits_d": False,
        }
        result = {
            "mode": "demo",
            "train_fraction": 0.7,
            "instruments": ["EURUSD"],
            "timeframes": ["1d"],
            "bars_loaded": 1,
            "note": "test",
            "feature_rates_test_mh": {
                "macd_strong": 0,
                "directional_sr": 0,
                "both": 0,
                "mh_total": 0,
            },
            "removal_counts": {
                "A_original": {"removed_mh": 0, "kept_mh": 0, "kept_total": 0},
                "B_require_macd_strong": {
                    "removed_mh": 0,
                    "kept_mh": 0,
                    "kept_total": 0,
                },
                "C_require_directional_sr": {
                    "removed_mh": 0,
                    "kept_mh": 0,
                    "kept_total": 0,
                },
                "D_macd_strong_and_sr": {
                    "removed_mh": 0,
                    "kept_mh": 0,
                    "kept_total": 0,
                },
            },
            "policies_test": {
                "A_original": pol,
                "B_require_macd_strong": pol,
                "C_require_directional_sr": pol,
                "D_macd_strong_and_sr": pol,
            },
            "breadth_test": {
                "A_original": None,
                "B_require_macd_strong": br,
                "C_require_directional_sr": br,
                "D_macd_strong_and_sr": br,
            },
            "min_signals_for_conclusion": 30,
        }
        text = build_confluence_report(result)
        self.assertIn("CONFLUENCE STUDY", text)
        self.assertIn("KEEP ORIGINAL", text)


if __name__ == "__main__":
    unittest.main()
