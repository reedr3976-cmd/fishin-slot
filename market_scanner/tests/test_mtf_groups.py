"""Tests for MTF group classification and policies A/B/C."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.metrics import TradeResult
from backtest.mtf_groups import apply_policy, classify_daily_mh_signals, run_mtf_group_study
from providers.yahoo import _synthetic_series
from scanner.scoring import ORIGINAL_RULES


def _t(group: str, conf: str = "MEDIUM", ret: float = 0.01) -> TradeResult:
    return TradeResult(
        instrument="EURUSD",
        asset_class="forex",
        timeframe="1d",
        confidence=conf,
        direction="bullish",
        score=50,
        entry_idx=0,
        exit_idx=5,
        entry_ts=0,
        exit_ts=1,
        entry_price=1.0,
        exit_price=1.0 + ret,
        gross_return=ret,
        cost=0.0,
        net_return=ret,
        win=ret > 0,
        mtf_status=group,
    )


class PolicyTests(unittest.TestCase):
    def test_policy_a_keeps_all_mh(self):
        trades = [_t("AGREE"), _t("DISAGREE"), _t("WEEKLY_UNKNOWN"), _t("LOW", "LOW")]
        kept = apply_policy(trades, "A")
        self.assertEqual(len(kept), 4)

    def test_policy_b_drops_disagree_only(self):
        trades = [_t("AGREE"), _t("DISAGREE"), _t("WEEKLY_UNKNOWN"), _t("LOW", "LOW")]
        kept = apply_policy(trades, "B")
        statuses = {t.mtf_status for t in kept if t.confidence != "LOW"}
        self.assertNotIn("DISAGREE", statuses)
        self.assertIn("AGREE", statuses)
        self.assertIn("WEEKLY_UNKNOWN", statuses)

    def test_policy_c_keeps_agree_only(self):
        trades = [_t("AGREE"), _t("DISAGREE"), _t("WEEKLY_UNKNOWN"), _t("LOW", "LOW")]
        kept = apply_policy(trades, "C")
        mh = [t for t in kept if t.confidence in ("HIGH", "MEDIUM")]
        self.assertTrue(all(t.mtf_status == "AGREE" for t in mh))
        self.assertTrue(any(t.confidence == "LOW" for t in kept))


class ClassifyTests(unittest.TestCase):
    def test_classify_runs(self):
        d1 = _synthetic_series("EURUSD", "1d", n=220)
        wk = _synthetic_series("EURUSD", "1wk", n=100)
        trades, counts = classify_daily_mh_signals(d1, wk, ORIGINAL_RULES)
        self.assertIsInstance(trades, list)
        self.assertEqual(
            set(counts), {"AGREE", "DISAGREE", "WEEKLY_UNKNOWN", "LOW"}
        )


class StudyDemoTests(unittest.TestCase):
    def test_group_study_demo(self):
        from backtest.report_mtf_groups import build_mtf_group_report

        result = run_mtf_group_study(
            demo=True,
            instruments=["EURUSD", "XAUUSD"],
            train_fraction=0.7,
        )
        report = build_mtf_group_report(result)
        self.assertIn("AGREE", report)
        self.assertIn("DISAGREE", report)
        self.assertIn("WEEKLY_UNKNOWN", report)
        self.assertIn("A_original", report)
        self.assertIn("B_suppress_disagree_only", report)
        self.assertIn("C_require_agree", report)


if __name__ == "__main__":
    unittest.main()
