"""Validation / backtest unit tests — offline, no brokerage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from backtest.engine import _slice_series, backtest_series, run_backtest_on_map
from backtest.metrics import TradeResult, summarize_trades
from backtest.validation import (
    analyze_feature_edges,
    make_recommendation,
    propose_revised_rules,
    refine_thresholds_on_train_scores,
)
from models import CandleSeries
from providers.yahoo import _synthetic_series
from scanner.opportunity import evaluate_opportunity
from scanner.scoring import ORIGINAL_RULES, ScoringRules, explain_why_high_is_rare


class ScoringExplainTests(unittest.TestCase):
    def test_explain_mentions_threshold(self):
        text = explain_why_high_is_rare()
        self.assertIn("HIGH", text)
        self.assertIn(str(ORIGINAL_RULES.score_high), text)


class LookAheadTests(unittest.TestCase):
    def test_slice_excludes_future_bars(self):
        series = _synthetic_series("EURUSD", "1d", n=100)
        sliced = _slice_series(series, 49)
        self.assertEqual(len(sliced), 50)
        self.assertEqual(int(sliced.timestamps[-1]), int(series.timestamps[49]))

    def test_train_window_does_not_start_in_test(self):
        series = _synthetic_series("BTCUSD", "1d", n=200)
        n = len(series)
        split = int(n * 0.7)
        train_trades = backtest_series(
            series, ORIGINAL_RULES, start_idx=0, end_idx_exclusive=split
        )
        test_trades = backtest_series(
            series, ORIGINAL_RULES, start_idx=split, end_idx_exclusive=n
        )
        if train_trades:
            self.assertTrue(all(t.entry_idx < split for t in train_trades))
        if test_trades:
            self.assertTrue(all(t.entry_idx >= split for t in test_trades))


class FeatureEdgeTests(unittest.TestCase):
    def test_feature_edge_counts(self):
        trades = []
        for i in range(40):
            trades.append(
                TradeResult(
                    instrument="EURUSD",
                    asset_class="forex",
                    timeframe="1d",
                    confidence="LOW",
                    direction="bullish",
                    score=30,
                    entry_idx=i,
                    exit_idx=i + 5,
                    entry_ts=i,
                    exit_ts=i + 5,
                    entry_price=1.0,
                    exit_price=1.01 if i % 2 == 0 else 0.99,
                    gross_return=0.01 if i % 2 == 0 else -0.01,
                    cost=0.0004,
                    net_return=0.0096 if i % 2 == 0 else -0.0104,
                    win=i % 2 == 0,
                    feature_flags={"sma_stack": 1 if i < 30 else 0, "macd_mild": 0},
                )
            )
        edges = {e.feature: e for e in analyze_feature_edges(trades)}
        self.assertEqual(edges["sma_stack"].hits, 30)
        self.assertIsNotNone(edges["sma_stack"].lift)


class RevisedProposalTests(unittest.TestCase):
    def test_propose_from_train_only_changes_name(self):
        series = _synthetic_series("EURUSD", "1d", n=250)
        trades = backtest_series(series, ORIGINAL_RULES)
        edges = analyze_feature_edges(trades)
        revised, rationale = propose_revised_rules(trades, edges)
        self.assertEqual(revised.name, "revised_candidate")
        self.assertIsInstance(rationale, list)
        self.assertGreaterEqual(revised.high_min_factors, 2)

    def test_refine_thresholds_ordering(self):
        rules = ScoringRules(name="tmp", score_high=60, score_medium=40, score_low=25)
        scores = list(range(20, 80))
        out = refine_thresholds_on_train_scores(rules, scores)
        self.assertGreater(out.score_high, out.score_medium)
        self.assertGreaterEqual(out.score_medium, out.score_low)


class RecommendationTests(unittest.TestCase):
    def test_need_more_data_when_high_tiny(self):
        def empty_bag(n=0):
            return summarize_trades("x", [])

        def bag_with(n, avg):
            trades = [
                TradeResult(
                    instrument="X",
                    asset_class="forex",
                    timeframe="1d",
                    confidence="LOW",
                    direction="bullish",
                    score=30,
                    entry_idx=0,
                    exit_idx=1,
                    entry_ts=0,
                    exit_ts=1,
                    entry_price=1,
                    exit_price=1,
                    gross_return=avg,
                    cost=0,
                    net_return=avg,
                    win=avg > 0,
                )
                for _ in range(n)
            ]
            return summarize_trades("x", trades)

        result = {
            "metrics": {
                "original_test": {
                    "by_confidence": {
                        "HIGH": bag_with(2, 0.01),
                        "MEDIUM": bag_with(40, 0.0),
                        "LOW": bag_with(40, 0.0),
                    },
                    "overall": bag_with(82, 0.0),
                },
                "revised_test": {
                    "by_confidence": {
                        "HIGH": bag_with(3, 0.02),
                        "MEDIUM": bag_with(40, 0.0),
                        "LOW": bag_with(40, -0.01),
                    },
                    "overall": bag_with(83, 0.0),
                },
            }
        }
        rec = make_recommendation(result)
        self.assertEqual(rec["decision"], "NEED MORE DATA")


class MetricsMedianTests(unittest.TestCase):
    def test_median_and_cumulative(self):
        trades = [
            TradeResult(
                instrument="EURUSD",
                asset_class="forex",
                timeframe="1d",
                confidence="LOW",
                direction="bullish",
                score=30,
                entry_idx=0,
                exit_idx=1,
                entry_ts=0,
                exit_ts=1,
                entry_price=1,
                exit_price=1,
                gross_return=r,
                cost=0,
                net_return=r,
                win=r > 0,
            )
            for r in (0.10, -0.05, 0.00)
        ]
        bag = summarize_trades("t", trades)
        self.assertAlmostEqual(bag.median_return, 0.0)
        self.assertIsNotNone(bag.cumulative_return)


class EndToEndDemoValidation(unittest.TestCase):
    def test_validation_pipeline_demo(self):
        from backtest.validation import run_validation
        from backtest.report_validation import build_validation_report

        result = run_validation(
            demo=True,
            instruments=["EURUSD", "GBPUSD", "XAUUSD"],
            timeframes=["1d"],
            train_fraction=0.7,
        )
        self.assertIn(result["recommendation"]["decision"], {
            "KEEP ORIGINAL",
            "ADOPT REVISED",
            "NEED MORE DATA",
        })
        report = build_validation_report(result)
        self.assertIn("WHY HIGH WAS SO RARE", report)
        self.assertIn("OUT-OF-SAMPLE", report)
        self.assertIn(result["recommendation"]["decision"], report)


if __name__ == "__main__":
    unittest.main()
