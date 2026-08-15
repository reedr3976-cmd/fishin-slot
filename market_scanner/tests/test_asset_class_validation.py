"""Smoke tests for asset-class validation helpers."""

from __future__ import annotations

import unittest

from backtest.report_asset_class_validation import build_asset_class_validation_report


class ReportSmokeTests(unittest.TestCase):
    def test_report_builds_minimal(self):
        empty = {
            "signals": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "cumulative_return": None,
            "avg_winner": None,
            "avg_loser": None,
            "profit_factor": None,
            "max_drawdown": None,
            "reliable": False,
        }
        cls = {c: dict(empty) for c in ("commodity", "forex", "stocks")}
        result = {
            "mode": "demo",
            "control": "test",
            "train_fraction": 0.7,
            "instruments": ["EURUSD"],
            "bars_loaded": 1,
            "baseline_test": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
            "baseline_train": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
            "why_strong_vs_weak": {
                "strong": {"n": 0},
                "weak": {"n": 0},
            },
            "period_thirds": {
                "P1_early": {"all": empty, "by_asset_class": cls, "commodity_symbols": {}},
                "P2_mid": {"all": empty, "by_asset_class": cls, "commodity_symbols": {}},
                "P3_late": {"all": empty, "by_asset_class": cls, "commodity_symbols": {}},
            },
            "test_early_vs_late": {
                "early": {"all": empty, "by_asset_class": cls},
                "late": {"all": empty, "by_asset_class": cls},
            },
            "stops_test": {
                "fixed_hold": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
                "stop_1.0atr": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
                "stop_1.5atr": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
                "stop_2.0atr": {"all": empty, "by_asset_class": cls, "by_symbol": {}},
            },
            "commodity_cost_stress": {
                "base_10bps": {
                    "round_trip_cost": 0.001,
                    "commodity": empty,
                    "USOIL": empty,
                    "XAGUSD": empty,
                    "XAUUSD": empty,
                },
                "stress_20bps": {
                    "round_trip_cost": 0.002,
                    "commodity": empty,
                    "USOIL": empty,
                    "XAGUSD": empty,
                    "XAUUSD": empty,
                },
                "stress_30bps": {
                    "round_trip_cost": 0.003,
                    "commodity": empty,
                    "USOIL": empty,
                    "XAGUSD": empty,
                    "XAUUSD": empty,
                },
                "stress_50bps": {
                    "round_trip_cost": 0.005,
                    "commodity": empty,
                    "USOIL": empty,
                    "XAGUSD": empty,
                    "XAUUSD": empty,
                },
            },
            "symbol_exclusion_study": {
                "rule": "x",
                "exclude_negative_train": {
                    "symbols": [],
                    "test_baseline": empty,
                    "test_kept": empty,
                    "test_removed": empty,
                },
                "exclude_worst3_train": {
                    "symbols": [],
                    "train_ranks": [],
                    "test_baseline": empty,
                    "test_kept": empty,
                    "test_removed": empty,
                },
            },
            "min_signals": 30,
        }
        text = build_asset_class_validation_report(result)
        self.assertIn("KEEP AS-IS", text)
        self.assertIn("REJECT", text)


if __name__ == "__main__":
    unittest.main()
