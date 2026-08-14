"""Backtest package — historical evaluation & OOS validation of confidence ratings."""

from backtest.engine import (
    BacktestRun,
    backtest_series,
    load_series_map,
    run_backtest,
    run_backtest_on_map,
    run_backtest_with_metrics,
)
from backtest.metrics import MetricBag, TradeResult, group_metrics, summarize_trades
from backtest.validation import (
    analyze_feature_edges,
    propose_revised_rules,
    run_validation,
)

__all__ = [
    "BacktestRun",
    "MetricBag",
    "TradeResult",
    "analyze_feature_edges",
    "backtest_series",
    "group_metrics",
    "load_series_map",
    "propose_revised_rules",
    "run_backtest",
    "run_backtest_on_map",
    "run_backtest_with_metrics",
    "run_validation",
    "summarize_trades",
]
