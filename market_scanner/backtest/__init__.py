"""Backtest package — historical evaluation of scanner confidence ratings."""

from backtest.engine import BacktestRun, backtest_series, run_backtest, run_backtest_with_metrics
from backtest.metrics import MetricBag, TradeResult, group_metrics, summarize_trades

__all__ = [
    "BacktestRun",
    "MetricBag",
    "TradeResult",
    "backtest_series",
    "group_metrics",
    "run_backtest",
    "run_backtest_with_metrics",
    "summarize_trades",
]
