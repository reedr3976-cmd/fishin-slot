"""Historical backtest metrics (analysis only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from config import MIN_SIGNALS_FOR_CONCLUSION


@dataclass
class TradeResult:
    instrument: str
    asset_class: str
    timeframe: str
    confidence: str
    direction: str
    score: int
    entry_idx: int
    exit_idx: int
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    gross_return: float
    cost: float
    net_return: float
    win: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricBag:
    label: str
    signals: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: Optional[float] = None
    avg_return: Optional[float] = None
    avg_winner: Optional[float] = None
    avg_loser: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_net_return: float = 0.0
    reliable: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _max_drawdown(returns: list[float]) -> float:
    """Max drawdown of an equity curve that compounds trade net returns sequentially."""
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return float(max_dd)


def summarize_trades(label: str, trades: Iterable[TradeResult]) -> MetricBag:
    trades_list = list(trades)
    bag = MetricBag(label=label, signals=len(trades_list))
    if not trades_list:
        bag.note = "No signals in this bucket."
        bag.reliable = False
        return bag

    nets = [t.net_return for t in trades_list]
    wins = [t.net_return for t in trades_list if t.net_return > 0]
    losses = [t.net_return for t in trades_list if t.net_return <= 0]
    bag.wins = len(wins)
    bag.losses = len(losses)
    bag.win_rate = len(wins) / len(trades_list)
    bag.avg_return = float(np.mean(nets))
    bag.avg_winner = float(np.mean(wins)) if wins else None
    bag.avg_loser = float(np.mean(losses)) if losses else None
    gross_win = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(-np.sum(losses)) if losses else 0.0
    if gross_loss > 0:
        bag.profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        bag.profit_factor = float("inf")
    else:
        bag.profit_factor = 0.0
    bag.max_drawdown = _max_drawdown(nets)
    bag.total_net_return = float(np.sum(nets))
    bag.reliable = len(trades_list) >= MIN_SIGNALS_FOR_CONCLUSION
    if not bag.reliable:
        bag.note = (
            f"Only {len(trades_list)} signals (need ≥ {MIN_SIGNALS_FOR_CONCLUSION} "
            "for a reliable conclusion)."
        )
    else:
        bag.note = "Sample size meets the minimum threshold."
    return bag


def group_metrics(trades: list[TradeResult]) -> dict[str, Any]:
    """Build metric bags by confidence, asset class, and timeframe."""
    by_conf = {
        conf: summarize_trades(conf, [t for t in trades if t.confidence == conf])
        for conf in ("HIGH", "MEDIUM", "LOW")
    }
    classes = sorted({t.asset_class for t in trades})
    by_class = {
        cls: summarize_trades(cls, [t for t in trades if t.asset_class == cls])
        for cls in classes
    }
    # Also confidence × class
    by_class_conf: dict[str, dict[str, MetricBag]] = {}
    for cls in classes:
        by_class_conf[cls] = {
            conf: summarize_trades(
                f"{cls}/{conf}",
                [t for t in trades if t.asset_class == cls and t.confidence == conf],
            )
            for conf in ("HIGH", "MEDIUM", "LOW")
        }

    tfs = sorted({t.timeframe for t in trades})
    by_tf = {
        tf: summarize_trades(tf, [t for t in trades if t.timeframe == tf]) for tf in tfs
    }
    by_tf_conf: dict[str, dict[str, MetricBag]] = {}
    for tf in tfs:
        by_tf_conf[tf] = {
            conf: summarize_trades(
                f"{tf}/{conf}",
                [t for t in trades if t.timeframe == tf and t.confidence == conf],
            )
            for conf in ("HIGH", "MEDIUM", "LOW")
        }

    overall = summarize_trades("ALL", trades)
    return {
        "overall": overall,
        "by_confidence": by_conf,
        "by_asset_class": by_class,
        "by_asset_class_confidence": by_class_conf,
        "by_timeframe": by_tf,
        "by_timeframe_confidence": by_tf_conf,
    }
