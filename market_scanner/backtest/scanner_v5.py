"""Scanner V5 — INDEPENDENT ROBUSTNESS VALIDATION of frozen V4_S1_STOCK.

Does NOT retune parameters from V4 OOS. Does NOT modify live ORIGINAL.
Does NOT enable live or paper trading.

Frozen rule (V4_S1_STOCK):
  • Regime: HH/HL or LH/LL structure + SMA stack & slope (no ADX)
  • Entry: 20-bar Donchian close breakout aligned with regime
  • Exit: ATR 1.5 adaptive stop/trail + structure-break + max 24 bars
  • Risk: 1% equity per 1R (ATR stop distance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    BACKTEST_WARMUP_BARS,
    SMA_SLOW,
    V2_RISK_FRACTION,
    V5_FROZEN_ATR_STOP_MULT,
    V5_FROZEN_LOOKBACK,
    V5_FROZEN_MAX_HOLD,
    V5_FROZEN_SMA_SLOPE_BARS,
    V5_FROZEN_STRUCT_PIVOT,
)
from indicators import swing_structure_dir
from models import CandleSeries
from backtest.scanner_v2 import (
    V2Trade,
    _adaptive_exit,
    _make_trade,
    _precompute,
)
from backtest.scanner_v3 import _breakout_dir, _simple_regime


@dataclass(frozen=True)
class FrozenV4S1Stock:
    """Immutable freeze of V4_S1_STOCK — do not edit after V4 promotion."""

    name: str = "V4_S1_STOCK"
    require_structure: bool = True
    require_ma: bool = True
    require_adx: bool = False
    break_mode: str = "donchian20"
    lookback: int = V5_FROZEN_LOOKBACK
    atr_stop_mult: float = V5_FROZEN_ATR_STOP_MULT
    max_hold: int = V5_FROZEN_MAX_HOLD
    struct_pivot: int = V5_FROZEN_STRUCT_PIVOT
    sma_slope_bars: int = V5_FROZEN_SMA_SLOPE_BARS
    filter_name: str = "none"
    exit_policy: str = "adaptive_v2"
    risk_fraction: float = V2_RISK_FRACTION


FROZEN = FrozenV4S1Stock()

AUDIT_FINDINGS: list[dict[str, str]] = [
    {
        "id": "A1",
        "severity": "info",
        "title": "Indicators are causal",
        "detail": (
            "SMA/ATR/structure use only bars ≤ i. Confirmed swings require `pivot` "
            "bars on the right (last confirmable index i-pivot). No future OHLC in features."
        ),
    },
    {
        "id": "A2",
        "severity": "info",
        "title": "Close-based entry timing",
        "detail": (
            "Breakout uses close[i] vs max/min of highs/lows on [i-lookback, i) "
            "(excludes bar i extreme from the prior channel). Signal is known at the "
            "close of bar i; backtest fills at that close. This is standard for "
            "close-triggered systems, not next-open fill."
        ),
    },
    {
        "id": "A3",
        "severity": "warn",
        "title": "Universe selection used V4 OOS attribution",
        "detail": (
            "Entry/exit parameters of V3_S2 were frozen before V4 OOS, but the "
            "decision to trade STOCKS ONLY (V4_S1_STOCK) was made after inspecting "
            "Stage-1 OOS by asset class. V5 therefore validates on held-out stocks "
            "and treats original V4 stocks as in-sample universe evidence, not "
            "independent confirmation alone."
        ),
    },
    {
        "id": "A4",
        "severity": "warn",
        "title": "Multi-symbol equity compounding is sequential",
        "detail": (
            "Within each symbol, trades are non-overlapping. Across symbols, calendar "
            "overlaps are possible; research metrics compound trades sorted by entry "
            "time as if sequential. Concurrent multi-position portfolio risk is not "
            "fully modelled. V5 reports trade-level expectancy primarily; equity DD "
            "is a sequential stress proxy."
        ),
    },
    {
        "id": "A5",
        "severity": "info",
        "title": "Stop fills at stop price (no gap model)",
        "detail": (
            "Adaptive stops fill at the stop level when the bar range touches it. "
            "No gap-through slippage in the base model; V5 adds explicit adverse "
            "entry slippage and cost multiples as stress."
        ),
    },
    {
        "id": "A6",
        "severity": "info",
        "title": "Costs and 1% risk sizing",
        "detail": (
            "Round-trip cost by asset class is converted to R via stop distance, then "
            "equity PnL = net_R × 1%. Cost multiples rescale the cost component only."
        ),
    },
    {
        "id": "A7",
        "severity": "info",
        "title": "Survivorship",
        "detail": (
            "Universe is current large-cap/ETF Yahoo listings. Delisted names are not "
            "reconstructed; mild survivorship bias is unavoidable here and noted."
        ),
    },
    {
        "id": "A8",
        "severity": "info",
        "title": "V5 uses true chronological entry windows",
        "detail": (
            "Unlike post-hoc entry-fraction filtering of a full-sample trade list, "
            "V5 simulates each train/test/fold window with entry eligibility restricted "
            "to that window (exits may realize beyond the window)."
        ),
    },
]


def _precompute_frozen(series: CandleSeries, slope_bars: int = V5_FROZEN_SMA_SLOPE_BARS) -> dict:
    feat = _precompute(series)
    # Rebuild structure with frozen pivot (same default as V4)
    n = len(series)
    structure = np.zeros(n, dtype=np.int8)
    for i in range(n):
        structure[i] = swing_structure_dir(
            series.high, series.low, i, pivot=FROZEN.struct_pivot
        )
    feat = dict(feat)
    feat["structure"] = structure
    sma_fast = feat["sma_fast"]
    slope = np.full(n, np.nan)
    if slope_bars > 0 and n > slope_bars:
        slope[slope_bars:] = sma_fast[slope_bars:] - sma_fast[:-slope_bars]
    feat["sma_slope"] = slope
    return feat


def backtest_frozen(
    series: CandleSeries,
    *,
    start_idx: Optional[int] = None,
    end_idx_exclusive: Optional[int] = None,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    lookback: int = V5_FROZEN_LOOKBACK,
    atr_stop_mult: float = V5_FROZEN_ATR_STOP_MULT,
    max_hold: int = V5_FROZEN_MAX_HOLD,
    stage_name: str = FROZEN.name,
) -> list[V2Trade]:
    """Simulate frozen V4_S1_STOCK (optional sensitivity overrides)."""
    feat = _precompute_frozen(series)
    warmup = max(BACKTEST_WARMUP_BARS, SMA_SLOW + 5, lookback + 5)
    n = len(series)
    i = max(warmup, start_idx or warmup)
    last_start = n - max_hold if end_idx_exclusive is None else min(n - max_hold, end_idx_exclusive)
    trades: list[V2Trade] = []

    while i < last_start:
        regime = _simple_regime(
            feat,
            i,
            require_structure=FROZEN.require_structure,
            require_ma=FROZEN.require_ma,
            require_adx=FROZEN.require_adx,
        )
        if regime not in ("bullish", "bearish"):
            i += 1
            continue
        br = _breakout_dir(series, i, lookback=lookback)
        if br != regime:
            i += 1
            continue
        atr0 = feat["atr"][i]
        if np.isnan(atr0) or atr0 <= 0:
            i += 1
            continue

        raw_entry = float(series.close[i])
        slip = entry_slip_atr * float(atr0)
        if regime == "bullish":
            entry = raw_entry + slip
        else:
            entry = raw_entry - slip

        atr_for_exit = float(atr0) * (atr_stop_mult / V5_FROZEN_ATR_STOP_MULT)
        # Temporarily patch max hold by slicing feat path through _adaptive_exit:
        # _adaptive_exit uses V2_MAX_HOLD_BARS; emulate by limiting series view via early break
        exit_idx, exit_px, reason, stop_dist = _adaptive_exit(
            series, feat, i, regime, entry, atr_for_exit
        )
        if exit_idx > i + max_hold:
            exit_idx = min(i + max_hold, n - 1)
            exit_px = float(series.close[exit_idx])
            reason = "max_hold"
            stop_dist = atr_stop_mult * float(atr0)

        trades.append(
            _make_trade(
                series=series,
                stage=stage_name,
                direction=regime,
                confidence="V5",
                score=0,
                entry_idx=i,
                exit_idx=exit_idx,
                entry=entry,
                exit_px=exit_px,
                stop_dist=stop_dist,
                cost_mult=cost_mult,
                trigger="donchian20",
                regime="trending",
                exit_reason=reason,
                feature_flags={"frozen_v4_s1_stock": 1},
                atr_at_entry=float(atr0),
            )
        )
        i = exit_idx + 1
    return trades


def run_on_map(
    series_map: dict[tuple[str, str], CandleSeries],
    *,
    instruments: Optional[tuple[str, ...] | list[str]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    cost_mult: float = 1.0,
    entry_slip_atr: float = 0.0,
    lookback: int = V5_FROZEN_LOOKBACK,
    atr_stop_mult: float = V5_FROZEN_ATR_STOP_MULT,
    max_hold: int = V5_FROZEN_MAX_HOLD,
    stage_name: str = FROZEN.name,
) -> list[V2Trade]:
    allow = set(instruments) if instruments is not None else None
    trades: list[V2Trade] = []
    for (key, _), series in series_map.items():
        if allow is not None and key not in allow:
            continue
        n = len(series)
        start_idx = int(n * start_frac)
        end_idx = int(n * end_frac)
        trades.extend(
            backtest_frozen(
                series,
                start_idx=start_idx,
                end_idx_exclusive=end_idx,
                cost_mult=cost_mult,
                entry_slip_atr=entry_slip_atr,
                lookback=lookback,
                atr_stop_mult=atr_stop_mult,
                max_hold=max_hold,
                stage_name=stage_name,
            )
        )
    trades.sort(key=lambda t: (t.entry_ts, t.instrument))
    return trades


def chronological_folds(
    series_map: dict[tuple[str, str], CandleSeries],
    *,
    instruments: Optional[tuple[str, ...] | list[str]] = None,
    n_folds: int = 4,
    cost_mult: float = 1.0,
) -> list[dict]:
    out = []
    for k in range(n_folds):
        start = k / n_folds
        end = (k + 1) / n_folds
        trades = run_on_map(
            series_map,
            instruments=instruments,
            start_frac=start,
            end_frac=end,
            cost_mult=cost_mult,
        )
        out.append({"fold": k + 1, "start_frac": start, "end_frac": end, "trades": trades})
    return out


def longest_losing_streak(returns: list[float]) -> int:
    worst = 0
    cur = 0
    for r in returns:
        if r <= 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def max_drawdown(returns: list[float]) -> float:
    eq = 1.0
    peak = 1.0
    dd = 0.0
    for r in returns:
        eq *= 1.0 + r
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak if peak > 0 else 0.0)
    return float(dd)


def total_return(returns: list[float]) -> float:
    eq = 1.0
    for r in returns:
        eq *= 1.0 + r
    return eq - 1.0


def monte_carlo(
    trades: list[V2Trade],
    *,
    n_runs: int = 500,
    seed: int = 42,
) -> dict:
    """Resample trade order / bootstrap returns — stress only, not new evidence."""
    rng = np.random.default_rng(seed)
    nets = np.array([t.net_return for t in trades], dtype=float)
    if len(nets) == 0:
        return {"n_runs": 0, "note": "no trades"}
    dds = []
    tots = []
    streaks = []
    for _ in range(n_runs):
        # Bootstrap with replacement preserves marginal trade distribution
        sample = rng.choice(nets, size=len(nets), replace=True)
        # Also shuffle a without-replacement permutation of the original path
        perm = rng.permutation(nets)
        for path in (sample, perm):
            rets = path.tolist()
            dds.append(max_drawdown(rets))
            tots.append(total_return(rets))
            streaks.append(longest_losing_streak(rets))
    dds_a = np.array(dds)
    tots_a = np.array(tots)
    streaks_a = np.array(streaks)
    return {
        "n_runs": n_runs,
        "paths_per_run": 2,
        "max_drawdown": {
            "median": float(np.median(dds_a)),
            "p90": float(np.percentile(dds_a, 90)),
            "p95": float(np.percentile(dds_a, 95)),
        },
        "total_return": {
            "median": float(np.median(tots_a)),
            "p10": float(np.percentile(tots_a, 10)),
            "p5": float(np.percentile(tots_a, 5)),
        },
        "longest_losing_streak": {
            "median": float(np.median(streaks_a)),
            "p90": float(np.percentile(streaks_a, 90)),
            "p95": float(np.percentile(streaks_a, 95)),
        },
    }


def leave_out_symbols(trades: list[V2Trade], drop: set[str]) -> list[V2Trade]:
    return [t for t in trades if t.instrument not in drop]
