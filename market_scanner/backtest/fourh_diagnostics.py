"""4H trending diagnostics for the ORIGINAL scanner (analysis only).

Focus: forex + commodities + stocks (no crypto). Live defaults unchanged.

Stages:
  1) Baseline MEDIUM / HIGH on 4H (all actionable + trending-only)
  2) Winner vs loser characteristics (TRAIN, then describe TEST)
  3) Per asset class / symbol breakdown
  4) Separate OOS experiments:
       - entry quality gates (not MACD/S/R hard filters)
       - stop-loss ATR multiples
       - take-profit ATR multiples
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config import (
    MIN_FEATURE_HITS_FOR_EDGE,
    MIN_SIGNALS_FOR_CONCLUSION,
    STUDY_ASSET_CLASSES,
    VALIDATION_TRAIN_FRACTION,
    study_instruments,
)
from backtest.engine import load_series_map
from backtest.fourh_exits import (
    ENTRY_STOP_POLICIES,
    ENTRY_TP_POLICIES,
    FIXED_HOLD,
    ExitPolicy,
    backtest_series_exits,
)
from backtest.metrics import TradeResult, summarize_trades
from backtest.validation import FEATURE_KEYS
from scanner.scoring import ORIGINAL_RULES


def _flag(t: TradeResult, name: str) -> bool:
    return int((t.feature_flags or {}).get(name, 0) or 0) == 1


def is_trending(t: TradeResult) -> bool:
    """Genuine trend stack: price + SMA20/SMA50 aligned."""
    return _flag(t, "sma_stack")


def _mh(trades: list[TradeResult]) -> list[TradeResult]:
    return [t for t in trades if t.confidence in ("HIGH", "MEDIUM")]


def _run_map(
    series_map,
    *,
    start_frac: float,
    end_frac: float,
    exit_policy: ExitPolicy = FIXED_HOLD,
    require_trending: bool = False,
) -> list[TradeResult]:
    trades: list[TradeResult] = []
    for (_, _tf), series in series_map.items():
        n = len(series)
        start_idx = int(n * start_frac)
        end_idx = int(n * end_frac)
        trades.extend(
            backtest_series_exits(
                series,
                ORIGINAL_RULES,
                exit_policy=exit_policy,
                start_idx=start_idx,
                end_idx_exclusive=end_idx,
                require_trending=require_trending,
            )
        )
    return trades


def _bag(trades: list[TradeResult], label: str) -> dict[str, Any]:
    return {
        "all": summarize_trades(label, trades),
        "HIGH": summarize_trades(f"{label}/HIGH", [t for t in trades if t.confidence == "HIGH"]),
        "MEDIUM": summarize_trades(
            f"{label}/MEDIUM", [t for t in trades if t.confidence == "MEDIUM"]
        ),
        "by_asset_class": {
            cls: summarize_trades(
                f"{label}/{cls}", [t for t in trades if t.asset_class == cls]
            )
            for cls in sorted({t.asset_class for t in trades})
        },
        "by_symbol": {
            inst: summarize_trades(
                f"{label}/{inst}", [t for t in trades if t.instrument == inst]
            )
            for inst in sorted({t.instrument for t in trades})
        },
    }


def _winner_loser_profile(trades: list[TradeResult]) -> dict[str, Any]:
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    profile: dict[str, Any] = {
        "n_wins": len(wins),
        "n_losses": len(losses),
        "avg_score_wins": float(np.mean([t.score for t in wins])) if wins else None,
        "avg_score_losses": float(np.mean([t.score for t in losses])) if losses else None,
        "avg_atr_pct_wins": None,
        "avg_atr_pct_losses": None,
        "features": {},
    }
    def atr_pct(t: TradeResult) -> Optional[float]:
        if t.atr_at_entry is None or t.entry_price <= 0:
            return None
        return t.atr_at_entry / t.entry_price

    w_atr = [atr_pct(t) for t in wins]
    l_atr = [atr_pct(t) for t in losses]
    w_atr_f = [x for x in w_atr if x is not None]
    l_atr_f = [x for x in l_atr if x is not None]
    profile["avg_atr_pct_wins"] = float(np.mean(w_atr_f)) if w_atr_f else None
    profile["avg_atr_pct_losses"] = float(np.mean(l_atr_f)) if l_atr_f else None

    for feat in FEATURE_KEYS:
        w_rate = (
            sum(1 for t in wins if _flag(t, feat)) / len(wins) if wins else None
        )
        l_rate = (
            sum(1 for t in losses if _flag(t, feat)) / len(losses) if losses else None
        )
        lift = None if w_rate is None or l_rate is None else w_rate - l_rate
        profile["features"][feat] = {
            "win_rate_with_flag": w_rate,
            "loss_rate_with_flag": l_rate,
            "delta_win_minus_loss": lift,
            "hits_total": sum(1 for t in trades if _flag(t, feat)),
        }
    return profile


def _entry_gate(trade: TradeResult, gate: str) -> bool:
    """Entry-quality gates that are NOT the rejected MACD / directional S/R filters."""
    if gate == "none":
        return True
    if gate == "trending_only":
        return is_trending(trade)
    if gate == "avoid_high_atr":
        return not _flag(trade, "high_atr")
    if gate == "trending_avoid_high_atr":
        return is_trending(trade) and not _flag(trade, "high_atr")
    if gate == "score_ge_45":
        return trade.score >= 45
    if gate == "trending_score_ge_45":
        return is_trending(trade) and trade.score >= 45
    raise ValueError(gate)


def _apply_entry_gate(trades: list[TradeResult], gate: str) -> list[TradeResult]:
    # LOW never gated in entry experiments — only MH filtered; keep LOW out of MH bags
    return [t for t in trades if t.confidence not in ("HIGH", "MEDIUM") or _entry_gate(t, gate)]


def _avg_r(trades: list[TradeResult]) -> Optional[float]:
    rs = [t.r_multiple for t in trades if t.r_multiple is not None]
    return float(np.mean(rs)) if rs else None


def _enrich_bag(trades: list[TradeResult], label: str) -> dict[str, Any]:
    base = _bag(trades, label)
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    base["exit_reasons"] = reasons
    base["avg_r_multiple"] = _avg_r(trades)
    return base


def run_fourh_diagnostics(
    *,
    demo: bool = False,
    instruments=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    keys = (
        list(instruments)
        if instruments is not None
        else list(study_instruments().keys())
    )
    # Guard: never include crypto in this study
    keys = [
        k
        for k in keys
        if study_instruments().get(k, {}).get("asset_class") != "crypto"
        and k in study_instruments()
    ]
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=demo)
    mode = "demo" if demo else "public_historical"

    # --- Baseline fixed-hold (all MH + trending MH) ---
    train_all = _run_map(
        series_map, start_frac=0.0, end_frac=train_fraction, require_trending=False
    )
    test_all = _run_map(
        series_map, start_frac=train_fraction, end_frac=1.0, require_trending=False
    )
    train_trend = _run_map(
        series_map, start_frac=0.0, end_frac=train_fraction, require_trending=True
    )
    test_trend = _run_map(
        series_map, start_frac=train_fraction, end_frac=1.0, require_trending=True
    )

    train_mh = _mh(train_all)
    test_mh = _mh(test_all)
    train_tmh = _mh(train_trend)
    test_tmh = _mh(test_trend)

    # Winner/loser on TRAIN trending MH (propose from train only)
    wl_train = _winner_loser_profile(train_tmh)
    wl_test = _winner_loser_profile(test_tmh)

    # Rank features by |delta| on train with enough hits
    ranked = sorted(
        (
            (f, info)
            for f, info in wl_train["features"].items()
            if info["hits_total"] >= MIN_FEATURE_HITS_FOR_EDGE
            and info["delta_win_minus_loss"] is not None
        ),
        key=lambda x: abs(x[1]["delta_win_minus_loss"]),
        reverse=True,
    )

    # --- Entry quality OOS (on fixed-hold all signals, then filter MH) ---
    entry_gates = [
        "none",
        "trending_only",
        "avoid_high_atr",
        "trending_avoid_high_atr",
        "score_ge_45",
        "trending_score_ge_45",
    ]
    entry_oos: dict[str, Any] = {}
    for gate in entry_gates:
        kept = [t for t in test_mh if _entry_gate(t, gate)]
        # For trending_* gates, also report from trending path consistency
        entry_oos[gate] = {
            "removed_mh": len(test_mh) - len(kept),
            "metrics": _enrich_bag(kept, f"entry/{gate}"),
        }

    # --- Stop-loss OOS (trending entries only; ORIGINAL scoring) ---
    stop_oos: dict[str, Any] = {}
    for pol in ENTRY_STOP_POLICIES:
        trades = _mh(
            _run_map(
                series_map,
                start_frac=train_fraction,
                end_frac=1.0,
                exit_policy=pol,
                require_trending=True,
            )
        )
        stop_oos[pol.name] = _enrich_bag(trades, f"stop/{pol.name}")

    # --- Take-profit OOS (trending entries only) ---
    tp_oos: dict[str, Any] = {}
    for pol in ENTRY_TP_POLICIES:
        trades = _mh(
            _run_map(
                series_map,
                start_frac=train_fraction,
                end_frac=1.0,
                exit_policy=pol,
                require_trending=True,
            )
        )
        tp_oos[pol.name] = _enrich_bag(trades, f"tp/{pol.name}")

    return {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": sorted({k for k, _ in series_map}),
        "asset_classes": list(STUDY_ASSET_CLASSES),
        "timeframe": "4h",
        "errors": errors,
        "rules": "original",
        "live_unchanged": True,
        "note": (
            "Analysis only. Live scanner still forex+commodity defaults and not "
            "forced to 4H. Crypto excluded. MACD/S/R hard filters NOT tested as "
            "enable candidates (rejected earlier)."
        ),
        "baseline": {
            "train_all_mh": _enrich_bag(train_mh, "train_all_mh"),
            "test_all_mh": _enrich_bag(test_mh, "test_all_mh"),
            "train_trending_mh": _enrich_bag(train_tmh, "train_trending_mh"),
            "test_trending_mh": _enrich_bag(test_tmh, "test_trending_mh"),
        },
        "winner_loser_train_trending_mh": wl_train,
        "winner_loser_test_trending_mh": wl_test,
        "top_train_feature_deltas": [
            {"feature": f, **info} for f, info in ranked[:8]
        ],
        "entry_quality_oos": entry_oos,
        "stop_loss_oos": stop_oos,
        "take_profit_oos": tp_oos,
        "min_signals": MIN_SIGNALS_FOR_CONCLUSION,
    }
