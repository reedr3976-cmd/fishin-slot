"""4H trending diagnostics for the ORIGINAL scanner (analysis only).

Focus: forex + commodities + stocks (no crypto). Live defaults unchanged.
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
    collect_entries,
    realize_entries,
)
from backtest.metrics import TradeResult, summarize_trades
from backtest.validation import FEATURE_KEYS
from scanner.scoring import ORIGINAL_RULES


def _flag(t: TradeResult, name: str) -> bool:
    return int((t.feature_flags or {}).get(name, 0) or 0) == 1


def is_trending(t: TradeResult) -> bool:
    return _flag(t, "sma_stack")


def _mh(trades: list[TradeResult]) -> list[TradeResult]:
    return [t for t in trades if t.confidence in ("HIGH", "MEDIUM")]


def _collect_map(
    series_map,
    *,
    start_frac: float,
    end_frac: float,
    require_trending: bool = False,
):
    """Return { (instrument,tf): (series, entries) }."""
    out = {}
    for key, series in series_map.items():
        n = len(series)
        start_idx = int(n * start_frac)
        end_idx = int(n * end_frac)
        entries = collect_entries(
            series,
            ORIGINAL_RULES,
            start_idx=start_idx,
            end_idx_exclusive=end_idx,
            require_trending=require_trending,
        )
        out[key] = (series, entries)
    return out


def _realize_map(collected, exit_policy: ExitPolicy = FIXED_HOLD) -> list[TradeResult]:
    trades: list[TradeResult] = []
    for _key, (series, entries) in collected.items():
        trades.extend(realize_entries(series, entries, exit_policy))
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

    w_atr_f = [x for x in (atr_pct(t) for t in wins) if x is not None]
    l_atr_f = [x for x in (atr_pct(t) for t in losses) if x is not None]
    profile["avg_atr_pct_wins"] = float(np.mean(w_atr_f)) if w_atr_f else None
    profile["avg_atr_pct_losses"] = float(np.mean(l_atr_f)) if l_atr_f else None

    for feat in FEATURE_KEYS:
        w_rate = sum(1 for t in wins if _flag(t, feat)) / len(wins) if wins else None
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
    keys = [k for k in keys if k in study_instruments()]

    print("  collecting series...", flush=True)
    series_map, errors, bars = load_series_map(keys, ["4h"], demo=demo)
    mode = "demo" if demo else "public_historical"

    print("  scoring TRAIN/TEST entries (once)...", flush=True)
    train_all_c = _collect_map(
        series_map, start_frac=0.0, end_frac=train_fraction, require_trending=False
    )
    test_all_c = _collect_map(
        series_map, start_frac=train_fraction, end_frac=1.0, require_trending=False
    )
    train_tr_c = _collect_map(
        series_map, start_frac=0.0, end_frac=train_fraction, require_trending=True
    )
    test_tr_c = _collect_map(
        series_map, start_frac=train_fraction, end_frac=1.0, require_trending=True
    )

    print("  realizing fixed-hold baselines...", flush=True)
    train_mh = _mh(_realize_map(train_all_c, FIXED_HOLD))
    test_mh = _mh(_realize_map(test_all_c, FIXED_HOLD))
    train_tmh = _mh(_realize_map(train_tr_c, FIXED_HOLD))
    test_tmh = _mh(_realize_map(test_tr_c, FIXED_HOLD))

    wl_train = _winner_loser_profile(train_tmh)
    wl_test = _winner_loser_profile(test_tmh)
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
        entry_oos[gate] = {
            "removed_mh": len(test_mh) - len(kept),
            "metrics": _enrich_bag(kept, f"entry/{gate}"),
        }

    print("  realizing stop-loss variants on same trending TEST entries...", flush=True)
    stop_oos: dict[str, Any] = {}
    for pol in ENTRY_STOP_POLICIES:
        trades = _mh(_realize_map(test_tr_c, pol))
        stop_oos[pol.name] = _enrich_bag(trades, f"stop/{pol.name}")

    print("  realizing take-profit variants on same trending TEST entries...", flush=True)
    tp_oos: dict[str, Any] = {}
    for pol in ENTRY_TP_POLICIES:
        trades = _mh(_realize_map(test_tr_c, pol))
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
            "enable candidates (rejected earlier). Exit variants share the same "
            "trending TEST entry set."
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
