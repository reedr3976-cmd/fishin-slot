"""Trend-quality filter study (analysis only).

Primary TF = 4H. ORIGINAL entries (MH + sma_stack) are the control.
Filters use causal indicators only. Thresholds are either classic fixed
levels (ADX 20/25) or TRAIN medians shared across instruments (no
per-symbol fit). Live scanner is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np

from config import (
    MIN_SIGNALS_FOR_CONCLUSION,
    ROUND_TRIP_COST,
    STUDY_ASSET_CLASSES,
    VALIDATION_TRAIN_FRACTION,
    study_instruments,
)
from indicators import compute_all, swing_structure_dir
from backtest.engine import load_series_map
from backtest.fourh_exits import (
    FIXED_HOLD,
    EntrySignal,
    collect_entries,
    realize_entries,
)
from backtest.metrics import TradeResult, summarize_trades
from scanner.scoring import ORIGINAL_RULES


def _is_mh_trend(e: EntrySignal) -> bool:
    return (
        e.confidence in ("HIGH", "MEDIUM")
        and int((e.feature_flags or {}).get("sma_stack", 0) or 0) == 1
    )


def _metrics(trades: list[TradeResult], label: str) -> dict[str, Any]:
    return summarize_trades(label, trades).to_dict()


def _by_class(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    return {
        cls: _metrics([t for t in trades if t.asset_class == cls], cls)
        for cls in ("commodity", "forex", "stocks")
    }


def _by_symbol(trades: list[TradeResult]) -> dict[str, dict[str, Any]]:
    out = {}
    for inst in sorted({t.instrument for t in trades}):
        subset = [t for t in trades if t.instrument == inst]
        d = _metrics(subset, inst)
        d["asset_class"] = subset[0].asset_class
        out[inst] = d
    return out


def _pack(trades: list[TradeResult], label: str) -> dict[str, Any]:
    return {
        "all": _metrics(trades, label),
        "by_asset_class": _by_class(trades),
        "by_symbol": _by_symbol(trades),
        "n_symbols_positive_avg": sum(
            1
            for m in _by_symbol(trades).values()
            if (m.get("avg_return") or 0) > 0 and m.get("signals", 0) >= 3
        ),
    }


def _enrich_series_entries(
    series, entries: list[EntrySignal], daily_series=None
) -> list[EntrySignal]:
    """Attach causal trend-quality fields using full-series indicators."""
    if not entries:
        return entries
    ind = compute_all(series.close, series.high, series.low)
    atr_arr = ind["atr"]
    # Precompute rolling median ATR for expansion ratio
    atr_med = np.full(len(series), np.nan)
    for i in range(len(series)):
        window = atr_arr[max(0, i - 49) : i + 1]
        window = window[~np.isnan(window)]
        if len(window) >= 10:
            atr_med[i] = float(np.median(window))

    # Daily direction map by timestamp (latest 1d bar <= ts)
    daily_dir_by_ts = None
    if daily_series is not None and len(daily_series) > 0:
        d_ind = compute_all(daily_series.close, daily_series.high, daily_series.low)
        daily_dir_by_ts = []
        for j in range(len(daily_series)):
            price = float(daily_series.close[j])
            s20 = d_ind["sma_fast"][j]
            s50 = d_ind["sma_slow"][j]
            direction = None
            if not (np.isnan(s20) or np.isnan(s50)):
                if price > s20 > s50:
                    direction = "bullish"
                elif price < s20 < s50:
                    direction = "bearish"
            daily_dir_by_ts.append((int(daily_series.timestamps[j]), direction))

    out = []
    for e in entries:
        i = e.entry_idx
        adx_v = float(ind["adx"][i]) if not np.isnan(ind["adx"][i]) else None
        pdi = float(ind["plus_di"][i]) if not np.isnan(ind["plus_di"][i]) else None
        mdi = float(ind["minus_di"][i]) if not np.isnan(ind["minus_di"][i]) else None
        ema_f = ind["ema_fast"][i]
        ema_s = ind["ema_slow"][i]
        atr_v = atr_arr[i]
        ema_sep = None
        ema_slope = None
        dist = None
        expand = None
        if not np.isnan(atr_v) and atr_v > 0:
            if not (np.isnan(ema_f) or np.isnan(ema_s)):
                ema_sep = abs(float(ema_f) - float(ema_s)) / float(atr_v)
            if i >= 5 and not np.isnan(ind["ema_slow"][i - 5]):
                ema_slope = (float(ema_s) - float(ind["ema_slow"][i - 5])) / float(atr_v)
            if e.sma50 is not None:
                dist = abs(e.entry_price - e.sma50) / float(atr_v)
            if not np.isnan(atr_med[i]) and atr_med[i] > 0:
                expand = float(atr_v) / float(atr_med[i])

        struct = swing_structure_dir(series.high, series.low, i)
        daily_agree = None
        if daily_dir_by_ts is not None:
            ddir = None
            for ts, direction in daily_dir_by_ts:
                if ts <= e.entry_ts:
                    ddir = direction
                else:
                    break
            if ddir is not None:
                daily_agree = ddir == e.direction

        out.append(
            replace(
                e,
                adx=adx_v,
                plus_di=pdi,
                minus_di=mdi,
                ema_sep_atr=ema_sep,
                ema_slope_atr=ema_slope,
                dist_sma50_atr=dist,
                atr_expand=expand,
                structure_dir=struct,
                daily_agree=daily_agree,
            )
        )
    return out


def _collect_enriched(series_4h, series_1d_map):
    collected = {}
    for key, series in series_4h.items():
        inst = key[0] if isinstance(key, tuple) else key
        entries = [e for e in collect_entries(series, ORIGINAL_RULES) if _is_mh_trend(e)]
        daily = series_1d_map.get(inst)
        collected[key] = (series, _enrich_series_entries(series, entries, daily))
    return collected


def _slice_collected(collected, start_frac: float, end_frac: float):
    out = {}
    for key, (series, entries) in collected.items():
        n = len(series)
        lo, hi = int(n * start_frac), int(n * end_frac)
        out[key] = (series, [e for e in entries if lo <= e.entry_idx < hi])
    return out


def _realize(collected, pred: Callable[[EntrySignal], bool], *, cost_mult: float = 1.0):
    trades: list[TradeResult] = []
    overrides = {
        "commodity": ROUND_TRIP_COST["commodity"] * cost_mult,
        "forex": ROUND_TRIP_COST["forex"] * cost_mult,
        "stocks": ROUND_TRIP_COST["stocks"] * cost_mult,
    }
    for _k, (series, entries) in collected.items():
        kept = [e for e in entries if pred(e)]
        trades.extend(
            realize_entries(
                series,
                kept,
                FIXED_HOLD,
                cost_override=overrides.get(series.asset_class),
            )
        )
    return trades


def _flatten(collected) -> list[EntrySignal]:
    out: list[EntrySignal] = []
    for _k, (_s, entries) in collected.items():
        out.extend(entries)
    return out


def _train_median(entries: list[EntrySignal], attr: str) -> Optional[float]:
    vals = [getattr(e, attr) for e in entries if getattr(e, attr) is not None]
    return float(np.median(vals)) if vals else None


@dataclass(frozen=True)
class TQCandidate:
    name: str
    description: str
    pred: Callable[[EntrySignal], bool]
    scope: str  # universal | commodity | forex | stocks


def run_trend_quality_study(
    *,
    demo: bool = False,
    instruments=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    keys = list(instruments) if instruments else list(study_instruments().keys())
    keys = [k for k in keys if k in study_instruments()]

    print("  loading 4H + 1D...", flush=True)
    series_4h, errors, bars = load_series_map(keys, ["4h"], demo=demo)
    series_1d, errors_1d, bars_1d = load_series_map(keys, ["1d"], demo=demo)
    errors = list(errors) + list(errors_1d)
    # Remap 1d to instrument key only
    daily_map = {k: series_1d[(k, "1d")] for k, _tf in series_1d if _tf == "1d"}

    print("  collecting + enriching trend-quality features...", flush=True)
    full = _collect_enriched(series_4h, daily_map)
    train_c = _slice_collected(full, 0.0, train_fraction)
    test_c = _slice_collected(full, train_fraction, 1.0)
    train_flat = _flatten(train_c)

    med_ema = _train_median(train_flat, "ema_sep_atr") or 1.0
    med_dist = _train_median(train_flat, "dist_sma50_atr") or 1.0
    med_slope = _train_median(train_flat, "ema_slope_atr")
    # slope is signed; use abs median for strength
    abs_slopes = [abs(e.ema_slope_atr) for e in train_flat if e.ema_slope_atr is not None]
    med_abs_slope = float(np.median(abs_slopes)) if abs_slopes else 0.2

    def orig(e: EntrySignal) -> bool:
        return True  # already MH trending

    def adx25(e: EntrySignal) -> bool:
        return e.adx is not None and e.adx >= 25

    def adx20(e: EntrySignal) -> bool:
        return e.adx is not None and e.adx >= 20

    def adx_dir(e: EntrySignal) -> bool:
        if e.adx is None or e.adx < 25 or e.plus_di is None or e.minus_di is None:
            return False
        if e.direction == "bullish":
            return e.plus_di > e.minus_di
        return e.minus_di > e.plus_di

    def structure(e: EntrySignal) -> bool:
        if e.direction == "bullish":
            return e.structure_dir == 1
        if e.direction == "bearish":
            return e.structure_dir == -1
        return False

    def ema_sep(e: EntrySignal) -> bool:
        return e.ema_sep_atr is not None and e.ema_sep_atr >= med_ema

    def atr_expand(e: EntrySignal) -> bool:
        return e.atr_expand is not None and e.atr_expand >= 1.0

    def dist_ma(e: EntrySignal) -> bool:
        return e.dist_sma50_atr is not None and e.dist_sma50_atr >= med_dist

    def slope_ok(e: EntrySignal) -> bool:
        if e.ema_slope_atr is None:
            return False
        if e.direction == "bullish":
            return e.ema_slope_atr >= med_abs_slope
        return e.ema_slope_atr <= -med_abs_slope

    def mtf_1d(e: EntrySignal) -> bool:
        return e.daily_agree is True

    def combo_adx_struct(e: EntrySignal) -> bool:
        return adx_dir(e) and structure(e)

    def combo_adx_ema(e: EntrySignal) -> bool:
        return adx_dir(e) and ema_sep(e)

    def combo_adx_mtf(e: EntrySignal) -> bool:
        return adx_dir(e) and mtf_1d(e)

    def combo_struct_expand(e: EntrySignal) -> bool:
        return structure(e) and atr_expand(e)

    def scoped(pred, cls: str):
        def _p(e: EntrySignal) -> bool:
            if e.asset_class != cls:
                return True
            return pred(e)

        return _p

    candidates = [
        TQCandidate("ORIGINAL", "MH trending + fixed hold (control)", orig, "universal"),
        TQCandidate("TQ_ADX25", "ADX ≥ 25 (classic)", adx25, "universal"),
        TQCandidate("TQ_ADX20", "ADX ≥ 20", adx20, "universal"),
        TQCandidate("TQ_ADX25_dir", "ADX ≥ 25 and DI aligned with direction", adx_dir, "universal"),
        TQCandidate("TQ_structure", "HH/HL or LH/LL aligns with direction", structure, "universal"),
        TQCandidate(
            "TQ_ema_sep",
            f"|EMA12-EMA26|/ATR ≥ TRAIN median ({med_ema:.3f})",
            ema_sep,
            "universal",
        ),
        TQCandidate("TQ_atr_expand", "ATR ≥ 50-bar median (expansion)", atr_expand, "universal"),
        TQCandidate(
            "TQ_dist_sma50",
            f"|price-SMA50|/ATR ≥ TRAIN median ({med_dist:.3f})",
            dist_ma,
            "universal",
        ),
        TQCandidate(
            "TQ_ema_slope",
            f"EMA26 slope / ATR aligned, |slope| ≥ TRAIN median ({med_abs_slope:.3f})",
            slope_ok,
            "universal",
        ),
        TQCandidate("TQ_mtf_1d", "Daily SMA stack agrees with 4H direction", mtf_1d, "universal"),
        TQCandidate("TQ_ADX_structure", "ADX25+DI and swing structure", combo_adx_struct, "universal"),
        TQCandidate("TQ_ADX_ema", "ADX25+DI and EMA separation", combo_adx_ema, "universal"),
        TQCandidate("TQ_ADX_mtf", "ADX25+DI and daily agree", combo_adx_mtf, "universal"),
        TQCandidate(
            "TQ_structure_expand",
            "Structure + ATR expansion",
            combo_struct_expand,
            "universal",
        ),
        # Class-scoped (same rule, applied only to one class)
        TQCandidate(
            "COMM_ADX25_dir",
            "Commodity-only ADX25+DI; FX/stocks unchanged",
            scoped(adx_dir, "commodity"),
            "commodity",
        ),
        TQCandidate(
            "FX_ADX25_dir",
            "Forex-only ADX25+DI; others unchanged",
            scoped(adx_dir, "forex"),
            "forex",
        ),
        TQCandidate(
            "STK_ADX25_dir",
            "Stocks-only ADX25+DI; others unchanged",
            scoped(adx_dir, "stocks"),
            "stocks",
        ),
        TQCandidate(
            "COMM_structure",
            "Commodity-only structure filter",
            scoped(structure, "commodity"),
            "commodity",
        ),
        TQCandidate(
            "FX_structure",
            "Forex-only structure filter",
            scoped(structure, "forex"),
            "forex",
        ),
    ]

    folds = [("F1", 0.0, 0.25), ("F2", 0.25, 0.5), ("F3", 0.5, 0.75), ("F4", 0.75, 1.0)]

    print("  evaluating candidates...", flush=True)
    results = {}
    for cand in candidates:
        test_1x = _realize(test_c, cand.pred, cost_mult=1.0)
        test_2x = _realize(test_c, cand.pred, cost_mult=2.0)
        fold_packs = {}
        fold_pos = 0
        for fname, a, b in folds:
            coll = _slice_collected(full, a, b)
            pack = _pack(_realize(coll, cand.pred, cost_mult=1.0), f"{cand.name}/{fname}")
            fold_packs[fname] = pack
            avg = pack["all"].get("avg_return")
            if avg is not None and avg > 0:
                fold_pos += 1

        results[cand.name] = {
            "description": cand.description,
            "scope": cand.scope,
            "test_1x": _pack(test_1x, cand.name),
            "test_2x": _pack(test_2x, f"{cand.name}/2x"),
            "folds": fold_packs,
            "folds_positive": fold_pos,
            "removed_vs_original": None,
        }

    orig_n = results["ORIGINAL"]["test_1x"]["all"]["signals"]
    orig_dd = results["ORIGINAL"]["test_1x"]["all"].get("max_drawdown") or 0.0
    orig_avg = results["ORIGINAL"]["test_1x"]["all"].get("avg_return")

    for name, block in results.items():
        block["removed_vs_original"] = orig_n - block["test_1x"]["all"]["signals"]
        # Robustness gate
        t = block["test_1x"]["all"]
        t2 = block["test_2x"]["all"]
        avg = t.get("avg_return")
        avg2 = t2.get("avg_return")
        n = t.get("signals", 0)
        dd = t.get("max_drawdown")
        dd_ok = dd is None or orig_dd is None or dd <= orig_dd + 0.05
        multi_sym = block["test_1x"].get("n_symbols_positive_avg", 0) >= 2
        beats = avg is not None and orig_avg is not None and avg > orig_avg
        block["passes_robustness"] = bool(
            name != "ORIGINAL"
            and beats
            and avg2 is not None
            and avg2 > 0
            and block["folds_positive"] >= 3
            and n >= MIN_SIGNALS_FOR_CONCLUSION
            and dd_ok
            and multi_sym
        )

    return {
        "mode": "demo" if demo else "public_historical",
        "train_fraction": train_fraction,
        "bars_loaded": bars + bars_1d,
        "instruments": sorted({k for k, _ in series_4h}),
        "asset_classes": list(STUDY_ASSET_CLASSES),
        "timeframe": "4h",
        "errors": errors,
        "live_unchanged": True,
        "train_thresholds": {
            "ema_sep_atr_median": med_ema,
            "dist_sma50_atr_median": med_dist,
            "abs_ema_slope_atr_median": med_abs_slope,
            "adx_levels": [20, 25],
        },
        "candidates": results,
        "original_test_n": orig_n,
        "original_test_avg": orig_avg,
        "original_test_dd": orig_dd,
        "min_signals": MIN_SIGNALS_FOR_CONCLUSION,
        "note": (
            "Trend-quality study on ORIGINAL 4H MH+sma_stack entries. "
            "No per-symbol optimization. No look-ahead. Live unchanged."
        ),
    }
