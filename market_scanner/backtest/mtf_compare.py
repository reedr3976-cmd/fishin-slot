"""Compare ORIGINAL vs multi-timeframe-filtered scanner on chronological TEST."""

from __future__ import annotations

from typing import Any, Optional

from config import VALIDATION_TRAIN_FRACTION
from backtest.engine import load_series_map, run_backtest_on_map
from backtest.metrics import group_metrics, summarize_trades
from backtest.mtf_backtest import run_mtf_backtest_on_map
from scanner.scoring import ORIGINAL_RULES


def run_mtf_comparison(
    *,
    demo: bool = False,
    instruments=None,
    timeframes=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    """ORIGINAL vs MTF-filtered using the same train/test split and costs.

    Scoring thresholds are unchanged (ORIGINAL_RULES). MTF only gates MEDIUM/HIGH.
    Default instrument universe excludes crypto.
    """
    tfs = list(timeframes) if timeframes is not None else ["1d", "1wk"]
    series_map, errors, bars = load_series_map(instruments, tfs, demo=demo)
    mode = "demo" if demo else "public_historical"

    # Restrict to instruments that have both 1d and 1wk for a fair comparison
    paired_keys = sorted(
        {
            k
            for k, _ in series_map
            if (k, "1d") in series_map and (k, "1wk") in series_map
        }
    )
    paired_map = {
        (k, tf): series_map[(k, tf)]
        for k in paired_keys
        for tf in ("1d", "1wk")
        if (k, tf) in series_map
    }

    orig_train = run_backtest_on_map(
        paired_map,
        ORIGINAL_RULES,
        start_frac=0.0,
        end_frac=train_fraction,
        mode=mode,
        errors=errors,
    )
    orig_test = run_backtest_on_map(
        paired_map,
        ORIGINAL_RULES,
        start_frac=train_fraction,
        end_frac=1.0,
        mode=mode,
        errors=errors,
    )

    mtf_train, mtf_train_stats = run_mtf_backtest_on_map(
        paired_map,
        ORIGINAL_RULES,
        start_frac=0.0,
        end_frac=train_fraction,
        mode=mode,
        errors=errors,
    )
    mtf_test, mtf_test_stats = run_mtf_backtest_on_map(
        paired_map,
        ORIGINAL_RULES,
        start_frac=train_fraction,
        end_frac=1.0,
        mode=mode,
        errors=errors,
    )

    return {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "instruments": paired_keys,
        "errors": errors,
        "original_train": orig_train,
        "original_test": orig_test,
        "mtf_train": mtf_train,
        "mtf_test": mtf_test,
        "mtf_train_stats": mtf_train_stats,
        "mtf_test_stats": mtf_test_stats,
        "metrics": {
            "original_train": group_metrics(orig_train.trades),
            "original_test": group_metrics(orig_test.trades),
            "mtf_train": group_metrics(mtf_train.trades),
            "mtf_test": group_metrics(mtf_test.trades),
        },
        "high_medium_test": {
            "original": summarize_trades(
                "orig_MH",
                [t for t in orig_test.trades if t.confidence in ("HIGH", "MEDIUM")],
            ),
            "mtf": summarize_trades(
                "mtf_MH",
                [t for t in mtf_test.trades if t.confidence in ("HIGH", "MEDIUM")],
            ),
        },
    }
