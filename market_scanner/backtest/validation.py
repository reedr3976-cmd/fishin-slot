"""Out-of-sample validation for scanner confidence scoring.

Chronological train/test split per series. Revised rules are proposed using
TRAINING data only, then frozen and evaluated on the unseen TEST window.
No shuffling. No look-ahead in features (entries still use sliced history).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from config import (
    MIN_FEATURE_HITS_FOR_EDGE,
    MIN_SIGNALS_FOR_CONCLUSION,
    REVISED_HIGH_QUANTILE,
    REVISED_MEDIUM_QUANTILE,
    VALIDATION_TRAIN_FRACTION,
)
from backtest.engine import load_series_map, run_backtest_on_map
from backtest.metrics import MetricBag, TradeResult, group_metrics, summarize_trades
from scanner.scoring import ORIGINAL_RULES, ScoringRules, explain_why_high_is_rare


FEATURE_KEYS = [
    "sma_cross",
    "sma_stack",
    "rsi_extreme",
    "rsi_exit",
    "rsi_mild",
    "macd_cross",
    "macd_strong",
    "macd_mild",
    "bb_touch",
    "near_support",
    "near_resistance",
    "high_atr",
]


@dataclass
class FeatureEdge:
    feature: str
    hits: int
    avg_net_when_on: Optional[float]
    avg_net_when_off: Optional[float]
    lift: Optional[float]  # on - off
    helpful: Optional[bool]  # True if lift > 0 with enough hits
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "hits": self.hits,
            "avg_net_when_on": self.avg_net_when_on,
            "avg_net_when_off": self.avg_net_when_off,
            "lift": self.lift,
            "helpful": self.helpful,
            "note": self.note,
        }


def analyze_feature_edges(trades: list[TradeResult]) -> list[FeatureEdge]:
    """Estimate whether each feature adds predictive value on a trade set (TRAIN)."""
    if not trades:
        return [
            FeatureEdge(f, 0, None, None, None, None, "No trades to analyze.")
            for f in FEATURE_KEYS
        ]

    edges: list[FeatureEdge] = []
    for feat in FEATURE_KEYS:
        on = [t.net_return for t in trades if (t.feature_flags or {}).get(feat, 0) == 1]
        off = [t.net_return for t in trades if (t.feature_flags or {}).get(feat, 0) != 1]
        hits = len(on)
        if hits < MIN_FEATURE_HITS_FOR_EDGE:
            edges.append(
                FeatureEdge(
                    feat,
                    hits,
                    float(np.mean(on)) if on else None,
                    float(np.mean(off)) if off else None,
                    None,
                    None,
                    f"Too few hits ({hits} < {MIN_FEATURE_HITS_FOR_EDGE}) — inconclusive.",
                )
            )
            continue
        avg_on = float(np.mean(on))
        avg_off = float(np.mean(off)) if off else 0.0
        lift = avg_on - avg_off
        helpful = lift > 0
        edges.append(
            FeatureEdge(
                feat,
                hits,
                avg_on,
                avg_off,
                lift,
                helpful,
                "Positive lift on train." if helpful else "Flat/negative lift on train.",
            )
        )
    return edges


def _clip_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def propose_revised_rules(
    train_trades: list[TradeResult],
    feature_edges: list[FeatureEdge],
    base: ScoringRules = ORIGINAL_RULES,
) -> tuple[ScoringRules, list[str]]:
    """Propose revised weights/thresholds from TRAINING diagnostics only.

    Design choices (pre-specified to limit overfitting):
    - Keep the same feature set; only modestly re-weight (±50%) using train lift sign.
    - Recalibrate HIGH/MEDIUM cutoffs from train actionable score quantiles
      (pre-specified quantiles — not chosen to maximize train return).
    - Require ≥2 core agreeing factors for HIGH to avoid one-off spikes.
    """
    rationale: list[str] = []
    weight_map = {
        "sma_cross": "sma_cross",
        "sma_stack": "sma_stack",
        "rsi_extreme": "rsi_extreme",
        "rsi_exit": "rsi_exit",
        "rsi_mild": "rsi_mild",
        "macd_cross": "macd_cross",
        "macd_strong": "macd_strong",
        "macd_mild": "macd_mild",
        "bb_touch": "bb_touch",
    }

    kwargs: dict[str, Any] = {"name": "revised_candidate", "notes": ""}
    for edge in feature_edges:
        attr = weight_map.get(edge.feature)
        if attr is None:
            continue  # S/R and ATR are diagnostic only in original scoring
        base_w = getattr(base, attr)
        if edge.helpful is True:
            new_w = _clip_int(base_w * 1.25, 1, 40)
            if new_w != base_w:
                rationale.append(
                    f"{edge.feature}: train lift +{edge.lift:.4f} → weight {base_w}→{new_w}"
                )
            kwargs[attr] = new_w
        elif edge.helpful is False:
            new_w = _clip_int(base_w * 0.75, 1, 40)
            if new_w != base_w:
                rationale.append(
                    f"{edge.feature}: train lift {edge.lift:.4f} → weight {base_w}→{new_w}"
                )
            kwargs[attr] = new_w
        # inconclusive → keep base (don't set)

    # Also scale rsi_extreme_strong with rsi_extreme if adjusted
    if "rsi_extreme" in kwargs:
        kwargs["rsi_extreme_strong"] = _clip_int(kwargs["rsi_extreme"] * 1.4, 1, 45)

    # Threshold recalibration from train actionable scores under ORIGINAL rules
    # (uses scores already collected on train with original weights).
    # After weight changes, we recompute cutoffs from a simple transform:
    # apply relative scale of mean score — but we don't have revised scores yet.
    # So: set quantiles on original train scores as the target HIGH/MEDIUM rates,
    # then after building provisional rules we optionally refine using a second
    # pass in run_validation (see propose_thresholds_from_scores).
    scores = [t.score for t in train_trades if t.confidence in ("HIGH", "MEDIUM", "LOW")]
    if len(scores) >= MIN_SIGNALS_FOR_CONCLUSION:
        high_cut = int(np.quantile(scores, REVISED_HIGH_QUANTILE))
        med_cut = int(np.quantile(scores, REVISED_MEDIUM_QUANTILE))
        # Ensure ordering and headroom above score_low
        high_cut = max(high_cut, base.score_low + 10)
        med_cut = max(min(med_cut, high_cut - 1), base.score_low)
        kwargs["score_high"] = high_cut
        kwargs["score_medium"] = med_cut
        kwargs["score_low"] = base.score_low
        rationale.append(
            f"Thresholds from train score quantiles "
            f"(HIGH≥q{REVISED_HIGH_QUANTILE:.2f}={high_cut}, "
            f"MEDIUM≥q{REVISED_MEDIUM_QUANTILE:.2f}={med_cut}, "
            f"LOW≥{base.score_low}). Quantiles were pre-specified, not return-optimized."
        )
    else:
        rationale.append(
            "Too few train actionable scores to recalibrate thresholds; kept originals."
        )

    kwargs["high_min_factors"] = 2
    rationale.append(
        "HIGH also requires ≥2 agreeing factors (pre-specified gate to reduce flukes)."
    )
    kwargs["notes"] = " | ".join(rationale)
    data = base.to_dict()
    data.update(kwargs)
    revised = ScoringRules(**data)
    return revised, rationale


def refine_thresholds_on_train_scores(
    rules: ScoringRules,
    train_scores: list[int],
) -> ScoringRules:
    """Freeze HIGH/MEDIUM cutoffs from revised-rule TRAIN scores (quantiles)."""
    if len(train_scores) < MIN_SIGNALS_FOR_CONCLUSION:
        return rules
    high_cut = int(np.quantile(train_scores, REVISED_HIGH_QUANTILE))
    med_cut = int(np.quantile(train_scores, REVISED_MEDIUM_QUANTILE))
    high_cut = max(high_cut, rules.score_low + 10)
    med_cut = max(min(med_cut, high_cut - 1), rules.score_low)
    data = rules.to_dict()
    data["score_high"] = high_cut
    data["score_medium"] = med_cut
    data["notes"] = (
        (rules.notes or "")
        + f" | Refined thresholds on revised train scores: HIGH≥{high_cut}, MEDIUM≥{med_cut}."
    )
    return ScoringRules(**data)


def _monotonic_high_med_low(
    by_conf: dict[str, MetricBag], *, require_reliable: bool = True
) -> Optional[bool]:
    """Return True if avg_return HIGH > MEDIUM > LOW among buckets with signals."""
    vals = []
    for name in ("HIGH", "MEDIUM", "LOW"):
        bag = by_conf[name]
        if bag.signals <= 0 or bag.avg_return is None:
            return None
        if require_reliable and not bag.reliable:
            return None
        vals.append(bag.avg_return)
    return vals[0] > vals[1] > vals[2]


def run_validation(
    *,
    demo: bool = False,
    instruments=None,
    timeframes=None,
    train_fraction: float = VALIDATION_TRAIN_FRACTION,
) -> dict[str, Any]:
    """Full OOS pipeline. Returns a structured result dict for reporting."""
    series_map, errors, bars = load_series_map(instruments, timeframes, demo=demo)
    mode = "demo" if demo else "public_historical"

    # --- ORIGINAL on train / test ---
    orig_train = run_backtest_on_map(
        series_map, ORIGINAL_RULES, start_frac=0.0, end_frac=train_fraction, mode=mode, errors=errors
    )
    orig_test = run_backtest_on_map(
        series_map, ORIGINAL_RULES, start_frac=train_fraction, end_frac=1.0, mode=mode, errors=errors
    )

    feature_edges = analyze_feature_edges(orig_train.trades)
    revised_draft, rationale = propose_revised_rules(orig_train.trades, feature_edges)

    # Second pass: run revised weights on TRAIN to freeze quantile thresholds
    revised_train_pass = run_backtest_on_map(
        series_map,
        revised_draft,
        start_frac=0.0,
        end_frac=train_fraction,
        mode=mode,
        errors=errors,
    )
    train_scores = [t.score for t in revised_train_pass.trades]
    revised = refine_thresholds_on_train_scores(revised_draft, train_scores)

    # Freeze revised; evaluate train (final) + test
    rev_train = run_backtest_on_map(
        series_map, revised, start_frac=0.0, end_frac=train_fraction, mode=mode, errors=errors
    )
    rev_test = run_backtest_on_map(
        series_map, revised, start_frac=train_fraction, end_frac=1.0, mode=mode, errors=errors
    )

    result = {
        "mode": mode,
        "train_fraction": train_fraction,
        "bars_loaded": bars,
        "errors": errors,
        "why_high_rare": explain_why_high_is_rare(ORIGINAL_RULES),
        "feature_edges_train": feature_edges,
        "rationale": rationale,
        "original_rules": ORIGINAL_RULES,
        "revised_rules": revised,
        "original_train": orig_train,
        "original_test": orig_test,
        "revised_train": rev_train,
        "revised_test": rev_test,
        "metrics": {
            "original_train": group_metrics(orig_train.trades),
            "original_test": group_metrics(orig_test.trades),
            "revised_train": group_metrics(rev_train.trades),
            "revised_test": group_metrics(rev_test.trades),
        },
    }
    result["recommendation"] = make_recommendation(result)
    return result


def make_recommendation(result: dict[str, Any]) -> dict[str, Any]:
    """Decide KEEP ORIGINAL / ADOPT REVISED / NEED MORE DATA from TEST evidence."""
    o = result["metrics"]["original_test"]["by_confidence"]
    r = result["metrics"]["revised_test"]["by_confidence"]
    o_all = result["metrics"]["original_test"]["overall"]
    r_all = result["metrics"]["revised_test"]["overall"]

    reasons: list[str] = []
    high_o, high_r = o["HIGH"].signals, r["HIGH"].signals

    if high_o < MIN_SIGNALS_FOR_CONCLUSION and high_r < MIN_SIGNALS_FOR_CONCLUSION:
        reasons.append(
            f"TEST HIGH samples remain too small (original={high_o}, revised={high_r}; "
            f"need ≥{MIN_SIGNALS_FOR_CONCLUSION})."
        )
        decision = "NEED MORE DATA"
    else:
        # Prefer revised only if: more usable HIGH sample OR better monotonicity,
        # and overall test avg return not worse in a meaningful way.
        mono_o = _monotonic_high_med_low(o)
        mono_r = _monotonic_high_med_low(r)
        reasons.append(f"Original TEST monotonic HIGH>MED>LOW: {mono_o}")
        reasons.append(f"Revised TEST monotonic HIGH>MED>LOW: {mono_r}")
        reasons.append(
            f"Original TEST overall avg={o_all.avg_return}, n={o_all.signals}; "
            f"Revised TEST overall avg={r_all.avg_return}, n={r_all.signals}"
        )

        improved_ranking = mono_r is True and mono_o is not True
        high_ok = high_r >= MIN_SIGNALS_FOR_CONCLUSION
        overall_ok = (
            r_all.avg_return is not None
            and o_all.avg_return is not None
            and r_all.avg_return >= o_all.avg_return - 0.001  # within 0.1% tolerance
        )

        if high_ok and improved_ranking and overall_ok:
            decision = "ADOPT REVISED"
            reasons.append(
                "Revised improves ranking structure with adequate HIGH sample "
                "without clearly hurting overall TEST return."
            )
        elif high_ok and mono_r is True and overall_ok and (o_all.avg_return or 0) <= (r_all.avg_return or 0):
            decision = "ADOPT REVISED"
            reasons.append("Revised shows HIGH>MED>LOW on TEST with enough HIGH trades.")
        else:
            decision = "KEEP ORIGINAL"
            reasons.append(
                "Evidence is not strong enough to replace the merged original rules. "
                "Revised may still be useful as an experimental profile."
            )
            if not high_ok:
                reasons.append("Revised HIGH sample still below reliability threshold.")
            if mono_r is not True:
                reasons.append("Revised did not demonstrate HIGH > MEDIUM > LOW on TEST.")

    # Extra caution: never adopt solely on tiny HIGH wins
    if decision == "ADOPT REVISED" and high_r < MIN_SIGNALS_FOR_CONCLUSION:
        decision = "NEED MORE DATA"
        reasons.append("Blocked adopt: HIGH sample too small to trust.")

    return {
        "decision": decision,
        "reasons": reasons,
        "original_test_high_n": high_o,
        "revised_test_high_n": high_r,
        "min_signals_required": MIN_SIGNALS_FOR_CONCLUSION,
    }
