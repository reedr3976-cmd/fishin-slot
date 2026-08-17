"""Causal macro/event features for V9 research.

All lookups use only observations with timestamp <= bar time.
Consensus/surprise remain UNKNOWN unless both actual and forecast were
available at that time (they are not, in this public dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np

from models import CandleSeries
from providers.macro_calendar import UNKNOWN, MacroEvent

ET = ZoneInfo("America/New_York")

PAIR_COUNTRIES: dict[str, tuple[str, str]] = {
    "EURUSD": ("EA", "US"),
    "GBPUSD": ("UK", "US"),
    "USDJPY": ("US", "JP"),
    "AUDUSD": ("AU", "US"),
    "USDCAD": ("US", "CA"),
    "USDCHF": ("US", "CH"),
}


def _asof(series: list[tuple[int, float]], ts: int) -> Optional[float]:
    if not series:
        return None
    lo, hi = 0, len(series) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= ts:
            ans = series[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _asof_px(series: Optional[CandleSeries], ts: int) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < 0:
        return None
    return float(series.close[idx])


def _change(series: Optional[CandleSeries], ts: int, lookback_days: int = 20) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < 1:
        return None
    # ~lookback_days on daily bars
    j = max(0, idx - lookback_days)
    a, b = float(series.close[j]), float(series.close[idx])
    if a == 0:
        return None
    return b - a


def _ret(series: Optional[CandleSeries], ts: int, lookback_days: int = 20) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    idx = int(np.searchsorted(series.timestamps, ts, side="right") - 1)
    if idx < 1:
        return None
    j = max(0, idx - lookback_days)
    a, b = float(series.close[j]), float(series.close[idx])
    if a == 0:
        return None
    return b / a - 1.0


def policy_cycle_from_short_rate(irx: Optional[CandleSeries], ts: int) -> str:
    """Market-observed US short-rate path (not a Fed narrative)."""
    ch = _change(irx, ts, 60)
    if ch is None:
        return UNKNOWN
    if ch > 0.10:  # 10bp over ~60 sessions
        return "tightening"
    if ch < -0.10:
        return "easing"
    return "unchanged"


@dataclass
class MacroContext:
    events: list[MacroEvent]
    events_by_asset: dict[str, list[MacroEvent]]
    boe_rate: list[tuple[int, float]]
    ecb_rate: list[tuple[int, float]]
    dxy: Optional[CandleSeries]
    us10y: Optional[CandleSeries]
    us3m: Optional[CandleSeries]
    tip: Optional[CandleSeries]

    def snapshot(self, instrument: str, ts: int, series: Optional[CandleSeries] = None) -> dict[str, Any]:
        """Causal snapshot at bar time. Never fills UNKNOWN with guesses."""
        nearest, dt_before, dt_after = self._nearest_event(instrument, ts)
        pre_vol, post_vol = UNKNOWN, UNKNOWN
        if nearest and nearest.ts_unix and series is not None:
            pre_vol, post_vol = self._event_vol(series, nearest.ts_unix)
        us_cycle = policy_cycle_from_short_rate(self.us3m, ts)
        irx = _asof_px(self.us3m, ts)
        tnx = _asof_px(self.us10y, ts)
        dxy = _asof_px(self.dxy, ts)
        real_proxy = UNKNOWN
        if tnx is not None:
            tip_ret = _ret(self.tip, ts, 20)
            # TIPS ETF return is not a real yield; keep labeled as proxy-only.
            real_proxy = {
                "us10y": tnx,
                "tip_20d_return": tip_ret if tip_ret is not None else UNKNOWN,
                "note": "Not a TIPS real-yield vintage; ETF return proxy only",
            }
        base_cty, quote_cty = PAIR_COUNTRIES.get(instrument, (UNKNOWN, UNKNOWN))
        rel = self._relative_policy(base_cty, quote_cty, ts, irx)
        return {
            "event": None
            if nearest is None
            else {
                "name": nearest.name,
                "category": nearest.category,
                "importance": nearest.importance,
                "country": nearest.country,
                "event_time": nearest.ts_unix,
                "time_precision": nearest.time_precision,
                "previous": nearest.previous,
                "consensus": nearest.consensus,
                "actual": nearest.actual,
                "surprise": nearest.surprise,
                "seconds_until": dt_after,
                "seconds_since": dt_before,
                "pre_event_vol": pre_vol,
                "post_event_vol": post_vol,
            },
            "us_policy_rate_proxy": irx if irx is not None else UNKNOWN,
            "us_10y": tnx if tnx is not None else UNKNOWN,
            "dxy": dxy if dxy is not None else UNKNOWN,
            "us_policy_cycle": us_cycle,
            "real_yield_proxy": real_proxy,
            "relative_policy": rel,
        }

    def _nearest_event(
        self, instrument: str, ts: int
    ) -> tuple[Optional[MacroEvent], Any, Any]:
        evs = self.events_by_asset.get(instrument) or self.events
        if not evs:
            return None, UNKNOWN, UNKNOWN
        # events sorted
        times = [e.ts_unix for e in evs if e.ts_unix]
        if not times:
            return None, UNKNOWN, UNKNOWN
        arr = np.array(times, dtype=np.int64)
        i = int(np.searchsorted(arr, ts, side="right") - 1)
        before = evs[i] if 0 <= i < len(evs) else None
        after = evs[i + 1] if 0 <= i + 1 < len(evs) else None
        # pick closer
        cand = []
        if before and before.ts_unix:
            cand.append((abs(ts - before.ts_unix), before, ts - before.ts_unix, None))
        if after and after.ts_unix:
            cand.append((abs(after.ts_unix - ts), after, None, after.ts_unix - ts))
        if not cand:
            return None, UNKNOWN, UNKNOWN
        cand.sort(key=lambda x: x[0])
        _, ev, since, until = cand[0]
        return ev, since if since is not None else UNKNOWN, until if until is not None else UNKNOWN

    def _event_vol(self, series: CandleSeries, event_ts: int) -> tuple[Any, Any]:
        """ATR% proxy 3 bars before vs 3 bars after event (causal series only)."""
        i = int(np.searchsorted(series.timestamps, event_ts, side="right") - 1)
        if i < 4 or i + 3 >= len(series):
            return UNKNOWN, UNKNOWN

        def atr_pct(lo: int, hi: int) -> Optional[float]:
            if hi <= lo:
                return None
            rng = (series.high[lo:hi] - series.low[lo:hi]) / np.maximum(series.close[lo:hi], 1e-12)
            return float(np.nanmean(rng))

        pre = atr_pct(max(0, i - 3), i)
        post = atr_pct(i + 1, min(len(series), i + 4))
        return (
            pre if pre is not None else UNKNOWN,
            post if post is not None else UNKNOWN,
        )

    def _relative_policy(self, base: str, quote: str, ts: int, us_irx: Optional[float]) -> dict[str, Any]:
        def rate_for(cty: str) -> Any:
            if cty == "US":
                return us_irx if us_irx is not None else UNKNOWN
            if cty == "UK":
                v = _asof(self.boe_rate, ts)
                return v if v is not None else UNKNOWN
            if cty == "EA":
                v = _asof(self.ecb_rate, ts)
                return v if v is not None else UNKNOWN
            return UNKNOWN

        br, qr = rate_for(base), rate_for(quote)
        diff: Any = UNKNOWN
        if br != UNKNOWN and qr != UNKNOWN and br is not None and qr is not None:
            diff = float(br) - float(qr)
        return {
            "base_country": base,
            "quote_country": quote,
            "base_policy_rate": br,
            "quote_policy_rate": qr,
            "rate_differential": diff,
            "note": "Foreign legs other than UK/EA are UNKNOWN; differential unused if any leg UNKNOWN.",
        }

    def in_blackout(
        self,
        instrument: str,
        ts: int,
        *,
        before_sec: int,
        after_sec: int,
        calendar_day: bool = False,
        skip_event_bar: bool = False,
        bar_seconds: int = 14400,
        min_importance: str = "HIGH",
    ) -> bool:
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, UNKNOWN: 0}
        need = rank.get(min_importance, 3)
        for ev in self.events_by_asset.get(instrument, []):
            if rank.get(ev.importance, 0) < need:
                continue
            if ev.ts_unix is None:
                continue
            # Intraday windows require a usable clock; date_only events only
            # participate in calendar_day blackouts.
            if calendar_day:
                ev_day = datetime.fromtimestamp(ev.ts_unix, tz=ET).date()
                bar_day = datetime.fromtimestamp(ts, tz=ET).date()
                if ev_day == bar_day:
                    return True
                continue
            if ev.time_precision == "date_only":
                continue
            if ev.time_precision == UNKNOWN:
                continue
            if before_sec == 0 and after_sec == 0 and not skip_event_bar:
                continue
            if ts + before_sec >= ev.ts_unix and ts - after_sec <= ev.ts_unix:
                return True
            if skip_event_bar:
                # still inside the 4H bar that contains the event
                if ev.ts_unix <= ts < ev.ts_unix + bar_seconds:
                    return True
        return False


def build_macro_context(
    bundle: dict[str, Any],
    series_4h: dict[tuple[str, str], CandleSeries],
    daily_map: dict[str, CandleSeries],
) -> MacroContext:
    events: list[MacroEvent] = list(bundle.get("events") or [])
    by_asset: dict[str, list[MacroEvent]] = {}
    for ev in events:
        for a in ev.affected_assets:
            by_asset.setdefault(a, []).append(ev)
    for a in by_asset:
        by_asset[a].sort(key=lambda e: e.ts_unix or 0)
    return MacroContext(
        events=events,
        events_by_asset=by_asset,
        boe_rate=list(bundle.get("boe_rate") or []),
        ecb_rate=list(bundle.get("ecb_rate") or []),
        dxy=daily_map.get("DXY") or series_4h.get(("DXY", "4h")),
        us10y=daily_map.get("US10Y") or series_4h.get(("US10Y", "4h")),
        us3m=daily_map.get("US3M") or series_4h.get(("US3M", "4h")),
        tip=daily_map.get("TIP") or series_4h.get(("TIP", "4h")),
    )


def gold_filter_ok(ctx: MacroContext, ts: int, mode: str) -> bool:
    """Independent gold-context filters. UNKNOWN ⇒ do not block (cannot decide)."""
    if mode in ("none", "", None):
        return True
    if mode == "dxy_not_rising":
        r = _ret(ctx.dxy, ts, 20)
        if r is None:
            return True  # UNKNOWN
        return r <= 0
    if mode == "yields_not_rising":
        ch = _change(ctx.us10y, ts, 20)
        if ch is None:
            return True
        return ch <= 0
    if mode == "not_tightening":
        cyc = policy_cycle_from_short_rate(ctx.us3m, ts)
        if cyc == UNKNOWN:
            return True
        return cyc != "tightening"
    return True


def fx_relative_ok(ctx: MacroContext, instrument: str, direction: str, ts: int, mode: str) -> bool:
    """Relative policy filter. UNKNOWN differential ⇒ do not use it to decide."""
    if mode in ("none", "", None):
        return True
    snap = ctx.snapshot(instrument, ts)
    diff = (snap.get("relative_policy") or {}).get("rate_differential")
    if diff == UNKNOWN or diff is None:
        return True
    # Long base / short quote when base rate > quote rate (carry alignment)
    if mode == "carry_align":
        if direction == "bullish":
            return float(diff) >= 0
        return float(diff) <= 0
    return True
