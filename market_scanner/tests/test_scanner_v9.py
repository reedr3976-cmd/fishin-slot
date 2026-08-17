"""Offline tests for V9 macro safety and live-universe protection."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import INSTRUMENTS, V9_EVENT_WINDOWS, active_instruments
from backtest.macro_features import MacroContext, gold_filter_ok
from backtest.report_scanner_v9 import build_specs, select_on_train
from providers.macro_calendar import UNKNOWN, MacroEvent, _parse_bls_datetime, weekly_us_claims_events
from backtest.scanner_v8 import FAMILIES, backtest_family
from models import CandleSeries
from providers.yahoo import _synthetic_series


class V9SafetyTests(unittest.TestCase):
    def test_research_macros_excluded_from_live(self):
        active = active_instruments()
        for key in ("DXY", "US10Y", "US3M", "TIP", "NATGAS", "COPPER", "CORN"):
            self.assertTrue(INSTRUMENTS[key].get("research_only"))
            self.assertNotIn(key, active)

    def test_bls_datetime_parse(self):
        ts, prec = _parse_bls_datetime("Friday, January 05, 2024 08:30 AM Employment Situation")
        self.assertEqual(prec, "exact")
        self.assertIsInstance(ts, int)
        self.assertGreater(ts, 1_700_000_000)

    def test_unknown_does_not_block_gold(self):
        ctx = MacroContext(events=[], events_by_asset={}, boe_rate=[], ecb_rate=[], dxy=None, us10y=None, us3m=None, tip=None)
        self.assertTrue(gold_filter_ok(ctx, 1_700_000_000, "dxy_not_rising"))

    def test_date_only_event_skipped_for_intraday_window(self):
        ev = MacroEvent(
            event_id="x",
            name="test",
            country="US",
            category="X",
            importance="HIGH",
            ts_unix=1_700_000_000,
            time_precision="date_only",
            provenance="test",
            affected_assets=("EURUSD",),
        )
        ctx = MacroContext(
            events=[ev],
            events_by_asset={"EURUSD": [ev]},
            boe_rate=[],
            ecb_rate=[],
            dxy=None,
            us10y=None,
            us3m=None,
            tip=None,
        )
        self.assertFalse(
            ctx.in_blackout("EURUSD", 1_700_000_000, before_sec=3600, after_sec=3600)
        )
        self.assertTrue(
            ctx.in_blackout("EURUSD", 1_700_000_000, before_sec=0, after_sec=0, calendar_day=True)
        )

    def test_train_selection_ignores_final(self):
        rows = [
            {"name": "A", "is_control": False, "train_n": 40, "train_expectancy": 0.002, "train_diversified": True},
            {"name": "B", "is_control": False, "train_n": 40, "train_expectancy": 0.001, "train_diversified": True},
            {"name": "CTRL", "is_control": True, "train_n": 80, "train_expectancy": 0.01, "train_diversified": True},
        ]
        self.assertEqual(select_on_train(rows)["name"], "A")

    def test_windows_pre_specified(self):
        keys = [w["key"] for w in V9_EVENT_WINDOWS]
        self.assertEqual(keys, ["none", "30m", "1h", "2h", "4h_after", "calendar_day"])

    def test_claims_thursdays(self):
        evs = weekly_us_claims_events(datetime(2024, 1, 1), datetime(2024, 1, 31))
        self.assertTrue(evs)
        self.assertTrue(all(e.consensus == UNKNOWN and e.surprise == UNKNOWN for e in evs))

    def test_v8_base_still_runs(self):
        raw = _synthetic_series("EURUSD", "4h", n=200)
        s = CandleSeries(
            instrument="EURUSD",
            symbol="EURUSD",
            asset_class="forex",
            timeframe="4h",
            timestamps=raw.timestamps,
            open=raw.open,
            high=raw.high,
            low=raw.low,
            close=raw.close,
            volume=raw.volume,
        )
        fam = next(f for f in FAMILIES if f.key == "S1")
        self.assertIsInstance(backtest_family(s, fam), list)

    def test_spec_count_bounded(self):
        specs = build_specs()
        total = sum(len(v) for v in specs.values())
        self.assertLessEqual(total, 30)


if __name__ == "__main__":
    unittest.main()
