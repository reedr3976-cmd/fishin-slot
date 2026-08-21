"""V13 frozen E3 confirmation tests (including HTF look-ahead audit)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.causal_audit_e3 import audit_daily_htf_look_ahead, run_causal_audit
from backtest.frozen_e3_spec import FROZEN_E3_DOCUMENT, FROZEN_E3_SPEC, FROZEN_E3_VERSION, frozen_e3_hash
from config import ENABLED_ASSET_CLASSES


class V13Tests(unittest.TestCase):
    def test_live_universe_unchanged(self):
        self.assertEqual(ENABLED_ASSET_CLASSES, ("forex", "commodity"))

    def test_frozen_e3_component(self):
        self.assertEqual(FROZEN_E3_SPEC.component, "FVG_SWEEP_HTF")
        self.assertEqual(FROZEN_E3_DOCUMENT["component"], "FVG_SWEEP_HTF")
        self.assertTrue(FROZEN_E3_VERSION.startswith("E3-"))
        self.assertEqual(len(frozen_e3_hash()), 64)

    def test_spec_hash_stable(self):
        self.assertEqual(frozen_e3_hash(), frozen_e3_hash())

    def test_htf_look_ahead_audit_runs(self):
        result = audit_daily_htf_look_ahead()
        self.assertIn("look_ahead_detected", result)
        self.assertIn("pass", result)
        # Document detection outcome — if leakage exists, audit must flag it
        audit = run_causal_audit()
        self.assertIn("htf_look_ahead", audit)
        if result["look_ahead_detected"]:
            self.assertFalse(result["pass"])
            self.assertTrue(audit["look_ahead_detected"])


if __name__ == "__main__":
    unittest.main()
