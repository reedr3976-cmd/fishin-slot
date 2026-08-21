"""V12 data expansion and frozen-spec tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.frozen_specs import FROZEN_V11_S_FVG_SWEEP, build_v12_experiments
from providers.data_source_audit import build_data_source_audit


class V12Tests(unittest.TestCase):
    def test_experiment_count(self):
        ex = build_v12_experiments()
        self.assertEqual(len(ex), 5)

    def test_frozen_spec_documented(self):
        self.assertEqual(FROZEN_V11_S_FVG_SWEEP["component"], "FVG_SWEEP")

    def test_audit_lists_dukascopy(self):
        audit = build_data_source_audit()
        self.assertEqual(audit["v12_selected_primary"], "Dukascopy (dukascopy-python)")
        paid = [s for s in audit["sources"] if s.get("cost") == "paid"]
        self.assertTrue(all(not s.get("integrated_v12") for s in paid))


if __name__ == "__main__":
    unittest.main()
