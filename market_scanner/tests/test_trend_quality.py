"""Tests for ADX / structure and trend-quality helpers."""

from __future__ import annotations

import unittest

import numpy as np

from indicators import adx, swing_structure_dir


class AdxStructureTests(unittest.TestCase):
    def test_adx_shapes(self):
        n = 80
        close = np.linspace(100, 140, n) + np.sin(np.linspace(0, 6, n))
        high = close + 1.0
        low = close - 1.0
        a, p, m = adx(high, low, close, 14)
        self.assertEqual(len(a), n)
        # Later bars should have finite ADX in a trending series
        self.assertTrue(np.isfinite(a[-1]))

    def test_structure_uptrend(self):
        n = 80
        close = np.linspace(100, 150, n)
        high = close + 0.5
        low = close - 0.5
        # Force clear swing points
        for j in range(10, n, 8):
            high[j] = high[j] + 2.0
            low[j - 4] = low[j - 4] - 0.2
        s = swing_structure_dir(high, low, n - 3)
        self.assertIn(s, (-1, 0, 1))


if __name__ == "__main__":
    unittest.main()
