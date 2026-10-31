import unittest
import numpy as np
from inno_medical_hai import core


class TestHai(unittest.TestCase):
    def test_poisson_tail(self):
        self.assertAlmostEqual(core.poisson_sf(1, 2.0), 1 - np.exp(-2.0), places=6)
        self.assertEqual(core.poisson_sf(0, 3.0), 1.0)
        self.assertLess(core.poisson_sf(15, 2.0), 1e-8)

    def test_quiet_series_no_alert(self):
        rng = np.random.default_rng(0)
        self.assertEqual(core.merge(core.scan(rng.poisson(0.2, 365))), [])

    def test_injected_burst_detected(self):
        rng = np.random.default_rng(1)
        s = rng.poisson(0.2, 365); s[200:212] += 3
        al = core.merge(core.scan(s))
        self.assertTrue(any(a <= 211 and b >= 200 for a, b, _ in al))

    def test_merge_adjacent(self):
        self.assertEqual(len(core.merge([(1, 5, .1), (6, 9, .2), (50, 55, .1)])), 2)

    def test_wilson_bounds(self):
        lo, hi = core.wilson(5, 10)
        self.assertTrue(0 <= lo < 0.5 < hi <= 1)
        self.assertEqual(core.wilson(0, 0), (0.0, 1.0))
