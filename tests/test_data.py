import unittest

import pandas as pd

from factorlab.data import generate_demo_panel, validate_panel


class DataTests(unittest.TestCase):
    def test_demo_is_deterministic_and_sorted(self):
        first = generate_demo_panel(days=40, assets=8, seed=11)
        second = generate_demo_panel(days=40, assets=8, seed=11)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first[["date", "ticker"]].duplicated().sum() == 0)
        self.assertTrue(first["date"].is_monotonic_increasing)

    def test_validation_rejects_duplicate_and_bad_price(self):
        base = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"], "close": [10.0]})
        with self.assertRaises(ValueError):
            validate_panel(pd.concat([base, base], ignore_index=True))
        bad = base.copy()
        bad.loc[0, "close"] = 0
        with self.assertRaises(ValueError):
            validate_panel(bad)
