import unittest

import pandas as pd

from factorlab.data import (
    asof_universe,
    generate_demo_panel,
    validate_panel,
    validate_point_in_time,
)


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

    def test_validation_rejects_missing_date_and_ticker(self):
        missing_date = pd.DataFrame({"date": [None], "ticker": ["A"], "close": [10.0]})
        with self.assertRaises(ValueError):
            validate_panel(missing_date)
        missing_ticker = pd.DataFrame(
            {"date": ["2024-01-01"], "ticker": [None], "close": [10.0]}
        )
        with self.assertRaises(ValueError):
            validate_panel(missing_ticker)

    def test_point_in_time_release_date_is_enforced(self):
        panel = generate_demo_panel(days=30, assets=6, seed=1)
        panel["release_date"] = panel["date"]
        panel.loc[0, "release_date"] = panel.loc[0, "date"] + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            validate_point_in_time(panel)

    def test_asof_universe_respects_effective_interval(self):
        membership = pd.DataFrame(
            {
                "ticker": ["A", "B", "A"],
                "effective_date": ["2020-01-01", "2020-01-01", "2021-01-01"],
                "end_date": ["2020-12-31", None, None],
            }
        )
        result = asof_universe(membership, "2020-06-01")
        self.assertEqual(set(result["ticker"]), {"A", "B"})
