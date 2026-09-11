import unittest

import pandas as pd

from factorlab.data import filter_panel_by_asof_universe, generate_demo_panel
from factorlab.execution import ExecutionConfig, is_tradeable, round_lot_weight
from factorlab.research import BacktestConfig, run_research


class ExecutionTests(unittest.TestCase):
    def test_lot_rounding_and_status_filters(self):
        config = ExecutionConfig()
        self.assertEqual(round_lot_weight(0.013, price=10, notional=100_000, lot_size=100), 0.01)
        self.assertFalse(is_tradeable(pd.Series({"suspended": True}), side="long", config=config))
        self.assertFalse(is_tradeable(pd.Series({"shortable": False}), side="short", config=ExecutionConfig(require_shortable=True)))

    def test_point_in_time_membership_is_applied_per_date(self):
        panel = generate_demo_panel(days=40, assets=6)
        membership = pd.DataFrame(
            {"ticker": ["S000", "S001"], "effective_date": [panel.date.min(), panel.date.min()],
             "end_date": [panel.date.min() + pd.Timedelta(days=10)] * 2}
        )
        filtered = filter_panel_by_asof_universe(panel, membership)
        self.assertLess(filtered.date.max(), panel.date.max())
        self.assertEqual(filtered.ticker.nunique(), 2)

    def test_long_only_mode_has_non_negative_exposure(self):
        result = run_research(
            generate_demo_panel(days=70, assets=8),
            backtest=BacktestConfig(min_assets=6, execution=ExecutionConfig(market_mode="long_only")),
        )
        self.assertTrue((result.weights["weight"] >= 0).all())

