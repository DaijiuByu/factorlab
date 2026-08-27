import unittest

import numpy as np
import pandas as pd

from factorlab.data import generate_demo_panel
from factorlab.research import BacktestConfig, run_research


class ResearchTests(unittest.TestCase):
    def test_research_canonicalizes_unsorted_input(self):
        panel = generate_demo_panel(days=80, assets=12, seed=9)
        shuffled = panel.sample(frac=1.0, random_state=2).reset_index(drop=True)
        result = run_research(
            shuffled,
            factor="momentum",
            lookback=10,
            backtest=BacktestConfig(quantile=0.2, cost_bps=5, min_assets=8),
        )
        self.assertFalse(result.daily.empty)
        self.assertTrue(result.daily["date"].is_monotonic_increasing)
        self.assertIsInstance(result.daily["date"].iloc[0], pd.Timestamp)

    def test_backtest_has_no_last_day_return(self):
        panel = generate_demo_panel(days=100, assets=12, seed=5)
        result = run_research(
            panel,
            factor="momentum",
            lookback=10,
            backtest=BacktestConfig(quantile=0.25, cost_bps=5, min_assets=8),
            split_date="2018-04-01",
        )
        self.assertFalse(result.daily.empty)
        self.assertTrue((result.daily["turnover"] >= 0).all())
        self.assertEqual(
            result.weights["date"].nunique(), result.daily["date"].nunique()
        )
        self.assertIn("mean_ic", result.metrics)
        self.assertTrue(np.isfinite(result.metrics["total_return"]))

    def test_column_factor_and_costs(self):
        panel = generate_demo_panel(days=80, assets=12, seed=8)
        cheap = run_research(
            panel,
            factor="column",
            raw_column="value_score",
            backtest=BacktestConfig(quantile=0.2, cost_bps=0, min_assets=8),
        )
        expensive = run_research(
            panel,
            factor="column",
            raw_column="value_score",
            backtest=BacktestConfig(quantile=0.2, cost_bps=100, min_assets=8),
        )
        self.assertLessEqual(
            expensive.metrics["total_return"], cheap.metrics["total_return"]
        )
