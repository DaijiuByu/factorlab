import unittest

import numpy as np
import pandas as pd

from factorlab.costs import TransactionCostModel
from factorlab.data import generate_demo_panel
from factorlab.metrics import bootstrap_mean_ci
from factorlab.research import BacktestConfig, cost_sensitivity, run_research


class ResearchTests(unittest.TestCase):
    def test_cost_model_uses_base_impact_without_adv(self):
        model = TransactionCostModel(impact_bps=10.0)
        self.assertAlmostEqual(model.estimate(100_000.0), 100.0)

    def test_cost_model_scales_impact_with_adv(self):
        model = TransactionCostModel(impact_bps=10.0)
        self.assertLess(
            model.estimate(1_000.0, adv=1_000_000.0),
            model.estimate(1_000.0, adv=1_000.0),
        )

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
        self.assertIn("ic_tstat_newey_west", result.metrics)
        self.assertIn("sortino", result.metrics)
        self.assertIn("calmar", result.metrics)
        self.assertFalse(result.quantile_returns.empty)
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

    def test_position_cap_preserves_side_exposure(self):
        panel = generate_demo_panel(days=80, assets=12, seed=3)
        result = run_research(
            panel,
            factor="momentum",
            lookback=5,
            backtest=BacktestConfig(
                quantile=0.2, cost_bps=5, min_assets=8, max_position_weight=0.3
            ),
        )
        totals = result.weights.groupby("date")["weight"].sum()
        self.assertTrue(np.allclose(totals.to_numpy(), 0.0))
        self.assertLessEqual(result.weights["weight"].abs().max(), 0.3 + 1e-12)

    def test_cost_sensitivity_is_monotonic_for_fixed_gross_returns(self):
        panel = generate_demo_panel(days=80, assets=12, seed=4)
        table = cost_sensitivity(
            panel,
            costs_bps=(0.0, 10.0, 50.0),
            factor="momentum",
            lookback=5,
            quantile=0.2,
            min_assets=8,
        )
        self.assertEqual(table["cost_bps"].tolist(), [0.0, 10.0, 50.0])
        self.assertTrue(table["total_return"].is_monotonic_decreasing)

    def test_split_metrics_include_turnover(self):
        panel = generate_demo_panel(days=100, assets=12, seed=5)
        result = run_research(
            panel,
            factor="momentum",
            lookback=10,
            backtest=BacktestConfig(quantile=0.25, cost_bps=5, min_assets=8),
            split_date="2018-04-01",
        )
        self.assertIsNotNone(result.split_metrics["before_split"]["average_turnover"])

    def test_bootstrap_ci_is_reproducible_and_contains_mean(self):
        values = np.array([0.01, 0.02, 0.03, 0.00, -0.01])
        interval = bootstrap_mean_ci(values, n_bootstrap=500, seed=19)
        self.assertEqual(interval, bootstrap_mean_ci(values, n_bootstrap=500, seed=19))
        self.assertLessEqual(interval[0], values.mean())
        self.assertGreaterEqual(interval[1], values.mean())
