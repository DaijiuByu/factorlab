import unittest

import pandas as pd

from factorlab.backtest.vectorbt_runner import run_vectorbt, target_weights_from_scores


class BacktestAdapterTests(unittest.TestCase):
    def test_scores_become_next_session_target_weights(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-02",
                    ]
                ),
                "ticker": ["A", "B", "A", "B"],
                "close": [10.0, 20.0, 10.2, 19.8],
                "alpha_score": [1.0, -1.0, 1.0, -1.0],
            }
        )
        prices, weights = target_weights_from_scores(panel, quantile=0.2, min_assets=2)
        self.assertEqual(weights.iloc[0].abs().sum(), 0.0)
        self.assertAlmostEqual(weights.loc[pd.Timestamp("2024-01-02"), "A"], 0.5)
        self.assertAlmostEqual(weights.loc[pd.Timestamp("2024-01-02"), "B"], -0.5)

    def test_adapter_reports_optional_dependency(self):
        prices = pd.DataFrame(
            {"A": [10.0, 10.2, 10.1], "B": [20.0, 20.1, 20.3]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        try:
            result = run_vectorbt(prices, weights)
        except RuntimeError as error:
            self.assertIn("VectorBT", str(error))
        else:
            self.assertIsNotNone(result.stats)
