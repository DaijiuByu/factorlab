import unittest

import numpy as np

from factorlab.data import generate_demo_panel
from factorlab.features import build_features
from factorlab.models import BayesianShrinkageRegressor, ModelConfig, walk_forward_alpha


class ModelTests(unittest.TestCase):
    def setUp(self):
        panel = generate_demo_panel(days=95, assets=10, seed=33)
        self.features = build_features(panel, lookbacks=(5, 20), fft_window=16)
        self.columns = [
            "return_1d",
            "momentum_5d",
            "momentum_20d",
            "volatility_5d",
            "spectral_entropy",
        ]

    def test_ridge_walk_forward_is_out_of_sample(self):
        result = walk_forward_alpha(
            self.features,
            self.columns,
            config=ModelConfig(
                train_days=40, test_days=10, purge_days=1, model="ridge"
            ),
        )
        self.assertGreater(len(result.predictions), 0)
        self.assertEqual(result.metrics["folds"], len(result.folds))
        self.assertTrue(np.isfinite(result.predictions["alpha_score"]).all())
        self.assertFalse(result.feature_importance.empty)

    def test_bayesian_model_returns_uncertainty_and_shrunk_alpha(self):
        model = BayesianShrinkageRegressor(prior_precision=1.0)
        x = np.array(
            [
                [0.0, 1.0, np.nan],
                [1.0, 0.0, np.nan],
                [2.0, 1.0, np.nan],
                [3.0, 0.0, np.nan],
                [4.0, 1.0, np.nan],
            ]
        )
        y = np.array([0.0, 0.2, 0.4, 0.5, 0.8])
        model.fit(x, y)
        self.assertEqual(len(model.predict(x)), 5)
        self.assertEqual(len(model.predict_std(x)), 5)
        self.assertEqual(len(model.coefficient_positive_probability()), 4)
        self.assertTrue(np.isfinite(model.predict_confidence_adjusted(x)).all())
