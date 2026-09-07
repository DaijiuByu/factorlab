import unittest

import numpy as np
import pandas as pd

from factorlab.risk import optimize_scores, shrink_covariance


class RiskTests(unittest.TestCase):
    def test_shrink_covariance_is_positive_semidefinite(self):
        returns = pd.DataFrame(np.random.default_rng(4).normal(size=(40, 4)))
        covariance = shrink_covariance(returns, shrinkage=0.2)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1e-10))

    def test_optimizer_is_bounded_and_neutral(self):
        scores = pd.Series([3.0, 2.0, -1.0, -4.0], index=list("ABCD"))
        weights = optimize_scores(scores, max_weight=0.2, gross_exposure=0.8)
        self.assertLessEqual(weights.abs().max(), 0.2 + 1e-12)
        self.assertAlmostEqual(float(weights.sum()), 0.0, places=10)
