import tempfile
import unittest
from pathlib import Path

import numpy as np

from factorlab.registry import FactorRecord, FactorRegistry
from factorlab.stats import benjamini_hochberg, block_bootstrap_mean_ci, deflated_sharpe_ratio


class StatisticsAndRegistryTests(unittest.TestCase):
    def test_bh_controls_sorted_q_values(self):
        result = benjamini_hochberg([0.001, 0.02, 0.5])
        self.assertTrue(np.all(np.diff(result["q_value"]) >= 0))
        self.assertTrue(bool(result.loc[0, "reject"]))

    def test_block_bootstrap_is_reproducible(self):
        values = np.arange(20, dtype=float)
        first = block_bootstrap_mean_ci(values, block_size=4, n_bootstrap=200, seed=2)
        self.assertEqual(first, block_bootstrap_mean_ci(values, block_size=4, n_bootstrap=200, seed=2))

    def test_deflated_sharpe_probability_is_bounded(self):
        value = deflated_sharpe_ratio(1.0, n_trials=10, observations=252)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_registry_rejects_duplicate_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = FactorRegistry(Path(directory) / "factors.jsonl")
            record = FactorRecord("mom", "rank(momentum(close,20))")
            registry.register(record)
            with self.assertRaises(ValueError):
                registry.register(record)
