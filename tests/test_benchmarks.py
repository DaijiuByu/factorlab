import unittest

from factorlab.benchmarks import compare_variants
from factorlab.data import generate_demo_panel


class BenchmarkTests(unittest.TestCase):
    def test_compare_variants_returns_comparable_rows(self):
        panel = generate_demo_panel(days=60, assets=10, seed=33)
        result = compare_variants(panel, lookback=5, quantile=0.2, min_assets=8)
        self.assertEqual(result["variant"].tolist(), ["raw", "sector_neutral"])
        self.assertIn("average_turnover", result)
