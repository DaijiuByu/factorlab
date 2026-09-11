import unittest

from factorlab.benchmarks import compare_variants, run_benchmark_suite
from factorlab.data import generate_demo_panel


class BenchmarkTests(unittest.TestCase):
    def test_compare_variants_returns_comparable_rows(self):
        panel = generate_demo_panel(days=60, assets=10, seed=33)
        result = compare_variants(panel, lookback=5, quantile=0.2, min_assets=8)
        self.assertEqual(result["variant"].tolist(), ["raw", "sector_neutral"])
        self.assertIn("average_turnover", result)

    def test_benchmark_suite_contains_placebo_and_controls(self):
        table = run_benchmark_suite(generate_demo_panel(days=80, assets=10), min_assets=6)
        self.assertIn("random_placebo", set(table["variant"]))
        self.assertIn("factor_long_only", set(table["variant"]))
