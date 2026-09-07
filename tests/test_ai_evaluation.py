import unittest

from factorlab.ai.evaluation import evaluate_factor_proposal
from factorlab.data import generate_demo_panel


class AiEvaluationTests(unittest.TestCase):
    def test_evaluation_is_allow_listed_and_structured(self):
        panel = generate_demo_panel(days=60, assets=10, seed=22)
        report = evaluate_factor_proposal(
            panel, "rank(reversal(close,5))", quantile=0.2, min_assets=8
        )
        self.assertTrue(report["checks"]["formula_allow_list"])
        self.assertIn("metrics", report)
