import unittest

import pandas as pd

from factorlab.ai.formula import FormulaError, evaluate_formula, validate_formula


class FormulaTests(unittest.TestCase):
    def test_safe_formula_is_evaluable(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
                ),
                "ticker": ["A", "B", "A", "B"],
                "close": [10.0, 20.0, 11.0, 19.0],
                "turnover_pct": [1.0, 2.0, 1.1, 1.9],
            }
        )
        expression = validate_formula("rank(reversal(close,1))")
        scores = evaluate_formula(panel, expression)
        self.assertEqual(len(scores), len(panel))
        self.assertTrue(scores.notna().any())

    def test_unsafe_formula_is_rejected(self):
        with self.assertRaises(FormulaError):
            validate_formula("__import__('os').system('whoami')")
