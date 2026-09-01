import json
import unittest
from unittest.mock import patch

from factorlab.ai.research_assistant import DeepSeekClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return self.payload


class DeepSeekTests(unittest.TestCase):
    def test_structured_factor_proposal_is_validated(self):
        response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "name": "reversal_5d",
                                    "formula": "rank(reversal(close,5))",
                                    "intuition": "short-term reversal",
                                    "required_columns": ["close"],
                                    "validation_plan": ["rank IC"],
                                    "risk_notes": ["turnover"],
                                }
                            )
                        }
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", return_value=response):
            proposal = DeepSeekClient("test-key").propose_factor(
                "find a short reversal factor", ["close"]
            )
        self.assertEqual(proposal.formula, "rank(reversal(close,5))")
        self.assertEqual(proposal.model, "deepseek-chat")

    def test_invalid_model_formula_is_rejected(self):
        response = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"name": "bad", "formula": "eval(os)"}
                            )
                        }
                    }
                ]
            }
        )
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(RuntimeError):
                DeepSeekClient("test-key").propose_factor("bad", ["close"])
