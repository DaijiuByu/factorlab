import unittest

import pandas as pd

from factorlab.quality import QualityConfig, audit_panel


class QualityTests(unittest.TestCase):
    def test_quality_separates_warning_and_quarantine(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "ticker": ["A", "A", "A", "A"],
                "open": [10.0, 10.0, 10.0, 20.0],
                "high": [10.0, 10.0, 10.0, 20.0],
                "low": [10.0, 10.0, 10.0, 20.0],
                "close": [10.0, 10.0, 10.0, 20.0],
                "volume": [100, 100, 100, 100],
            }
        )
        result = audit_panel(
            panel, config=QualityConfig(jump_threshold=0.5, stale_days=2)
        )
        self.assertEqual(len(result.cleaned), 4)
        self.assertIn("price_jump", set(result.summary["check"]))
        self.assertIn("stale_price", set(result.summary["check"]))
        self.assertTrue((result.summary["severity"] == "warning").all())

        strict = audit_panel(
            panel,
            config=QualityConfig(
                jump_threshold=0.5, stale_days=2, quarantine_price_jumps=True
            ),
        )
        self.assertLess(len(strict.cleaned), len(panel))
        self.assertGreater(len(strict.quarantine), 0)
