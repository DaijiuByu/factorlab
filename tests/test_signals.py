import unittest

import numpy as np

from factorlab.data import generate_demo_panel
from factorlab.research import with_forward_returns
from factorlab.signals import low_volatility, momentum


class SignalTests(unittest.TestCase):
    def test_momentum_does_not_use_future_close(self):
        panel = generate_demo_panel(days=45, assets=8, seed=2)
        scored = momentum(panel, lookback=5)
        last_date = panel["date"].max()
        self.assertTrue(scored.loc[scored["date"] == last_date, "score"].notna().all())
        labeled = with_forward_returns(scored)
        self.assertTrue(
            labeled.loc[labeled["date"] == last_date, "forward_return"].isna().all()
        )

        changed = panel.copy()
        future_row = changed.index[changed["date"] == last_date][0]
        changed.loc[future_row, "close"] *= 10
        changed_scored = momentum(changed, lookback=5)
        prior = panel["date"] < last_date
        np.testing.assert_allclose(
            scored.loc[prior, "score"].fillna(-999),
            changed_scored.loc[prior, "score"].fillna(-999),
        )

    def test_low_volatility_has_expected_score_column(self):
        panel = generate_demo_panel(days=50, assets=8, seed=4)
        scored = low_volatility(panel, window=10)
        self.assertIn("score", scored)
        self.assertGreater(scored["score"].notna().sum(), 0)
