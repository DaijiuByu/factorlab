import unittest

import numpy as np

from factorlab.data import generate_demo_panel
from factorlab.features import build_features, rolling_fft_features


class FeatureTests(unittest.TestCase):
    def test_fft_is_causal(self):
        panel = generate_demo_panel(days=70, assets=8, seed=31)
        first = rolling_fft_features(panel, window=16)
        changed = panel.copy()
        last = changed.index[changed["date"] == changed["date"].max()][0]
        changed.loc[last, "close"] *= 100
        second = rolling_fft_features(changed, window=16)
        before_last = panel["date"] < panel["date"].max()
        np.testing.assert_allclose(
            first.loc[before_last, "spectral_entropy"].fillna(-1),
            second.loc[before_last, "spectral_entropy"].fillna(-1),
        )

    def test_feature_builder_outputs_model_columns(self):
        panel = generate_demo_panel(days=70, assets=8, seed=32)
        features = build_features(panel, lookbacks=(5, 20), fft_window=16)
        for name in (
            "return_1d",
            "momentum_5d",
            "volatility_20d",
            "spectral_high_low_ratio",
        ):
            self.assertIn(name, features)
        self.assertEqual(len(features), len(panel))
