"""Small, auditable feature set for supervised Alpha research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import validate_panel
from .spectral import rolling_fft_features


def build_features(
    panel: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...] = (5, 20, 60),
    include_fourier: bool = True,
    fft_window: int = 32,
) -> pd.DataFrame:
    """Build point-in-time features from an OHLCV panel."""

    if not lookbacks or any(value < 2 for value in lookbacks):
        raise ValueError("lookbacks must contain values of at least 2")
    result = validate_panel(panel)
    grouped = result.groupby("ticker", sort=False)
    result["return_1d"] = grouped["close"].pct_change()
    result["log_return_1d"] = grouped["close"].transform(
        lambda series: np.log(series).diff()
    )
    for lookback in lookbacks:
        result[f"momentum_{lookback}d"] = grouped["close"].transform(
            lambda series, n=lookback: series / series.shift(n) - 1.0
        )
        result[f"reversal_{lookback}d"] = -grouped["close"].transform(
            lambda series, n=lookback: series / series.shift(n) - 1.0
        )
        result[f"volatility_{lookback}d"] = (
            result["return_1d"]
            .groupby(result["ticker"], sort=False)
            .transform(
                lambda series, n=lookback: series.rolling(n).std() * np.sqrt(252)
            )
        )
        if "amount" in result:
            result[f"amount_volatility_{lookback}d"] = (
                result["amount"]
                .groupby(result["ticker"], sort=False)
                .transform(lambda series, n=lookback: series.rolling(n).std())
            )
    if "volume" in result:
        result["log_volume"] = np.log1p(result["volume"].clip(lower=0))
    if "amount" in result:
        result["log_amount"] = np.log1p(result["amount"].clip(lower=0))
        result["amihud_1d"] = result["return_1d"].abs() / result["amount"].replace(
            0, np.nan
        )
    if "turnover_pct" not in result:
        result["turnover_pct"] = np.nan
    if include_fourier:
        result = rolling_fft_features(result, window=fft_window)
    return result
