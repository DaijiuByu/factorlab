"""Causal rolling Fourier features for price and return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import validate_panel


SPECTRAL_COLUMNS = (
    "spectral_low_energy",
    "spectral_high_energy",
    "spectral_high_low_ratio",
    "spectral_entropy",
    "spectral_dominant_period",
)


def _window_features(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 4:
        return (np.nan,) * 5
    x = values - np.linspace(values[0], values[-1], len(values))
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    if len(spectrum) <= 2:
        return (np.nan,) * 5
    spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total <= 0:
        return (0.0, 0.0, np.nan, 0.0, np.nan)
    frequencies = np.arange(len(spectrum))
    low_cut = max(1, len(spectrum) // 4)
    low_energy = float(spectrum[1:low_cut].sum() / total)
    high_energy = float(spectrum[low_cut:].sum() / total)
    probability = spectrum[1:] / total
    entropy = float(
        -(probability[probability > 0] * np.log(probability[probability > 0])).sum()
        / np.log(len(probability))
    )
    dominant_index = int(frequencies[1:][np.argmax(spectrum[1:])])
    dominant_period = float(len(values) / dominant_index) if dominant_index else np.nan
    return (
        low_energy,
        high_energy,
        high_energy / low_energy if low_energy else np.nan,
        entropy,
        dominant_period,
    )


def rolling_fft_features(
    panel: pd.DataFrame,
    *,
    window: int = 32,
    price_column: str = "close",
) -> pd.DataFrame:
    """Append causal rolling FFT features aligned to each ticker/date.

    The window contains log returns through the current observation. No future
    close is read, so the resulting features can be used to predict t+1.
    """

    if window < 8:
        raise ValueError("FFT window must be at least 8")
    if price_column not in panel:
        raise ValueError(f"price column not found: {price_column}")
    result = validate_panel(panel)
    feature_values = {
        column: np.full(len(result), np.nan) for column in SPECTRAL_COLUMNS
    }
    for _, group in result.groupby("ticker", sort=False):
        indices = group.index.to_numpy()
        log_returns = np.log(group[price_column].to_numpy(dtype=float))
        log_returns = np.diff(log_returns, prepend=np.nan)
        for local_index in range(window, len(indices)):
            values = log_returns[local_index - window + 1 : local_index + 1]
            features = _window_features(values)
            for column, value in zip(SPECTRAL_COLUMNS, features):
                feature_values[column][indices[local_index]] = value
    for column, values in feature_values.items():
        result[column] = values
    return result
