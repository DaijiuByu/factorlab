"""Multiple-testing and dependence-aware research statistics."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: pd.Series | list[float], *, alpha: float = 0.05) -> pd.DataFrame:
    """Return BH adjusted q-values and rejection flags in original order."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    values = pd.to_numeric(pd.Series(p_values), errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 1).any():
        raise ValueError("p_values must be finite values between 0 and 1")
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for index in range(n - 1, -1, -1):
        running = min(running, ranked[index] * n / (index + 1))
        adjusted[index] = running
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    return pd.DataFrame({"p_value": values, "q_value": restored, "reject": restored <= alpha})


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Approximate probability that a Sharpe exceeds the multiple-test hurdle.

    This is a lightweight diagnostic based on the expected maximum of normal
    trials. It is intentionally labelled approximate; production research can
    replace it with a full Deflated Sharpe implementation without changing the
    report contract.
    """

    if n_trials < 1 or observations < 2:
        raise ValueError("n_trials must be positive and observations at least 2")
    if not math.isfinite(observed_sharpe):
        raise ValueError("observed_sharpe must be finite")
    volatility = math.sqrt(max(1e-12, (1.0 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2) / observations))
    hurdle = math.sqrt(2.0 * math.log(max(2, n_trials)))
    z = (observed_sharpe - hurdle) / volatility
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def block_bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    block_size: int = 5,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 7,
) -> tuple[float, float] | None:
    """Bootstrap a mean while resampling contiguous blocks."""

    if block_size < 1 or n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    sample = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(sample) < 2:
        return None
    block_size = min(block_size, len(sample))
    starts = np.arange(len(sample) - block_size + 1)
    rng = np.random.default_rng(seed)
    draws = []
    blocks_per_draw = math.ceil(len(sample) / block_size)
    for _ in range(n_bootstrap):
        chosen = rng.choice(starts, size=blocks_per_draw, replace=True)
        draw = np.concatenate([sample[start : start + block_size] for start in chosen])[: len(sample)]
        draws.append(float(draw.mean()))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))
