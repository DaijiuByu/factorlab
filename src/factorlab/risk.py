"""Deterministic portfolio risk and turnover constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    """Simple constraints shared by reference and adapter backtests."""

    max_position_weight: float | None = None
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.0
    max_turnover: float | None = None

    def __post_init__(self) -> None:
        if self.max_position_weight is not None and not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be between 0 and 1")
        if not math.isfinite(self.max_gross_exposure) or self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if not math.isfinite(self.max_net_exposure) or self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be non-negative")
        if self.max_turnover is not None and (
            not math.isfinite(self.max_turnover) or self.max_turnover < 0
        ):
            raise ValueError("max_turnover must be finite and non-negative")


def enforce_weight_limits(
    weights: dict[str, float],
    *,
    config: RiskConfig,
    previous: dict[str, float] | None = None,
) -> dict[str, float]:
    """Apply caps, gross/net exposure and optional turnover limits.

    This is a transparent projection suitable for a reference engine. A
    production deployment can replace it with a quadratic optimizer while
    retaining the same constraint contract.
    """

    result = dict(weights)
    if config.max_position_weight is not None:
        cap = config.max_position_weight
        result = {name: max(-cap, min(cap, value)) for name, value in result.items()}
    gross = sum(abs(value) for value in result.values())
    if gross > config.max_gross_exposure and gross:
        scale = config.max_gross_exposure / gross
        result = {name: value * scale for name, value in result.items()}
    net = sum(result.values())
    if abs(net) > config.max_net_exposure and result:
        correction = (abs(net) - config.max_net_exposure) / len(result)
        sign = 1.0 if net > 0 else -1.0
        result = {name: value - sign * correction for name, value in result.items()}
        if config.max_position_weight is not None:
            cap = config.max_position_weight
            result = {name: max(-cap, min(cap, value)) for name, value in result.items()}
    if previous is not None and config.max_turnover is not None:
        names = set(previous) | set(result)
        turnover = 0.5 * sum(
            abs(result.get(name, 0.0) - previous.get(name, 0.0)) for name in names
        )
        if turnover > config.max_turnover and turnover > 0:
            fraction = config.max_turnover / turnover
            result = {
                name: previous.get(name, 0.0)
                + fraction * (result.get(name, 0.0) - previous.get(name, 0.0))
                for name in names
            }
    return {name: value for name, value in result.items() if abs(value) > 1e-15}


def shrink_covariance(
    returns: pd.DataFrame | np.ndarray,
    *,
    shrinkage: float = 0.1,
    annualization: int = 252,
) -> np.ndarray:
    """Ledoit-Wolf-style diagonal shrinkage without a heavyweight dependency."""

    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1")
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("returns must be a 2D matrix with at least two rows")
    sample = np.cov(matrix, rowvar=False, ddof=1)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])
    target = np.diag(np.diag(sample))
    return (1.0 - shrinkage) * sample * annualization + shrinkage * target * annualization


def optimize_scores(
    scores: pd.Series,
    *,
    covariance: np.ndarray | None = None,
    max_weight: float = 0.1,
    gross_exposure: float = 1.0,
    net_exposure: float = 0.0,
    risk_aversion: float = 0.0,
) -> pd.Series:
    """Project alpha scores into bounded dollar-neutral target weights.

    With ``risk_aversion`` set, a diagonalized covariance penalty shrinks large
    bets. The routine is deterministic and intentionally transparent; it is a
    reference optimizer contract for a future QP/OSQP implementation.
    """

    if scores.empty or not 0 < max_weight <= 1 or gross_exposure <= 0:
        raise ValueError("scores must be non-empty and limits must be positive")
    values = pd.to_numeric(scores, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if covariance is not None:
        matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (len(values), len(values)):
            raise ValueError("covariance shape must match scores")
        diagonal = np.clip(np.diag(matrix), 1e-12, None)
        values = values / (1.0 + max(0.0, risk_aversion) * diagonal)
    centered = values - values.mean()
    order = np.argsort(np.abs(centered))[::-1]
    weights = np.zeros(len(centered), dtype=float)
    for index in order:
        weights[index] = np.sign(centered[index]) * min(max_weight, abs(centered[index]))
    gross = np.abs(weights).sum()
    if gross > gross_exposure:
        weights *= gross_exposure / gross
    net = weights.sum()
    if abs(net) > net_exposure:
        weights -= (net - np.sign(net) * net_exposure) / len(weights)
        weights = np.clip(weights, -max_weight, max_weight)
    return pd.Series(weights, index=scores.index, name="weight")
