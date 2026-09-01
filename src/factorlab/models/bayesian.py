"""A small conjugate Bayesian linear model with coefficient shrinkage."""

from __future__ import annotations

import math

import numpy as np


class BayesianShrinkageRegressor:
    """Bayesian linear regression with a zero-mean Gaussian coefficient prior.

    The intercept is left unregularized. The model uses a closed-form posterior,
    so it is practical for rolling cross-sectional research on a CPU. It
    exposes coefficient sign probabilities and prediction uncertainty, which a
    point-estimate Ridge model does not provide.
    """

    def __init__(
        self,
        prior_precision: float = 1.0,
        noise_precision: float | None = None,
        fit_intercept: bool = True,
    ) -> None:
        if prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if noise_precision is not None and noise_precision <= 0:
            raise ValueError("noise_precision must be positive")
        self.prior_precision = float(prior_precision)
        self.noise_precision = noise_precision
        self.fit_intercept = fit_intercept

    def _prepare_x(self, x: np.ndarray, *, fit: bool = False) -> np.ndarray:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X must be a 2D array")
        if fit:
            self.feature_medians_ = np.zeros(matrix.shape[1], dtype=float)
            for column in range(matrix.shape[1]):
                finite = matrix[:, column][np.isfinite(matrix[:, column])]
                if len(finite):
                    self.feature_medians_[column] = float(np.median(finite))
        if not hasattr(self, "feature_medians_"):
            raise RuntimeError("model is not fitted")
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix = matrix.copy()
            rows, columns = np.where(missing)
            matrix[rows, columns] = self.feature_medians_[columns]
        if self.fit_intercept:
            matrix = np.column_stack([np.ones(len(matrix)), matrix])
        return matrix

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BayesianShrinkageRegressor":
        design = self._prepare_x(x, fit=True)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(design) != len(target):
            raise ValueError("X and y must have the same number of rows")
        if len(target) < design.shape[1] + 1:
            raise ValueError("not enough observations for Bayesian regression")
        if not np.isfinite(target).all():
            raise ValueError("y must contain finite values")
        prior = np.eye(design.shape[1]) * self.prior_precision
        if self.fit_intercept:
            prior[0, 0] = 0.0
        if self.noise_precision is None:
            initial = np.linalg.lstsq(design, target, rcond=None)[0]
            residual = target - design @ initial
            variance = float(np.mean(residual**2))
            self.noise_precision_ = 1.0 / max(variance, 1e-8)
        else:
            self.noise_precision_ = float(self.noise_precision)
        posterior_precision = prior + self.noise_precision_ * design.T @ design
        self.posterior_covariance_ = np.linalg.pinv(posterior_precision)
        self.coef_ = self.posterior_covariance_ @ (
            self.noise_precision_ * design.T @ target
        )
        self.posterior_std_ = np.sqrt(
            np.clip(np.diag(self.posterior_covariance_), 0.0, None)
        )
        self.n_features_in_ = x.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("model is not fitted")

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._prepare_x(x) @ self.coef_

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Return epistemic prediction standard deviation, excluding noise."""

        self._check_fitted()
        design = self._prepare_x(x)
        variance = np.einsum("ij,jk,ik->i", design, self.posterior_covariance_, design)
        return np.sqrt(np.clip(variance, 0.0, None))

    def coefficient_positive_probability(self) -> np.ndarray:
        """Posterior probability that each coefficient is positive."""

        self._check_fitted()
        safe_std = np.maximum(self.posterior_std_, 1e-12)
        return np.array(
            [
                0.5 * (1.0 + math.erf(float(mean / std) / math.sqrt(2.0)))
                for mean, std in zip(self.coef_, safe_std)
            ]
        )

    def predict_confidence_adjusted(self, x: np.ndarray) -> np.ndarray:
        """Shrink predictions when feature coefficient direction is uncertain."""

        self._check_fitted()
        design = self._prepare_x(x)
        probability = self.coefficient_positive_probability()
        sign_confidence = 2.0 * probability - 1.0
        return design @ (self.coef_ * sign_confidence)
