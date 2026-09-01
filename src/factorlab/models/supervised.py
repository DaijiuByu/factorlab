"""Walk-forward machine-learning Alpha research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data import validate_panel
from .bayesian import BayesianShrinkageRegressor


@dataclass(frozen=True)
class ModelConfig:
    model: str = "ridge"
    horizon: int = 1
    train_days: int = 252
    test_days: int = 21
    purge_days: int = 1
    n_jobs: int = 4
    random_state: int = 7

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.train_days < 5 or self.test_days < 1:
            raise ValueError("horizon, train_days, and test_days must be positive")
        if self.purge_days < self.horizon:
            raise ValueError("purge_days must be at least horizon")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be zero")


@dataclass
class ModelResult:
    predictions: pd.DataFrame
    metrics: dict[str, float | int | None]
    feature_importance: pd.DataFrame
    folds: pd.DataFrame
    model_config: dict[str, Any]


def build_regressor(name: str, *, random_state: int = 7, n_jobs: int = 4) -> Any:
    """Build a scikit-learn or optional third-party regressor."""

    try:
        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for model research") from exc

    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        )
    if name in {"elastic_net", "elastic-net"}:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(
                alpha=0.001, l1_ratio=0.25, max_iter=5_000, random_state=random_state
            ),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=10,
            max_features=0.7,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "bayesian_shrinkage":
        return BayesianShrinkageRegressor(prior_precision=1.0)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "install the optional lightgbm dependency to use model=lightgbm"
            ) from exc
        return LGBMRegressor(
            n_estimators=150,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=30,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "install the optional xgboost dependency to use model=xgboost"
            ) from exc
        return XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
        )
    raise ValueError(
        "model must be ridge, elastic_net, hist_gradient_boosting, random_forest, bayesian_shrinkage, lightgbm, or xgboost"
    )


def _target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    return frame.groupby("ticker", sort=False)["close"].transform(
        lambda series: series.shift(-horizon) / series - 1.0
    )


def _rank_ic(predictions: pd.DataFrame) -> float | None:
    values = predictions.dropna(subset=["alpha_score", "target"])
    daily: list[float] = []
    for _, group in values.groupby("date", sort=True):
        if (
            len(group) >= 3
            and group["alpha_score"].nunique() > 1
            and group["target"].nunique() > 1
        ):
            daily.append(
                float(group["alpha_score"].rank().corr(group["target"].rank()))
            )
    return float(np.mean(daily)) if daily else None


def _importance(model: Any, columns: list[str]) -> pd.DataFrame:
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
    elif hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_importances_"):
                values = np.asarray(step.feature_importances_, dtype=float)
                break
            if hasattr(step, "coef_"):
                values = np.abs(np.asarray(step.coef_, dtype=float).reshape(-1))
                break
    if values is None or len(values) != len(columns):
        if (
            values is not None
            and len(values) == len(columns) + 1
            and getattr(model, "fit_intercept", False)
        ):
            values = values[1:]
        else:
            return pd.DataFrame(columns=["feature", "importance"])
    if values is None or len(values) != len(columns):
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": columns, "importance": values})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def walk_forward_alpha(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    config: ModelConfig | None = None,
) -> ModelResult:
    """Train a model chronologically and return out-of-sample Alpha scores."""

    cfg = config or ModelConfig()
    clean = validate_panel(panel)
    missing = set(feature_columns) - set(clean.columns)
    if missing:
        raise ValueError(f"feature columns not found: {', '.join(sorted(missing))}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    clean["target"] = _target(clean, cfg.horizon)
    usable = clean.dropna(subset=["target"]).copy()
    dates = pd.Index(sorted(usable["date"].drop_duplicates()))
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    if len(dates) <= cfg.train_days:
        raise ValueError("not enough dates for the configured train_days")
    for test_start in range(cfg.train_days, len(dates), cfg.test_days):
        train_end = test_start - cfg.purge_days
        train_dates = dates[max(0, train_end - cfg.train_days) : train_end]
        test_dates = dates[test_start : test_start + cfg.test_days]
        train = usable.loc[usable["date"].isin(train_dates)]
        test = usable.loc[usable["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        model = build_regressor(
            cfg.model, random_state=cfg.random_state, n_jobs=cfg.n_jobs
        )
        x_train = train[feature_columns].to_numpy(dtype=float)
        y_train = train["target"].to_numpy(dtype=float)
        model.fit(x_train, y_train)
        if cfg.model == "bayesian_shrinkage":
            score = model.predict_confidence_adjusted(
                test[feature_columns].to_numpy(dtype=float)
            )
        else:
            score = np.asarray(
                model.predict(test[feature_columns].to_numpy(dtype=float)), dtype=float
            )
        fold_prediction = test[["date", "ticker", "target"]].copy()
        fold_prediction["alpha_score"] = score
        fold_prediction["fold"] = len(folds)
        predictions.append(fold_prediction)
        folds.append(
            {
                "fold": len(folds),
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        importance = _importance(model, feature_columns)
        if not importance.empty:
            importance["fold"] = len(folds) - 1
            importances.append(importance)
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=["date", "ticker", "target", "alpha_score", "fold"])
    )
    fold_frame = pd.DataFrame(folds)
    importance_frame = (
        pd.concat(importances, ignore_index=True)
        if importances
        else pd.DataFrame(columns=["feature", "importance", "fold"])
    )
    if prediction_frame.empty:
        metrics = {
            "observations": 0,
            "rank_ic": None,
            "pearson_ic": None,
            "rmse": None,
            "folds": 0,
        }
    else:
        metrics = {
            "observations": int(len(prediction_frame)),
            "rank_ic": _rank_ic(prediction_frame),
            "pearson_ic": float(
                prediction_frame["alpha_score"].corr(prediction_frame["target"])
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        (prediction_frame["alpha_score"] - prediction_frame["target"])
                        ** 2
                    )
                )
            ),
            "folds": int(len(fold_frame)),
        }
    return ModelResult(
        predictions=prediction_frame,
        metrics=metrics,
        feature_importance=importance_frame,
        folds=fold_frame,
        model_config={
            "model": cfg.model,
            "horizon": cfg.horizon,
            "train_days": cfg.train_days,
            "test_days": cfg.test_days,
            "purge_days": cfg.purge_days,
            "n_jobs": cfg.n_jobs,
        },
    )
