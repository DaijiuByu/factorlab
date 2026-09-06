"""Per-asset metrics used in the live and CSV research reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_panel


METRIC_COLUMNS = ("momentum", "reversal", "volatility", "turnover_pct")


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    seed: int = 7,
) -> tuple[float, float] | None:
    """Return a reproducible percentile-bootstrap CI for a sample mean.

    The interval is descriptive rather than a replacement for a full
    block-bootstrap time-series analysis. It is useful for reporting an
    uncertainty band alongside IC and return estimates.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    sample = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(sample) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(n_bootstrap, len(sample)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def newey_west_tstat(values: pd.Series, *, lags: int | None = None) -> float | None:
    """Return a HAC/Newey-West t-statistic for a time series.

    IC observations are serially correlated in practice.  This small
    dependency-free implementation uses Bartlett weights and a conservative
    automatic lag choice, making the reported significance less optimistic
    than an IID t-statistic while keeping the calculation auditable.
    """

    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(series)
    if n < 2:
        return None
    mean = float(series.mean())
    centered = series.to_numpy() - mean
    if lags is None:
        lags = max(0, min(n - 1, int(4 * (n / 100) ** (2 / 9))))
    lags = max(0, min(int(lags), n - 1))
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    if not np.isfinite(variance) or variance <= 0:
        return None
    return float(mean / np.sqrt(variance / n))


def factor_quantile_returns(
    scored: pd.DataFrame,
    *,
    score_column: str = "score",
    return_column: str = "forward_return",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Calculate equal-weight forward returns for each daily score bucket.

    Buckets are assigned using a stable cross-sectional rank, avoiding
    ``qcut`` failures when ties are common.  The result is intentionally long
    format so it can be plotted or joined to other experiment metadata.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    required = {"date", score_column, return_column}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, object]] = []
    usable = scored.dropna(subset=[score_column, return_column]).copy()
    for date, group in usable.groupby("date", sort=True):
        if group.empty:
            continue
        ranks = group[score_column].rank(method="first", pct=True)
        bucket = np.ceil(ranks * quantiles).clip(1, quantiles).astype(int)
        for label, bucket_group in group.assign(_quantile=bucket).groupby(
            "_quantile", sort=True
        ):
            returns = pd.to_numeric(bucket_group[return_column], errors="coerce").dropna()
            if returns.empty:
                continue
            rows.append(
                {
                    "date": date,
                    "quantile": int(label),
                    "n_assets": int(len(returns)),
                    "mean_return": float(returns.mean()),
                }
            )
    return pd.DataFrame(
        rows, columns=["date", "quantile", "n_assets", "mean_return"]
    )


def compute_asset_metrics(
    panel: pd.DataFrame,
    *,
    lookback: int = 20,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one latest-observation row per asset in the selected interval.

    The features are calculated on the complete input panel first, so a live
    one-month request can include a warm-up period without leaking future
    observations into the first requested date.
    """

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    clean = validate_panel(panel)
    if (
        start_date is not None
        and end_date is not None
        and pd.Timestamp(start_date) > pd.Timestamp(end_date)
    ):
        raise ValueError("start_date must not be after end_date")

    grouped = clean.groupby("ticker", sort=False)
    clean["return_1d"] = grouped["close"].pct_change()
    clean["momentum"] = grouped["close"].transform(
        lambda series: series / series.shift(lookback) - 1.0
    )
    clean["reversal"] = -clean["return_1d"]
    clean["volatility"] = (
        clean["return_1d"]
        .groupby(clean["ticker"], sort=False)
        .transform(lambda series: series.rolling(lookback).std() * np.sqrt(252))
    )
    if "turnover_pct" in clean:
        clean["turnover_mean_pct"] = (
            clean["turnover_pct"]
            .groupby(clean["ticker"], sort=False)
            .transform(lambda series: series.rolling(lookback).mean())
        )
    else:
        clean["turnover_pct"] = np.nan
        clean["turnover_mean_pct"] = np.nan

    eligible = clean
    if start_date is not None:
        eligible = eligible.loc[eligible["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        eligible = eligible.loc[eligible["date"] <= pd.Timestamp(end_date)]
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "board",
                "asof_date",
                "close",
                "momentum",
                "reversal",
                "volatility",
                "turnover_pct",
                "turnover_mean_pct",
                "observations",
            ]
        )

    latest = (
        eligible.sort_values(["ticker", "date"], kind="stable")
        .groupby("ticker", sort=False)
        .tail(1)
    )
    observations = eligible.groupby("ticker", sort=False).size().rename("observations")
    columns = [
        "ticker",
        "date",
        "close",
        "momentum",
        "reversal",
        "volatility",
        "turnover_pct",
        "turnover_mean_pct",
    ]
    for optional in ("name", "board"):
        if optional in latest:
            columns.insert(1, optional)
    result = latest[columns].rename(columns={"date": "asof_date"}).copy()
    result["observations"] = result["ticker"].map(observations).astype("int64")
    return result.sort_values(
        ["momentum", "ticker"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def compute_metric_summary(
    panel: pd.DataFrame,
    *,
    lookback: int = 20,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate each metric across every available stock for every date."""

    clean = validate_panel(panel)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    grouped = clean.groupby("ticker", sort=False)
    clean["return_1d"] = grouped["close"].pct_change()
    clean["momentum"] = grouped["close"].transform(
        lambda series: series / series.shift(lookback) - 1.0
    )
    clean["reversal"] = -clean["return_1d"]
    clean["volatility"] = (
        clean["return_1d"]
        .groupby(clean["ticker"], sort=False)
        .transform(lambda series: series.rolling(lookback).std() * np.sqrt(252))
    )
    if "turnover_pct" not in clean:
        clean["turnover_pct"] = np.nan
    eligible = clean
    if start_date is not None:
        eligible = eligible.loc[eligible["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        eligible = eligible.loc[eligible["date"] <= pd.Timestamp(end_date)]
    if eligible.empty:
        return pd.DataFrame(
            columns=["date"]
            + [
                f"{metric}_{stat}"
                for metric in METRIC_COLUMNS
                for stat in ("mean", "median", "std", "count")
            ]
        )
    aggregations = {
        metric: ["mean", "median", "std", "count"] for metric in METRIC_COLUMNS
    }
    summary = eligible.groupby("date", sort=True).agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
