"""Point-in-time-safe factor construction and cross-sectional transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - mean) / std


def _winsorize(
    values: pd.Series, lower: float = 0.025, upper: float = 0.975
) -> pd.Series:
    if values.dropna().empty:
        return values
    low, high = values.quantile([lower, upper])
    return values.clip(lower=low, upper=high)


def cross_sectional_score(
    values: pd.Series,
    dates: pd.Series,
    sectors: pd.Series | None = None,
    *,
    winsorize: bool = True,
) -> pd.Series:
    """Convert raw values into date-wise z-scores.

    If sectors are supplied, the raw value is demeaned inside each date/sector
    bucket before the date-wise z-score. This is a simple and transparent form
    of sector neutralization; it is not a factor model regression.
    """

    values = pd.to_numeric(values, errors="coerce")
    if sectors is not None:
        frame = pd.DataFrame({"date": dates, "value": values, "sector": sectors})
        group = frame.groupby(["date", "sector"], sort=False)["value"]
        values = values - group.transform("mean")
    if winsorize:
        values = values.groupby(dates, sort=False, group_keys=False).transform(
            _winsorize
        )
    return values.groupby(dates, sort=False, group_keys=False).transform(_zscore)


def add_factor_score(
    panel: pd.DataFrame,
    raw: pd.Series,
    *,
    name: str,
    direction: float = 1.0,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """Add a cross-sectional factor score to a validated panel."""

    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be either 1 or -1")
    sector = panel["sector"] if sector_neutral and "sector" in panel else None
    result = panel.copy()
    result[name] = cross_sectional_score(
        raw * direction,
        result["date"],
        sector,
    )
    return result


def momentum(
    panel: pd.DataFrame, lookback: int = 20, *, sector_neutral: bool = False
) -> pd.DataFrame:
    """Trailing close-to-close return, using only information through date t."""

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    raw = panel.groupby("ticker", sort=False)["close"].transform(
        lambda s: s / s.shift(lookback) - 1.0
    )
    return add_factor_score(panel, raw, name="score", sector_neutral=sector_neutral)


def reversal(panel: pd.DataFrame, *, sector_neutral: bool = False) -> pd.DataFrame:
    """Negative one-day return; positive score means recent underperformance."""

    raw = -panel.groupby("ticker", sort=False)["close"].pct_change()
    return add_factor_score(panel, raw, name="score", sector_neutral=sector_neutral)


def low_volatility(
    panel: pd.DataFrame, window: int = 20, *, sector_neutral: bool = False
) -> pd.DataFrame:
    """Negative trailing realized volatility; lower volatility ranks higher."""

    if window < 5:
        raise ValueError("window must be at least 5")
    returns = panel.groupby("ticker", sort=False)["close"].pct_change()
    raw = -returns.groupby(panel["ticker"], sort=False).transform(
        lambda s: s.rolling(window).std()
    )
    return add_factor_score(panel, raw, name="score", sector_neutral=sector_neutral)


def column_factor(
    panel: pd.DataFrame,
    column: str,
    *,
    direction: float = 1.0,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """Turn a user-provided point-in-time column into a factor score."""

    if column not in panel:
        raise ValueError(f"factor column not found: {column}")
    return add_factor_score(
        panel,
        panel[column],
        name="score",
        direction=direction,
        sector_neutral=sector_neutral,
    )
