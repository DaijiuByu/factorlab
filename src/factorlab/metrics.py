"""Per-asset metrics used in the live and CSV research reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_panel


METRIC_COLUMNS = ("momentum", "reversal", "volatility", "turnover_pct")


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
