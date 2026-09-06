"""A thin VectorBT adapter for multi-asset target-weight backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class VectorBTResult:
    portfolio: Any
    stats: pd.Series


def target_weights_from_scores(
    panel: pd.DataFrame,
    *,
    score_column: str = "alpha_score",
    quantile: float = 0.2,
    min_assets: int = 10,
    max_position_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert point-in-time scores into next-session target weights."""

    if score_column not in panel or "close" not in panel:
        raise ValueError("panel must contain close and the score column")
    if not 0.01 <= quantile <= 0.49:
        raise ValueError("quantile must be between 0.01 and 0.49")
    if max_position_weight is not None and not 0 < max_position_weight <= 0.5:
        raise ValueError("max_position_weight must be between 0 and 0.5")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["ticker"] = frame["ticker"].astype(str)
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    prices = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date, group in frame.dropna(subset=[score_column]).groupby("date", sort=True):
        group = group.drop_duplicates("ticker")
        if len(group) < min_assets:
            continue
        count = max(1, int(len(group) * quantile))
        if count * 2 > len(group):
            continue
        ranked = group.sort_values([score_column, "ticker"], kind="stable")
        shorts = ranked.head(count)["ticker"].tolist()
        longs = ranked.tail(count)["ticker"].tolist()
        weights.loc[date, longs] = _side_weights(longs, 0.5, max_position_weight)
        weights.loc[date, shorts] = _side_weights(shorts, -0.5, max_position_weight)
    # A score observed at t is tradable from t+1 onward.
    return prices, weights.shift(1).fillna(0.0)


def _side_weights(names: list[str], gross: float, cap: float | None) -> list[float]:
    if cap is None:
        return [gross / len(names)] * len(names)
    magnitude = abs(gross)
    if cap * len(names) + 1e-12 < magnitude:
        raise ValueError(
            "max_position_weight is too small for the selected quantile portfolio"
        )
    remaining = magnitude
    result: list[float] = []
    for index in range(len(names)):
        slots = len(names) - index
        value = min(cap, remaining / slots)
        result.append(float(value if gross > 0 else -value))
        remaining -= value
    return result


def run_vectorbt(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    slippage: float = 0.0005,
    freq: str = "1D",
) -> VectorBTResult:
    """Run a VectorBT target-percent portfolio from wide price/weight tables.

    ``prices`` and ``target_weights`` must have the same datetime index and
    ticker columns. Weights are target percentages of the portfolio value.
    This function is intentionally an adapter; FactorLab's reference engine
    remains the default so the base install does not require VectorBT.
    """

    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "VectorBT is not installed; install the optional backtest dependency"
        ) from exc
    if not isinstance(prices, pd.DataFrame) or not isinstance(
        target_weights, pd.DataFrame
    ):
        raise TypeError("prices and target_weights must be pandas DataFrames")
    if prices.empty:
        raise ValueError("prices must not be empty")
    close = prices.sort_index().astype(float)
    weights = (
        target_weights.reindex(index=close.index, columns=close.columns)
        .fillna(0.0)
        .astype(float)
    )
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=weights,
        size_type="targetpercent",
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        freq=freq,
    )
    return VectorBTResult(portfolio=portfolio, stats=portfolio.stats())
