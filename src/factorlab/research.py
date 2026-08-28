"""Factor diagnostics and a simple dollar-neutral portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import validate_panel
from .metrics import compute_asset_metrics, compute_metric_summary
from .signals import column_factor, low_volatility, momentum, reversal


@dataclass(frozen=True)
class BacktestConfig:
    quantile: float = 0.2
    cost_bps: float = 5.0
    min_assets: int = 10

    def __post_init__(self) -> None:
        if not 0.01 <= self.quantile <= 0.49:
            raise ValueError("quantile must be between 0.01 and 0.49")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.min_assets < 2:
            raise ValueError("min_assets must be at least 2")


@dataclass
class ResearchResult:
    config: dict[str, Any]
    daily: pd.DataFrame
    ic_by_date: pd.DataFrame
    metrics: dict[str, float | int | None]
    split_metrics: dict[str, dict[str, float | int | None]]
    weights: pd.DataFrame
    asset_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    market_summary: pd.DataFrame | None = None
    data_metadata: dict[str, Any] | None = None


def with_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session close return to signal-date rows.

    A score at date t is evaluated against close(t+1)/close(t)-1. The last
    observation of each ticker has no forward return and is excluded later.
    """

    result = panel.copy()
    result["forward_return"] = result.groupby("ticker", sort=False)["close"].transform(
        lambda series: series.shift(-1) / series - 1.0
    )
    return result


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def information_coefficient(scored: pd.DataFrame, min_assets: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = scored.dropna(subset=["score", "forward_return"])
    for date, group in usable.groupby("date", sort=True):
        group = group.drop_duplicates("ticker")
        if (
            len(group) < min_assets
            or group["score"].nunique() < 2
            or group["forward_return"].nunique() < 2
        ):
            continue
        rows.append(
            {
                "date": date,
                "ic": _spearman(group["score"], group["forward_return"]),
                "n_assets": len(group),
            }
        )
    return pd.DataFrame(rows, columns=["date", "ic", "n_assets"])


def _metrics(
    returns: pd.Series, turnover: pd.Series | None = None
) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "hit_rate": None,
            "average_turnover": None,
        }
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annualized_vol = (
        float(values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 else 0.0
    )
    annualized_return = float(equity.iloc[-1] ** (252 / len(values)) - 1.0)
    return {
        "observations": int(len(values)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": float(annualized_return / annualized_vol) if annualized_vol else None,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((values > 0).mean()),
        "average_turnover": float(turnover.loc[values.index].mean())
        if turnover is not None
        else None,
    }


def _portfolio(
    scored: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = scored.dropna(subset=["score", "forward_return"])
    daily_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for date, group in usable.groupby("date", sort=True):
        group = group.drop_duplicates("ticker").copy()
        if len(group) < config.min_assets:
            continue
        side_count = max(1, int(np.floor(len(group) * config.quantile)))
        if 2 * side_count > len(group):
            continue
        ranked = group.sort_values(["score", "ticker"], kind="stable")
        short_names = set(ranked.head(side_count)["ticker"])
        long_names = set(ranked.tail(side_count)["ticker"])
        long_weight = 0.5 / len(long_names)
        short_weight = -0.5 / len(short_names)
        current = {
            ticker: (long_weight if ticker in long_names else short_weight)
            for ticker in long_names | short_names
        }
        names = set(previous) | set(current)
        turnover = 0.5 * sum(
            abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names
        )
        returns = group.set_index("ticker")["forward_return"]
        gross = sum(
            current.get(ticker, 0.0) * float(returns[ticker])
            for ticker in current
            if ticker in returns
        )
        net = gross - turnover * config.cost_bps / 10_000.0
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross,
                "turnover": turnover,
                "net_return": net,
            }
        )
        weight_rows.extend(
            {"date": date, "ticker": ticker, "weight": weight}
            for ticker, weight in current.items()
        )
        previous = current
    return pd.DataFrame(daily_rows), pd.DataFrame(weight_rows)


def _split_metrics(
    daily: pd.DataFrame, split_date: str | None
) -> dict[str, dict[str, float | int | None]]:
    if not split_date or daily.empty:
        return {}
    boundary = pd.Timestamp(split_date)
    return {
        "before_split": _metrics(daily.loc[daily["date"] < boundary, "net_return"]),
        "after_split": _metrics(daily.loc[daily["date"] >= boundary, "net_return"]),
    }


def run_research(
    panel: pd.DataFrame,
    *,
    factor: str = "momentum",
    lookback: int = 20,
    raw_column: str | None = None,
    direction: float = 1.0,
    sector_neutral: bool = False,
    backtest: BacktestConfig | None = None,
    split_date: str | None = None,
    analysis_start: str | None = None,
    analysis_end: str | None = None,
    market_summary: pd.DataFrame | None = None,
    data_metadata: dict[str, Any] | None = None,
) -> ResearchResult:
    """Run factor scoring, IC analysis, and a dollar-neutral backtest."""

    panel = validate_panel(panel)
    config = backtest or BacktestConfig()
    if (
        analysis_start
        and analysis_end
        and pd.Timestamp(analysis_start) > pd.Timestamp(analysis_end)
    ):
        raise ValueError("analysis_start must not be after analysis_end")
    if factor == "momentum":
        scored = momentum(panel, lookback, sector_neutral=sector_neutral)
    elif factor == "reversal":
        scored = reversal(panel, sector_neutral=sector_neutral)
    elif factor in {"low_volatility", "low-volatility"}:
        scored = low_volatility(panel, lookback, sector_neutral=sector_neutral)
    elif factor == "column":
        if not raw_column:
            raise ValueError("raw_column is required when factor=column")
        scored = column_factor(
            panel, raw_column, direction=direction, sector_neutral=sector_neutral
        )
    else:
        raise ValueError("factor must be momentum, reversal, low_volatility, or column")
    scored = with_forward_returns(scored)
    evaluation = scored
    if analysis_start is not None:
        evaluation = evaluation.loc[evaluation["date"] >= pd.Timestamp(analysis_start)]
    if analysis_end is not None:
        evaluation = evaluation.loc[evaluation["date"] <= pd.Timestamp(analysis_end)]
    ic_by_date = information_coefficient(evaluation, config.min_assets)
    daily, weights = _portfolio(evaluation, config)
    asset_metrics = compute_asset_metrics(
        panel,
        lookback=lookback,
        start_date=analysis_start,
        end_date=analysis_end,
    )
    metric_summary = compute_metric_summary(
        panel,
        lookback=lookback,
        start_date=analysis_start,
        end_date=analysis_end,
    )
    metrics = _metrics(
        daily["net_return"] if not daily.empty else pd.Series(dtype=float),
        daily["turnover"] if not daily.empty else None,
    )
    if not ic_by_date.empty:
        ic_std = float(ic_by_date["ic"].std(ddof=1))
        metrics.update(
            {
                "mean_ic": float(ic_by_date["ic"].mean()),
                "icir": float(ic_by_date["ic"].mean() / ic_std * np.sqrt(252))
                if ic_std
                else None,
                "ic_positive_ratio": float((ic_by_date["ic"] > 0).mean()),
                "ic_observations": int(len(ic_by_date)),
            }
        )
    else:
        metrics.update(
            {
                "mean_ic": None,
                "icir": None,
                "ic_positive_ratio": None,
                "ic_observations": 0,
            }
        )
    return ResearchResult(
        config={
            "factor": factor,
            "lookback": lookback,
            "raw_column": raw_column,
            "direction": direction,
            "sector_neutral": sector_neutral,
            "quantile": config.quantile,
            "cost_bps": config.cost_bps,
            "min_assets": config.min_assets,
            "split_date": split_date,
            "analysis_start": analysis_start,
            "analysis_end": analysis_end,
        },
        daily=daily,
        ic_by_date=ic_by_date,
        metrics=metrics,
        split_metrics=_split_metrics(daily, split_date),
        weights=weights,
        asset_metrics=asset_metrics,
        metric_summary=metric_summary,
        market_summary=market_summary,
        data_metadata=data_metadata,
    )
