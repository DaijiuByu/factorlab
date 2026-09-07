"""Factor diagnostics and a simple dollar-neutral portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .data import validate_panel
from .costs import TransactionCostModel
from .metrics import (
    compute_asset_metrics,
    compute_metric_summary,
    factor_quantile_returns,
    bootstrap_mean_ci,
    newey_west_tstat,
)
from .provenance import build_run_manifest
from .risk import RiskConfig, enforce_weight_limits, optimize_scores
from .stats import benjamini_hochberg, block_bootstrap_mean_ci, deflated_sharpe_ratio
from .signals import column_factor, low_volatility, momentum, reversal


@dataclass(frozen=True)
class BacktestConfig:
    quantile: float = 0.2
    cost_bps: float = 5.0
    min_assets: int = 10
    max_position_weight: float | None = None
    max_turnover: float | None = None
    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    borrow_bps_annual: float = 0.0
    research_trials: int = 1
    portfolio_notional: float = 1_000_000.0
    impact_exponent: float = 0.5
    adv_window: int = 20
    optimizer_risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        if not 0.01 <= self.quantile <= 0.49:
            raise ValueError("quantile must be between 0.01 and 0.49")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.min_assets < 2:
            raise ValueError("min_assets must be at least 2")
        if self.max_position_weight is not None and not 0 < self.max_position_weight <= 0.5:
            raise ValueError("max_position_weight must be between 0 and 0.5")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative")
        TransactionCostModel(
            commission_bps=self.commission_bps,
            spread_bps=self.spread_bps,
            slippage_bps=self.slippage_bps,
            impact_bps=self.impact_bps,
            borrow_bps_annual=self.borrow_bps_annual,
            impact_exponent=self.impact_exponent,
        )
        if self.research_trials < 1:
            raise ValueError("research_trials must be positive")
        if not np.isfinite(self.portfolio_notional) or self.portfolio_notional <= 0:
            raise ValueError("portfolio_notional must be finite and positive")
        if not isinstance(self.adv_window, int) or self.adv_window < 1:
            raise ValueError("adv_window must be a positive integer")
        if not np.isfinite(self.optimizer_risk_aversion) or self.optimizer_risk_aversion < 0:
            raise ValueError("optimizer_risk_aversion must be finite and non-negative")

    @property
    def cost_model(self) -> TransactionCostModel:
        return TransactionCostModel(
            commission_bps=self.commission_bps,
            spread_bps=self.spread_bps,
            slippage_bps=self.slippage_bps,
            impact_bps=self.impact_bps,
            borrow_bps_annual=self.borrow_bps_annual,
            impact_exponent=self.impact_exponent,
        )

    @property
    def effective_cost_bps(self) -> float:
        return self.cost_bps + self.cost_model.fixed_bps


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
    quantile_returns: pd.DataFrame
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
                "p_value": _correlation_p_value(
                    _spearman(group["score"], group["forward_return"]), len(group)
                ),
            }
        )
    return pd.DataFrame(rows, columns=["date", "ic", "n_assets", "p_value"])


def _correlation_p_value(correlation: float, observations: int) -> float | None:
    """Approximate two-sided p-value for a rank correlation."""

    if observations < 3 or not np.isfinite(correlation) or abs(correlation) >= 1:
        return 0.0 if abs(correlation) >= 1 else None
    statistic = abs(correlation) * np.sqrt((observations - 2) / max(1e-12, 1 - correlation**2))
    # Normal-tail approximation is deterministic and dependency-free.
    return float(math.erfc(statistic / np.sqrt(2.0)))


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
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "hit_rate": None,
            "average_turnover": None,
        }
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annualized_vol = (
        float(values.std(ddof=1) * np.sqrt(252)) if len(values) > 1 else 0.0
    )
    downside = values.where(values < 0.0, 0.0)
    downside_vol = float(downside.pow(2).mean() ** 0.5 * np.sqrt(252))
    annualized_return = float(equity.iloc[-1] ** (252 / len(values)) - 1.0)
    max_drawdown = float(drawdown.min())
    return {
        "observations": int(len(values)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": float(annualized_return / annualized_vol) if annualized_vol else None,
        "sortino": float(annualized_return / downside_vol) if downside_vol else None,
        "max_drawdown": max_drawdown,
        "calmar": float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else None,
        "hit_rate": float((values > 0).mean()),
        "average_turnover": float(turnover.loc[values.index].mean())
        if turnover is not None
        else None,
    }


def _portfolio(
    scored: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = scored.dropna(subset=["score", "forward_return"]).copy()
    # ADV is optional. When available, use a trailing mean of traded notional
    # to convert each weight change into a participation rate. This keeps the
    # impact model explicit while preserving a deterministic fallback for CSVs
    # that only contain date/ticker/close.
    adv_source = None
    if "amount" in usable.columns:
        adv_source = pd.to_numeric(usable["amount"], errors="coerce")
    elif "volume" in usable.columns:
        adv_source = pd.to_numeric(usable["volume"], errors="coerce") * pd.to_numeric(
            usable["close"], errors="coerce"
        )
    if adv_source is not None:
        usable["_adv_notional"] = (
            adv_source.where(np.isfinite(adv_source) & (adv_source > 0))
            .groupby(usable["ticker"])
            .transform(
                lambda series: series.rolling(
                    config.adv_window, min_periods=1
                ).mean()
            )
        )
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
        long_weights = _capped_side_weights(
            sorted(long_names), 0.5, config.max_position_weight
        )
        short_weights = _capped_side_weights(
            sorted(short_names), -0.5, config.max_position_weight
        )
        current = {
            **long_weights,
            **short_weights,
        }
        if config.optimizer_risk_aversion > 0:
            selected = group[group["ticker"].isin(current)].set_index("ticker")["score"]
            optimized = optimize_scores(
                selected,
                max_weight=config.max_position_weight or 0.5,
                gross_exposure=1.0,
                net_exposure=0.0,
                risk_aversion=config.optimizer_risk_aversion,
            )
            current = optimized.to_dict()
        current = enforce_weight_limits(
            current,
            config=RiskConfig(
                max_position_weight=config.max_position_weight,
                max_gross_exposure=1.0,
                max_net_exposure=0.0,
                max_turnover=config.max_turnover,
            ),
            previous=previous,
        )
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
        # ``cost_bps`` is the legacy all-in flat assumption. The explicit
        # TransactionCostModel is applied per name so market impact can scale
        # with ADV participation when the panel provides amount/volume data.
        notional = config.portfolio_notional
        adv = (
            group.set_index("ticker")["_adv_notional"]
            if "_adv_notional" in group
            else pd.Series(dtype=float)
        )
        model_cost = 0.0
        for name in names:
            change = abs(current.get(name, 0.0) - previous.get(name, 0.0))
            if change == 0.0:
                continue
            adv_value = adv.get(name)
            model_cost += config.cost_model.estimate(
                change * notional,
                adv=float(adv_value) if adv_value is not None and np.isfinite(adv_value) else None,
            )
        model_cost_return = model_cost / notional
        legacy_cost_return = turnover * config.cost_bps / 10_000.0
        short_notional = sum(-weight for weight in current.values() if weight < 0)
        borrow_cost = config.cost_model.estimate(
            0.0,
            short_notional=short_notional * notional,
            holding_days=1.0,
        )
        borrow_cost_return = borrow_cost / notional
        net = gross - legacy_cost_return - model_cost_return - borrow_cost_return
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross,
                "turnover": turnover,
                "legacy_cost": legacy_cost_return,
                "transaction_cost": model_cost_return,
                "borrow_cost": borrow_cost_return,
                "net_return": net,
            }
        )
        weight_rows.extend(
            {"date": date, "ticker": ticker, "weight": weight}
            for ticker, weight in current.items()
        )
        previous = current
    return pd.DataFrame(daily_rows), pd.DataFrame(weight_rows)


def _capped_side_weights(
    names: list[str], gross: float, cap: float | None
) -> dict[str, float]:
    """Allocate one side of a portfolio with an optional per-name cap.

    The deterministic water-filling allocation keeps the side fully invested
    whenever the requested cap is feasible (``cap * len(names) >= abs(gross)``).
    """

    if not names:
        return {}
    magnitude = abs(gross)
    if cap is None:
        weight = gross / len(names)
        return {name: weight for name in names}
    if cap * len(names) + 1e-12 < magnitude:
        raise ValueError(
            "max_position_weight is too small for the selected quantile portfolio"
        )
    remaining = magnitude
    active = list(names)
    weights: dict[str, float] = {}
    while active:
        proposed = remaining / len(active)
        if proposed <= cap + 1e-12:
            signed = np.sign(gross) * proposed
            weights.update({name: float(signed) for name in active})
            break
        name = active.pop(0)
        weights[name] = float(np.sign(gross) * cap)
        remaining -= cap
    return weights


def _split_metrics(
    daily: pd.DataFrame, split_date: str | None
) -> dict[str, dict[str, float | int | None]]:
    if not split_date or daily.empty:
        return {}
    boundary = pd.Timestamp(split_date)
    return {
        "before_split": _metrics(
            daily.loc[daily["date"] < boundary, "net_return"],
            daily.loc[daily["date"] < boundary, "turnover"],
        ),
        "after_split": _metrics(
            daily.loc[daily["date"] >= boundary, "net_return"],
            daily.loc[daily["date"] >= boundary, "turnover"],
        ),
    }


def cost_sensitivity(
    panel: pd.DataFrame,
    *,
    costs_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0),
    factor: str = "momentum",
    lookback: int = 20,
    raw_column: str | None = None,
    direction: float = 1.0,
    sector_neutral: bool = False,
    quantile: float = 0.2,
    min_assets: int = 10,
    max_position_weight: float | None = None,
    max_turnover: float | None = None,
    commission_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    impact_bps: float = 0.0,
    borrow_bps_annual: float = 0.0,
    portfolio_notional: float = 1_000_000.0,
    impact_exponent: float = 0.5,
    adv_window: int = 20,
    optimizer_risk_aversion: float = 0.0,
) -> pd.DataFrame:
    """Measure how explicit transaction-cost assumptions change results.

    This helper intentionally reuses the reference engine for each scenario,
    making the comparison directly attributable to the cost assumption.
    """

    if not costs_bps:
        raise ValueError("costs_bps must not be empty")
    rows: list[dict[str, float | int | None]] = []
    for cost in costs_bps:
        if cost < 0 or not np.isfinite(cost):
            raise ValueError("costs_bps must contain finite non-negative values")
        result = run_research(
            panel,
            factor=factor,
            lookback=lookback,
            raw_column=raw_column,
            direction=direction,
            sector_neutral=sector_neutral,
            backtest=BacktestConfig(
                quantile=quantile,
                cost_bps=float(cost),
                min_assets=min_assets,
                max_position_weight=max_position_weight,
                max_turnover=max_turnover,
                commission_bps=commission_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                impact_bps=impact_bps,
                borrow_bps_annual=borrow_bps_annual,
                portfolio_notional=portfolio_notional,
                impact_exponent=impact_exponent,
                adv_window=adv_window,
                optimizer_risk_aversion=optimizer_risk_aversion,
            ),
        )
        rows.append(
            {
                "cost_bps": float(cost),
                "total_return": result.metrics["total_return"],
                "annualized_return": result.metrics["annualized_return"],
                "sharpe": result.metrics["sharpe"],
                "average_turnover": result.metrics["average_turnover"],
            }
        )
    return pd.DataFrame(rows)


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
    if not ic_by_date.empty and "p_value" in ic_by_date:
        corrected = benjamini_hochberg(ic_by_date["p_value"])
        ic_by_date["q_value"] = corrected["q_value"].to_numpy()
        ic_by_date["q_value_reject_05"] = corrected["reject"].to_numpy()
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
                "ic_tstat_newey_west": newey_west_tstat(ic_by_date["ic"]),
            }
        )
        ic_ci = bootstrap_mean_ci(ic_by_date["ic"])
        metrics["mean_ic_ci_low"] = ic_ci[0] if ic_ci else None
        metrics["mean_ic_ci_high"] = ic_ci[1] if ic_ci else None
        block_ci = block_bootstrap_mean_ci(ic_by_date["ic"], block_size=5)
        metrics["mean_ic_block_ci_low"] = block_ci[0] if block_ci else None
        metrics["mean_ic_block_ci_high"] = block_ci[1] if block_ci else None
    else:
        metrics.update(
            {
                "mean_ic": None,
                "icir": None,
                "ic_positive_ratio": None,
                "ic_observations": 0,
                "ic_tstat_newey_west": None,
                "mean_ic_ci_low": None,
                "mean_ic_ci_high": None,
                "mean_ic_block_ci_low": None,
                "mean_ic_block_ci_high": None,
            }
        )
    if metrics.get("sharpe") is not None:
        metrics["deflated_sharpe_probability"] = deflated_sharpe_ratio(
            float(metrics["sharpe"]),
            n_trials=config.research_trials,
            observations=max(2, int(metrics["observations"])),
        )
    else:
        metrics["deflated_sharpe_probability"] = None
    quantile_returns = factor_quantile_returns(evaluation, quantiles=5)
    run_config = {
        "factor": factor,
        "lookback": lookback,
        "raw_column": raw_column,
        "direction": direction,
        "sector_neutral": sector_neutral,
        "quantile": config.quantile,
        "cost_bps": config.cost_bps,
        "min_assets": config.min_assets,
        "max_position_weight": config.max_position_weight,
        "max_turnover": config.max_turnover,
        "commission_bps": config.commission_bps,
        "spread_bps": config.spread_bps,
        "slippage_bps": config.slippage_bps,
        "impact_bps": config.impact_bps,
        "borrow_bps_annual": config.borrow_bps_annual,
        "research_trials": config.research_trials,
        "portfolio_notional": config.portfolio_notional,
        "impact_exponent": config.impact_exponent,
        "adv_window": config.adv_window,
        "optimizer_risk_aversion": config.optimizer_risk_aversion,
        "split_date": split_date,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
    }
    metadata = dict(data_metadata or {})
    metadata["run_manifest"] = build_run_manifest(
        panel,
        config=run_config,
        source="run_research",
        data_metadata=data_metadata,
    )
    return ResearchResult(
        config=run_config,
        daily=daily,
        ic_by_date=ic_by_date,
        metrics=metrics,
        split_metrics=_split_metrics(daily, split_date),
        weights=weights,
        asset_metrics=asset_metrics,
        metric_summary=metric_summary,
        quantile_returns=quantile_returns,
        market_summary=market_summary,
        data_metadata=metadata,
    )
