"""Benchmark and ablation helpers for honest factor comparisons."""

from __future__ import annotations

import pandas as pd

from .research import BacktestConfig, run_research


def compare_variants(
    panel: pd.DataFrame,
    *,
    factor: str = "momentum",
    lookback: int = 20,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Compare raw versus sector-neutral variants under identical costs."""

    rows: list[dict[str, object]] = []
    for name, neutral in (("raw", False), ("sector_neutral", True)):
        result = run_research(
            panel,
            factor=factor,
            lookback=lookback,
            sector_neutral=neutral,
            backtest=BacktestConfig(
                quantile=quantile, cost_bps=cost_bps, min_assets=min_assets
            ),
        )
        rows.append(
            {
                "variant": name,
                "mean_ic": result.metrics.get("mean_ic"),
                "total_return": result.metrics.get("total_return"),
                "sharpe": result.metrics.get("sharpe"),
                "max_drawdown": result.metrics.get("max_drawdown"),
                "average_turnover": result.metrics.get("average_turnover"),
            }
        )
    return pd.DataFrame(rows)
