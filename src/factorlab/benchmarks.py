"""Benchmark and ablation helpers for honest factor comparisons."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .research import BacktestConfig, run_research
from .execution import ExecutionConfig


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
                "observations": result.metrics.get("observations"),
                "statistical_warning": result.metrics.get("statistical_warning"),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark_suite(
    panel: pd.DataFrame,
    *,
    factor: str = "momentum",
    lookback: int = 20,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    min_assets: int = 10,
    seed: int = 17,
) -> pd.DataFrame:
    """Run placebo, direction, market-mode and neutralization controls.

    Every row uses the same dates, costs and engine. The suite is intended for
    model-selection discipline and interview reproducibility, not performance
    cherry-picking.
    """

    variants = [
        ("factor_raw", factor, False, 1.0, "long_short"),
        ("factor_sector_neutral", factor, True, 1.0, "long_short"),
        ("factor_reversed", factor, False, -1.0, "long_short"),
        ("factor_long_only", factor, False, 1.0, "long_only"),
    ]
    rows: list[dict[str, object]] = []
    for name, selected_factor, neutral, direction, mode in variants:
        try:
            result = run_research(
                panel,
                factor=selected_factor,
                lookback=lookback,
                direction=direction,
                sector_neutral=neutral,
                backtest=BacktestConfig(
                    quantile=quantile,
                    cost_bps=cost_bps,
                    min_assets=min_assets,
                    execution=ExecutionConfig(market_mode=mode),
                ),
            )
            metrics = result.metrics
            rows.append({"variant": name, **{key: metrics.get(key) for key in (
                "observations", "total_return", "sharpe", "max_drawdown", "mean_ic", "average_turnover", "statistical_warning"
            )}})
        except (ValueError, RuntimeError) as exc:
            rows.append({"variant": name, "error": str(exc)})

    # Date-wise random placebo preserves the cross-sectional missingness shape.
    rng = np.random.default_rng(seed)
    placebo = panel.copy()
    placebo["placebo_score"] = rng.normal(size=len(placebo))
    result = run_research(
        placebo,
        factor="column",
        raw_column="placebo_score",
        backtest=BacktestConfig(quantile=quantile, cost_bps=cost_bps, min_assets=min_assets),
    )
    rows.append({"variant": "random_placebo", **{key: result.metrics.get(key) for key in (
        "observations", "total_return", "sharpe", "max_drawdown", "mean_ic", "average_turnover", "statistical_warning"
    )}})
    return pd.DataFrame(rows)
