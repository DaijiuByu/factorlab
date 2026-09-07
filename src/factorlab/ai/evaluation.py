"""Guardrails for evaluating LLM-proposed factors without executing code."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .formula import evaluate_formula, validate_formula
from ..research import BacktestConfig, run_research


def evaluate_factor_proposal(
    panel: pd.DataFrame,
    formula: str,
    *,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    min_assets: int = 10,
) -> dict[str, Any]:
    """Validate a formula and run a deterministic diagnostic backtest.

    The returned dictionary is suitable for an AI evaluation dataset. A model
    proposal is never treated as trusted code; it must pass the fixed formula
    grammar before any numbers are computed.
    """

    normalized = validate_formula(formula)
    values = evaluate_formula(panel, normalized)
    finite_ratio = float(np.isfinite(values.to_numpy(dtype=float)).mean())
    enriched = panel.copy()
    enriched["proposal_score"] = values
    result = run_research(
        enriched,
        factor="column",
        raw_column="proposal_score",
        backtest=BacktestConfig(
            quantile=quantile, cost_bps=cost_bps, min_assets=min_assets
        ),
    )
    return {
        "formula": normalized,
        "finite_score_ratio": finite_ratio,
        "metrics": result.metrics,
        "checks": {
            "formula_allow_list": True,
            "future_data_execution": False,
            "has_oos_backtest": bool(result.metrics.get("observations")),
        },
    }
