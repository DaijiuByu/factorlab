"""Configuration-driven orchestration for reproducible FactorLab studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import load_panel
from .quality import QualityConfig, audit_panel, write_quality_artifacts
from .report import write_artifacts
from .research import BacktestConfig, ResearchResult, run_research


@dataclass(frozen=True)
class ExperimentSpec:
    """Portable JSON experiment specification.

    JSON is used deliberately so a study can be replayed without adding a
    configuration dependency. Unknown keys are rejected by ``from_dict``.
    """

    input: str
    output: str = "artifacts"
    factor: str = "momentum"
    lookback: int = 20
    quantile: float = 0.2
    cost_bps: float = 5.0
    min_assets: int = 10
    max_position_weight: float | None = None
    sector_neutral: bool = False
    split_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    raw_column: str | None = None
    direction: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown experiment keys: {', '.join(sorted(unknown))}")
        try:
            spec = cls(**value)
        except TypeError as exc:
            raise ValueError(f"invalid experiment fields: {exc}") from exc
        # Reuse the same validation as the runtime engine before touching data.
        BacktestConfig(
            quantile=spec.quantile,
            cost_bps=spec.cost_bps,
            min_assets=spec.min_assets,
            max_position_weight=spec.max_position_weight,
        )
        if spec.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if spec.factor not in {"momentum", "reversal", "low_volatility", "column"}:
            raise ValueError("factor is not registered")
        if spec.factor == "column" and not spec.raw_column:
            raise ValueError("raw_column is required when factor=column")
        if not spec.input.strip():
            raise ValueError("input must not be empty")
        if spec.direction not in {-1.0, 1.0}:
            raise ValueError("direction must be -1.0 or 1.0")
        return spec


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load and validate a JSON experiment specification."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a JSON object")
    return ExperimentSpec.from_dict(value)


def run_experiment(spec: ExperimentSpec) -> ResearchResult:
    """Run quality audit and research from one validated specification."""

    panel = load_panel(spec.input)
    quality = audit_panel(panel, config=QualityConfig())
    output = Path(spec.output)
    write_quality_artifacts(quality, str(output), source=spec.input)
    result = run_research(
        quality.cleaned,
        factor=spec.factor,
        lookback=spec.lookback,
        raw_column=spec.raw_column,
        direction=spec.direction,
        sector_neutral=spec.sector_neutral,
        backtest=BacktestConfig(
            quantile=spec.quantile,
            cost_bps=spec.cost_bps,
            min_assets=spec.min_assets,
            max_position_weight=spec.max_position_weight,
        ),
        split_date=spec.split_date,
        analysis_start=spec.start_date,
        analysis_end=spec.end_date,
    )
    write_artifacts(result, output, source=spec.input)
    return result
