"""FactorLab: a small, reproducible cross-sectional factor research toolkit."""

__version__ = "0.2.0"

from .data import asof_universe, generate_demo_panel, load_panel, validate_point_in_time
from .costs import TransactionCostModel
from .benchmarks import compare_variants
from .experiment import ExperimentSpec, load_experiment_spec, run_experiment
from .metrics import bootstrap_mean_ci, compute_asset_metrics
from .quality import QualityConfig, QualityResult, audit_panel
from .research import ResearchResult, cost_sensitivity, run_research
from .risk import RiskConfig, enforce_weight_limits, optimize_scores, shrink_covariance
from .stats import benjamini_hochberg, block_bootstrap_mean_ci, deflated_sharpe_ratio
from .storage import query_duckdb, read_dataset, write_dataset
from .registry import FactorRecord, FactorRegistry

__all__ = [
    "ResearchResult",
    "compute_asset_metrics",
    "TransactionCostModel",
    "compare_variants",
    "bootstrap_mean_ci",
    "ExperimentSpec",
    "load_experiment_spec",
    "run_experiment",
    "QualityConfig",
    "QualityResult",
    "audit_panel",
    "generate_demo_panel",
    "validate_point_in_time",
    "asof_universe",
    "load_panel",
    "run_research",
    "cost_sensitivity",
    "RiskConfig",
    "enforce_weight_limits",
    "optimize_scores",
    "shrink_covariance",
    "benjamini_hochberg",
    "block_bootstrap_mean_ci",
    "deflated_sharpe_ratio",
    "read_dataset",
    "write_dataset",
    "query_duckdb",
    "FactorRecord",
    "FactorRegistry",
]
