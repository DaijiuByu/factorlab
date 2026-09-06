"""FactorLab: a small, reproducible cross-sectional factor research toolkit."""

__version__ = "0.2.0"

from .data import generate_demo_panel, load_panel
from .experiment import ExperimentSpec, load_experiment_spec, run_experiment
from .metrics import bootstrap_mean_ci, compute_asset_metrics
from .quality import QualityConfig, QualityResult, audit_panel
from .research import ResearchResult, cost_sensitivity, run_research

__all__ = [
    "ResearchResult",
    "compute_asset_metrics",
    "bootstrap_mean_ci",
    "ExperimentSpec",
    "load_experiment_spec",
    "run_experiment",
    "QualityConfig",
    "QualityResult",
    "audit_panel",
    "generate_demo_panel",
    "load_panel",
    "run_research",
    "cost_sensitivity",
]
