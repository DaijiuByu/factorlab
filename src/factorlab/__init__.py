"""FactorLab: a small, reproducible cross-sectional factor research toolkit."""

__version__ = "0.1.0"

from .data import generate_demo_panel, load_panel
from .metrics import compute_asset_metrics
from .research import ResearchResult, run_research

__all__ = [
    "ResearchResult",
    "compute_asset_metrics",
    "generate_demo_panel",
    "load_panel",
    "run_research",
]
