"""FactorLab: a small, reproducible cross-sectional factor research toolkit."""

__version__ = "0.1.0"

from .data import generate_demo_panel, load_panel
from .research import ResearchResult, run_research

__all__ = ["ResearchResult", "generate_demo_panel", "load_panel", "run_research"]
