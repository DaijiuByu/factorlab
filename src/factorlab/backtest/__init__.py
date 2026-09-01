"""Optional adapters for established backtesting engines."""

from .vectorbt_runner import (
    VectorBTResult,
    run_vectorbt,
    target_weights_from_scores,
)

__all__ = ["VectorBTResult", "run_vectorbt", "target_weights_from_scores"]
