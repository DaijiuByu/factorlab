"""Optional DeepSeek-assisted factor research."""

from .formula import FormulaError, evaluate_formula, validate_formula
from .research_assistant import DeepSeekClient, FactorProposal

__all__ = [
    "DeepSeekClient",
    "FactorProposal",
    "FormulaError",
    "evaluate_formula",
    "validate_formula",
]
from .evaluation import evaluate_factor_proposal

__all__ = ["evaluate_factor_proposal"]
