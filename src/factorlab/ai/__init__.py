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
