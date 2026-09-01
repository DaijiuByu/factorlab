"""A deliberately small, safe factor formula language.

The parser supports a useful subset of common research expressions and never
uses eval/exec. DeepSeek output is treated as untrusted input and must pass
this parser before it can be evaluated.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data import validate_panel


class FormulaError(ValueError):
    """Raised when a proposed factor formula is outside the allow-list."""


ALLOWED_COLUMNS = {"close", "open", "high", "low", "volume", "amount", "turnover_pct"}
_NUMBER = r"\d+"
_COLUMN = r"[A-Za-z_][A-Za-z0-9_]*"
_FUNC = r"(?:return|momentum|reversal|volatility)\((?:close|open|high|low|volume|amount|turnover_pct),\s*\d+\)"
_ATOM = rf"(?:{_FUNC}|{_COLUMN})"


def validate_formula(formula: str) -> str:
    """Validate and normalize a formula without evaluating it."""

    normalized = re.sub(r"\s+", " ", formula.strip())
    if not normalized or len(normalized) > 160:
        raise FormulaError("formula must be non-empty and at most 160 characters")
    if any(
        token in normalized.lower()
        for token in ("__", "eval", "exec", "import", "lambda", "future")
    ):
        raise FormulaError("formula contains a forbidden token")
    expression = normalized.replace(" ", "")
    if not re.fullmatch(
        rf"-?(?:(?:rank|zscore)\((?:-?{_ATOM}|(?:rank|zscore)\({_ATOM}\))\)|-?{_ATOM})",
        expression,
    ):
        raise FormulaError(
            "supported forms are column, -column, return(close,N), momentum(close,N), "
            "reversal(close,N), volatility(close,N), rank(expr), and zscore(expr)"
        )
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    allowed_functions = {
        "return",
        "momentum",
        "reversal",
        "volatility",
        "rank",
        "zscore",
    }
    unknown = [
        item
        for item in identifiers
        if item not in ALLOWED_COLUMNS and item not in allowed_functions
    ]
    if unknown:
        raise FormulaError(f"unknown identifier: {unknown[0]}")
    return expression


def _base_series(panel: pd.DataFrame, expression: str) -> pd.Series:
    expression = expression.strip()
    sign = 1.0
    if expression.startswith("-"):
        sign = -1.0
        expression = expression[1:]
    match = re.fullmatch(
        r"(return|momentum|reversal|volatility)\(([^,]+),(\d+)\)", expression
    )
    if match:
        function, column, lookback_text = match.groups()
        lookback = int(lookback_text)
        if column not in panel:
            raise FormulaError(f"formula column not found: {column}")
        grouped = panel.groupby("ticker", sort=False)[column]
        if function in {"return", "momentum", "reversal"}:
            values = grouped.transform(
                lambda series: series / series.shift(lookback) - 1.0
            )
            if function == "reversal":
                values = -values
        else:
            returns = panel.groupby("ticker", sort=False)["close"].pct_change()
            values = returns.groupby(panel["ticker"], sort=False).transform(
                lambda series: series.rolling(lookback).std() * np.sqrt(252)
            )
        return values * sign
    if expression not in panel:
        raise FormulaError(f"formula column not found: {expression}")
    return pd.to_numeric(panel[expression], errors="coerce") * sign


def evaluate_formula(panel: pd.DataFrame, formula: str) -> pd.Series:
    """Evaluate an allow-listed factor formula on a validated panel."""

    clean = validate_panel(panel)
    expression = validate_formula(formula)
    outer_sign = 1.0
    if expression.startswith("-"):
        outer_sign = -1.0
        expression = expression[1:]
    wrapper = None
    if expression.startswith("rank("):
        wrapper, expression = "rank", expression[5:-1]
    elif expression.startswith("zscore("):
        wrapper, expression = "zscore", expression[7:-1]
    values = _base_series(clean, expression) * outer_sign
    if wrapper == "rank":
        values = values.groupby(clean["date"], sort=False).rank(pct=True)
    elif wrapper == "zscore":
        grouped = values.groupby(clean["date"], sort=False)
        values = grouped.transform(
            lambda series: (series - series.mean()) / series.std(ddof=0)
        )
    return values
