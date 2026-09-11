"""Exchange-aware execution constraints for A-share research.

The functions in this module are deliberately deterministic and data-driven.
They provide a conservative research approximation of common SSE/SZSE rules;
they are not a broker simulator and must be calibrated before production use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    """A-share execution policy used by the reference backtest."""

    market_mode: str = "long_short"
    t_plus_one: bool = True
    lot_size: int = 100
    exclude_suspended: bool = True
    exclude_limit_up_down: bool = True
    sell_tax_bps: float = 5.0
    require_shortable: bool = False
    shortable_column: str = "shortable"

    def __post_init__(self) -> None:
        if self.market_mode not in {"long_short", "long_only"}:
            raise ValueError("market_mode must be long_short or long_only")
        if not isinstance(self.lot_size, int) or self.lot_size < 1:
            raise ValueError("lot_size must be a positive integer")
        if not math.isfinite(self.sell_tax_bps) or self.sell_tax_bps < 0:
            raise ValueError("sell_tax_bps must be finite and non-negative")
        if not self.shortable_column.strip():
            raise ValueError("shortable_column must not be empty")


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "停牌", "suspended"}
    return bool(value)


def is_tradeable(row: pd.Series, *, side: str, config: ExecutionConfig) -> bool:
    """Return whether a row can be traded for ``side`` under the policy."""

    if config.exclude_suspended:
        for column in ("suspended", "is_suspended", "停牌"):
            if column in row and _truthy(row[column]):
                return False
    if config.exclude_limit_up_down:
        for column in ("limit_up", "limit_down", "涨停", "跌停"):
            if column in row and _truthy(row[column]):
                return False
    if side == "short" and config.require_shortable and config.shortable_column in row:
        return _truthy(row[config.shortable_column])
    if side == "short" and config.require_shortable and config.shortable_column not in row:
        return False
    return True


def round_lot_weight(weight: float, *, price: float, notional: float, lot_size: int) -> float:
    """Round a target weight down to exchange lot size and return the weight."""

    if not math.isfinite(weight) or not math.isfinite(price) or price <= 0 or notional <= 0:
        return 0.0
    shares = math.floor(abs(weight) * notional / price / lot_size) * lot_size
    return math.copysign(shares * price / notional, weight) if shares else 0.0


def apply_sell_tax(traded_notional: float, *, sell_tax_bps: float, is_sell: bool) -> float:
    """Return currency stamp duty for a sell trade only."""

    if not is_sell or traded_notional <= 0:
        return 0.0
    return traded_notional * sell_tax_bps / 10_000.0

