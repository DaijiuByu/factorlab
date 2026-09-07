"""Explicit transaction-cost models for research-grade backtests."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TransactionCostModel:
    """A transparent, additive cost model expressed in basis points.

    ``commission_bps``, ``spread_bps`` and ``slippage_bps`` apply to traded
    notional. ``impact_bps`` is the base impact at one ADV participation and
    is scaled by ``participation_rate ** impact_exponent``. Borrow is an
    annualized charge applied to short notional by the caller.
    """

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    borrow_bps_annual: float = 0.0
    impact_exponent: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.commission_bps,
            self.spread_bps,
            self.slippage_bps,
            self.impact_bps,
            self.borrow_bps_annual,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("transaction costs must be finite and non-negative")
        if not math.isfinite(self.impact_exponent) or self.impact_exponent <= 0:
            raise ValueError("impact_exponent must be positive")

    @property
    def fixed_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps

    def effective_bps(self, participation_rate: float = 0.0) -> float:
        """Return all-in traded-notional cost for a participation rate."""

        if not math.isfinite(participation_rate) or participation_rate < 0:
            raise ValueError("participation_rate must be finite and non-negative")
        impact = self.impact_bps * participation_rate**self.impact_exponent
        return self.fixed_bps + impact

    def estimate(
        self,
        traded_notional: float,
        *,
        adv: float | None = None,
        short_notional: float = 0.0,
        holding_days: float = 1.0,
    ) -> float:
        """Estimate currency cost for one rebalance."""

        if any(
            not math.isfinite(value)
            for value in (traded_notional, short_notional, holding_days)
        ) or traded_notional < 0 or short_notional < 0 or holding_days < 0:
            raise ValueError("notional and holding_days must be non-negative")
        if not math.isfinite(traded_notional):
            raise ValueError("traded_notional must be finite")
        # Without a trustworthy ADV observation we conservatively charge the
        # model's base impact (one-ADV participation) instead of silently
        # dropping impact to zero. A supplied ADV still receives the
        # participation-rate scaling.
        participation = (
            1.0
            if adv is None or not math.isfinite(adv) or adv <= 0
            else traded_notional / adv
        )
        traded = traded_notional * self.effective_bps(participation) / 10_000.0
        borrow = short_notional * self.borrow_bps_annual / 10_000.0 * holding_days / 252.0
        return float(traded + borrow)
