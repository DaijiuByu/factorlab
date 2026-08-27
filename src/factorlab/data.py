"""Panel data loading, validation, and deterministic demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "ticker", "close"}


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a daily equity panel.

    The function deliberately fails early on duplicate observations and bad
    prices. Silent deduplication is dangerous in research code because it can
    change a result without leaving an audit trail.
    """

    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    clean = panel.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.tz_localize(None)
    clean["ticker"] = clean["ticker"].astype(str).str.strip()
    clean["close"] = pd.to_numeric(clean["close"], errors="raise")
    if clean["ticker"].eq("").any():
        raise ValueError("ticker must not be empty")
    if clean["close"].le(0).any() or not np.isfinite(clean["close"]).all():
        raise ValueError("close must contain finite positive values")
    if clean.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker observations are not allowed")
    clean = clean.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    return clean


def load_panel(path: str | Path) -> pd.DataFrame:
    """Load a CSV panel and apply :func:`validate_panel`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return validate_panel(pd.read_csv(source))


def generate_demo_panel(
    *,
    start: str = "2018-01-02",
    days: int = 756,
    assets: int = 40,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a deterministic panel for examples and tests.

    This is not market data. It is a synthetic sanity-check dataset with a
    weak, stable value signal and sector effects so the complete workflow can
    run without a data vendor or credentials.
    """

    if days < 30 or assets < 6:
        raise ValueError("demo data needs at least 30 days and 6 assets")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    tickers = [f"S{index:03d}" for index in range(assets)]
    sectors = np.array([f"sector_{index % 5}" for index in range(assets)])
    latent_value = rng.normal(0.0, 1.0, assets)
    market = rng.normal(0.0, 0.008, len(dates))
    rows: list[dict[str, object]] = []
    for asset_index, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0.0, 0.012, len(dates))
        predictive_component = 0.00015 * latent_value[asset_index]
        returns = market + idiosyncratic + predictive_component
        prices = 100.0 * np.exp(np.cumsum(returns))
        value_score = latent_value[asset_index] + rng.normal(0.0, 0.25, len(dates))
        volume = np.exp(rng.normal(12.0, 0.45, len(dates)))
        market_cap = prices * np.exp(rng.normal(8.0, 0.25, len(dates)))
        for date, close, traded, cap, value in zip(
            dates, prices, volume, market_cap, value_score
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": round(float(close), 6),
                    "volume": round(float(traded), 3),
                    "market_cap": round(float(cap), 3),
                    "sector": sectors[asset_index],
                    "value_score": round(float(value), 6),
                }
            )
    return validate_panel(pd.DataFrame(rows))
