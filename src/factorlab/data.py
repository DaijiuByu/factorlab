"""Panel data loading, validation, and deterministic demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "ticker", "close"}


def validate_point_in_time(
    panel: pd.DataFrame,
    *,
    observation_column: str = "date",
    release_column: str = "release_date",
    effective_column: str = "effective_date",
) -> pd.DataFrame:
    """Reject observations whose information was unavailable at observation time."""

    clean = validate_panel(panel)
    observation = pd.to_datetime(clean[observation_column], errors="raise").dt.tz_localize(None)
    for column, label in ((release_column, "release_date"), (effective_column, "effective_date")):
        if column not in clean:
            continue
        values = pd.to_datetime(clean[column], errors="raise").dt.tz_localize(None)
        if (values > observation).any():
            raise ValueError(f"{label} must not be after the observation date")
    return clean


def asof_universe(
    membership: pd.DataFrame,
    asof_date: str | pd.Timestamp,
    *,
    ticker_column: str = "ticker",
    effective_column: str = "effective_date",
    end_column: str = "end_date",
) -> pd.DataFrame:
    """Filter historical universe membership using point-in-time intervals."""

    required = {ticker_column, effective_column}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"membership missing columns: {', '.join(sorted(missing))}")
    point = pd.Timestamp(asof_date)
    frame = membership.copy()
    frame[effective_column] = pd.to_datetime(frame[effective_column], errors="raise")
    active = frame[effective_column] <= point
    if end_column in frame:
        frame[end_column] = pd.to_datetime(frame[end_column], errors="raise")
        active &= frame[end_column].isna() | (frame[end_column] > point)
    return frame.loc[active].drop_duplicates(ticker_column).reset_index(drop=True)


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
    if clean["date"].isna().any():
        raise ValueError("date must not be missing")
    if clean["ticker"].isna().any():
        raise ValueError("ticker must not be missing")
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
