# FactorLab

FactorLab is a small research toolkit for daily cross-sectional equity
signals. It is built around the parts of a backtest that are easy to get
wrong: signal timing, cross-sectional ranking, turnover, transaction costs,
and separating a research period from a later validation period.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Tests](https://img.shields.io/badge/tests-included-success)
![License](https://img.shields.io/badge/license-MIT-blue)

## What it does

Given a CSV with one row per `date` and `ticker`, FactorLab can:

- validate a price panel and reject duplicate observations;
- build momentum, short-term reversal, low-volatility, or user-column factors;
- winsorize and z-score signals within each date;
- optionally demean signals within date/sector buckets;
- calculate daily Spearman IC, mean IC, ICIR, and positive-IC ratio;
- form a dollar-neutral top/bottom quantile portfolio;
- charge explicit turnover-based transaction costs;
- report annualized return, volatility, Sharpe, drawdown, hit rate, and turnover;
- write CSV/JSON/Markdown outputs and an equity curve PNG.

The default workflow does not need a vendor account or network access.

## Timing convention

The central convention is:

```text
score(t) -> close(t+1) / close(t) - 1
```

Trailing features use data through date `t`. The forward return is only an
evaluation label and is not used to construct the score. The last row for each
ticker has no next-session return and is dropped from diagnostics and the
portfolio. This convention is simple, visible in the code, and testable.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate       # Windows
# source .venv/bin/activate  # macOS/Linux
python -m pip install -e .

factorlab demo --output demo_panel.csv
factorlab analyze \
  --input demo_panel.csv \
  --output artifacts \
  --factor momentum \
  --lookback 20 \
  --quantile 0.2 \
  --cost-bps 5 \
  --min-assets 20 \
  --split-date 2020-01-01
```

The output directory contains:

```text
artifacts/
├── daily_returns.csv
├── equity_curve.png
├── ic_by_date.csv
├── metrics.json
├── report.md
└── weights.csv
```

For a provided point-in-time factor column:

```bash
factorlab analyze \
  --input your_panel.csv \
  --factor column \
  --column earnings_yield \
  --sector-neutral
```

Required input columns are `date`, `ticker`, and positive `close`. Optional
columns such as `sector`, `market_cap`, and `earnings_yield` are left to the
researcher. The input data should already be point-in-time aligned; FactorLab
cannot repair survivorship bias, delistings, stale fundamentals, or corporate
action errors for you.

## Research notes

This is intentionally a research skeleton rather than a production trading
engine. It does not include order-book simulation, borrow fees, exchange
calendars, corporate-action adjustment, portfolio constraints, or live trading.
The synthetic dataset is only a plumbing check and should not be interpreted as
evidence of a profitable strategy.

The backtest uses a gross exposure of 1.0: long weights sum to 0.5 and short
weights sum to -0.5. Turnover is `0.5 × sum(abs(current - previous))`, with the
initial portfolio compared to zero weights. Net return is gross return minus
turnover times `cost_bps / 10,000`.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
ruff check src tests examples
```

The tests include a future-data mutation check: changing the final close for a
ticker must not change any earlier signal.

The synthetic example deliberately has no claim of predictive power. It exists
so the code path is easy to run in a fresh checkout.

## License

MIT
