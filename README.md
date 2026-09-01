# FactorLab

FactorLab is a small research toolkit for daily cross-sectional equity
signals. It is built around the parts of a backtest that are easy to get
wrong: signal timing, cross-sectional ranking, turnover, transaction costs,
and separating a research period from a later validation period. It has both a
local CSV mode and a live AkShare mode for Shanghai-listed A shares.

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

## Research workflow

FactorLab 0.2 adds a small set of research-layer commands around the original
factor and portfolio analysis:

```text
CSV / AkShare -> quality audit -> features -> walk-forward Alpha
                                      -> formula review -> backtest -> plots
```

- `quality`: separates hard-clean rows, warnings, and quarantine rows.
- `features`: builds point-in-time price/liquidity features and causal rolling
  Fourier features.
- `model`: trains a chronological Ridge, Elastic Net, tree, Bayesian shrinkage,
  LightGBM, or XGBoost regressor and writes out-of-sample Alpha scores.
- `model-backtest`: feeds those out-of-sample scores into the existing
  cost-aware long/short backtest.
- `ai propose`: optionally asks DeepSeek for one JSON factor proposal.
- `ai validate`: validates and evaluates the proposal using a small formula
  grammar with no `eval` or generated code execution.

The default models are deliberately CPU-friendly. LightGBM, XGBoost, VectorBT,
Polars, DuckDB, and PyMC remain optional extras so a basic CSV workflow stays
small and deterministic.

The live workflow uses three AkShare calls:

- `stock_sse_summary()` for the latest Shanghai Stock Exchange market overview;
- `stock_info_sh_name_code()` for the current SSE main-board and STAR-board
  stock universe;
- `stock_zh_a_hist()` for daily history for each symbol.

`stock_sse_summary()` is a latest-day snapshot, not a historical time series.
That is why FactorLab uses it as market context and obtains the one-month
per-stock history from `stock_zh_a_hist()`.

The interface references are the [AKShare stock documentation](https://akshare.akfamily.xyz/data/stock/stock.html)
and the [AKShare source repository](https://github.com/akfamily/akshare). The
Shanghai exchange page used by the summary interface is
http://www.sse.com.cn/market/stockdata/statistic/.

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

Audit and build features:

```bash
factorlab quality --input demo_panel.csv --output quality_artifacts
factorlab features --input demo_panel.csv --output features.csv --lookbacks 5,20,60 --fft-window 32
```

Train an out-of-sample Alpha model and backtest its predictions:

```bash
factorlab model \
  --input demo_panel.csv \
  --output model_artifacts \
  --model bayesian_shrinkage \
  --train-days 252 \
  --test-days 21 \
  --purge-days 1

factorlab model-backtest \
  --input demo_panel.csv \
  --predictions model_artifacts/predictions.csv \
  --output model_backtest_artifacts
```

The model command uses chronological folds. The score on date `t` is trained
only with dates before the test block, and the label is a later return. A model
prediction is treated as a candidate factor, not as proof of an exploitable
strategy.

Optional third-party extras:

```bash
python -m pip install -e ".[backtest,boosting]"
```

The project keeps a small reference backtester as the default. To cross-check
the same model predictions with VectorBT:

```bash
python -m pip install -e ".[backtest]"
factorlab model-backtest \
  --input demo_panel.csv \
  --predictions model_artifacts/predictions.csv \
  --engine vectorbt \
  --output vectorbt_artifacts
```

The VectorBT adapter shifts a score from `t` to the next available session
before creating target weights. This makes the framework comparison use the
same signal timing as the reference engine.

Ask DeepSeek for a reviewable factor idea:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
factorlab ai propose `
  --question "设计一个适合沪市股票的短期反转因子" `
  --output proposal.json

factorlab ai validate `
  --formula "rank(reversal(close,5))" `
  --input demo_panel.csv `
  --output proposal_scores.csv
```

The API key is read only from the environment. The assistant returns metadata
and a formula from the safe allow-list; it cannot access the repository, write
code, or place trades. DeepSeek's JSON output mode is documented at
https://api-docs.deepseek.com/guides/json_mode/.

## Use real SSE data

Install the project normally; `akshare` is included as a dependency:

```bash
python -m pip install -e .
```

Fetch the most recent calendar month and analyze the full current SSE A-share
universe:

```bash
factorlab live \
  --output artifacts_live \
  --factor momentum \
  --lookback 20 \
  --quantile 0.2 \
  --cost-bps 5 \
  --min-assets 50
```

For a smaller first run, limit the number of symbols. This is useful for
checking network access and the output format before requesting roughly two
thousand individual histories:

```bash
factorlab live \
  --start-date 2026-07-28 \
  --end-date 2026-08-28 \
  --max-stocks 30 \
  --sleep 0.2 \
  --retries 2
```

The live command keeps a per-symbol cache in `.factorlab_cache/`. A second run
with the same or a contained date window reuses completed histories. Failed
symbols are not silently discarded: they are listed in
`artifacts_live/fetch_errors.csv`.

Live output additionally contains:

```text
artifacts_live/
├── asset_metrics.csv       # latest selected-window metrics per ticker
├── asset_metrics_all_stocks.png
├── asset_metrics_heatmap.png
├── data_metadata.json      # dates, interfaces, universe and success counts
├── factor_diagnostics.png
├── fetch_errors.csv        # symbols that failed after retries
├── metric_summary.csv
├── metric_timeseries_all_stocks.png
├── portfolio_turnover_drawdown.png
├── sse_summary.csv         # latest stock_sse_summary snapshot
├── universe.csv            # symbols requested
└── report.md
```

`asset_metrics.csv` includes the latest available values for momentum,
reversal, annualized realized volatility, latest turnover percentage, and
lookback-average turnover. The report also includes IC and portfolio metrics
for the selected interval.

The visualization outputs are multi-stock diagnostics. Each metric panel plots
every selected ticker, the heatmap has one row per ticker, and the time-series
figure shows cross-sectional means and medians across all available stocks.
Ticker labels are sampled when the universe is large for readability, but all
observations remain in the plots and CSV outputs.

The latest market snapshot is taken after calling `stock_sse_summary()` once.
The exchange documents that this interface returns the most recent trading-day
overview and that same-day figures may only appear after the close; therefore a
run during market hours can legitimately show the previous trading day.

The output directory contains:

```text
artifacts/
├── asset_metrics_all_stocks.png
├── asset_metrics_heatmap.png
├── daily_returns.csv
├── equity_curve.png
├── factor_diagnostics.png
├── ic_by_date.csv
├── metric_summary.csv
├── metric_timeseries_all_stocks.png
├── metrics.json
├── portfolio_turnover_drawdown.png
├── report.md
└── weights.csv
```

`model_artifacts/` contains `predictions.csv`, `folds.csv`,
`feature_importance.csv`, `metrics.json`, `model_predictions.png`, and
`model_feature_importance.png`.

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

For live AkShare data, `--adjust hfq` is the default because AkShare documents
post-adjusted prices as a common choice for quantitative research. Use
`--adjust qfq` if that is the convention required by your study. Adjusted
prices are provider-derived and should not be mixed with unadjusted fields
without checking the corporate-action treatment.

## Research notes

This is intentionally a research skeleton rather than a production trading
engine. It does not include order-book simulation, borrow fees, exchange
calendars, corporate-action adjustment, portfolio constraints, or live trading.
The synthetic dataset is only a plumbing check and should not be interpreted as
evidence of a profitable strategy.

The live universe is the current list returned by AkShare, so a historical run
using it has survivorship bias. A production study should maintain historical
membership, delistings, suspension status, and point-in-time corporate-action
data. AkShare and the upstream market data sites can also rate-limit requests;
use the cache, a modest `--sleep`, and a small `--max-stocks` smoke test first.

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

The test suite also covers Fourier causality, data-quality quarantine, Bayesian
uncertainty, model walk-forward folds, formula injection rejection, and
multi-stock visualization output.

The code is organized into separate data, feature, model, AI, backtest, and
report layers so a new factor or model can be added without changing the data
source or portfolio accounting.

The synthetic example deliberately has no claim of predictive power. It exists
so the code path is easy to run in a fresh checkout.

## License

MIT
