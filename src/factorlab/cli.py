"""Command-line interface for FactorLab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .ai.formula import evaluate_formula, validate_formula
from .ai.research_assistant import client_from_environment
from .akshare_data import fetch_sse_panel, resolve_window
from .backtest import run_vectorbt, target_weights_from_scores
from .data import generate_demo_panel, load_panel
from .features import build_features
from .models import ModelConfig, walk_forward_alpha, write_model_plots
from .quality import QualityConfig, audit_panel, write_quality_artifacts
from .report import write_artifacts
from .research import BacktestConfig, run_research


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small cross-sectional factor research toolkit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write deterministic synthetic panel data")
    demo.add_argument("--output", type=Path, default=Path("demo_panel.csv"))
    demo.add_argument("--days", type=int, default=756)
    demo.add_argument("--assets", type=int, default=40)
    demo.add_argument("--seed", type=int, default=7)

    analyze = sub.add_parser(
        "analyze", help="run factor diagnostics and a long/short backtest"
    )
    analyze.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV panel with date,ticker,close columns",
    )
    analyze.add_argument("--output", type=Path, default=Path("artifacts"))
    analyze.add_argument(
        "--factor",
        choices=["momentum", "reversal", "low_volatility", "column"],
        default="momentum",
    )
    analyze.add_argument(
        "--column", dest="raw_column", help="raw factor column when --factor column"
    )
    analyze.add_argument("--direction", type=float, choices=[-1.0, 1.0], default=1.0)
    analyze.add_argument("--lookback", type=int, default=20)
    analyze.add_argument("--quantile", type=float, default=0.2)
    analyze.add_argument("--cost-bps", type=float, default=5.0)
    analyze.add_argument("--min-assets", type=int, default=10)
    analyze.add_argument("--sector-neutral", action="store_true")
    analyze.add_argument(
        "--split-date", help="optional YYYY-MM-DD boundary for before/after metrics"
    )
    analyze.add_argument("--start-date", help="first date included in the analysis")
    analyze.add_argument("--end-date", help="last date included in the analysis")

    live = sub.add_parser(
        "live", help="fetch recent SSE data with AkShare and run the analysis"
    )
    live.add_argument("--output", type=Path, default=Path("artifacts_live"))
    live.add_argument(
        "--start-date",
        help="first calendar date; defaults to one month before --end-date",
    )
    live.add_argument("--end-date", help="last calendar date; defaults to today")
    live.add_argument("--lookback", type=int, default=20)
    live.add_argument(
        "--factor",
        choices=["momentum", "reversal", "low_volatility", "column"],
        default="momentum",
    )
    live.add_argument(
        "--column", dest="raw_column", help="raw factor column when --factor column"
    )
    live.add_argument("--direction", type=float, choices=[-1.0, 1.0], default=1.0)
    live.add_argument("--quantile", type=float, default=0.2)
    live.add_argument("--cost-bps", type=float, default=5.0)
    live.add_argument("--min-assets", type=int, default=20)
    live.add_argument("--sector-neutral", action="store_true")
    live.add_argument(
        "--split-date", help="optional YYYY-MM-DD boundary for before/after metrics"
    )
    live.add_argument("--board", choices=["all", "main", "star"], default="all")
    live.add_argument("--adjust", choices=["hfq", "qfq", ""], default="hfq")
    live.add_argument(
        "--max-stocks",
        type=int,
        help="limit symbols for a quick run; omit for the full SSE A-share universe",
    )
    live.add_argument("--cache-dir", type=Path, default=Path(".factorlab_cache"))
    live.add_argument(
        "--sleep",
        dest="sleep_seconds",
        type=float,
        default=1.0,
        help="seconds between history requests",
    )
    live.add_argument("--retries", type=int, default=3)

    quality = sub.add_parser(
        "quality", help="audit a CSV panel and write clean/quarantine tables"
    )
    quality.add_argument("--input", type=Path, required=True)
    quality.add_argument("--output", type=Path, default=Path("quality_artifacts"))
    quality.add_argument("--jump-threshold", type=float, default=0.35)
    quality.add_argument("--stale-days", type=int, default=5)
    quality.add_argument("--quarantine-jumps", action="store_true")

    features = sub.add_parser(
        "features", help="build point-in-time features from a CSV panel"
    )
    features.add_argument("--input", type=Path, required=True)
    features.add_argument("--output", type=Path, default=Path("features.csv"))
    features.add_argument("--lookbacks", default="5,20,60")
    features.add_argument("--fft-window", type=int, default=32)
    features.add_argument("--no-fourier", action="store_true")

    model = sub.add_parser(
        "model", help="run a chronological machine-learning Alpha study"
    )
    model.add_argument("--input", type=Path, required=True)
    model.add_argument("--output", type=Path, default=Path("model_artifacts"))
    model.add_argument(
        "--model",
        choices=[
            "ridge",
            "elastic_net",
            "hist_gradient_boosting",
            "random_forest",
            "bayesian_shrinkage",
            "lightgbm",
            "xgboost",
        ],
        default="ridge",
    )
    model.add_argument(
        "--feature-columns",
        help="comma-separated columns; defaults to FactorLab features",
    )
    model.add_argument("--lookbacks", default="5,20,60")
    model.add_argument("--fft-window", type=int, default=32)
    model.add_argument("--no-fourier", action="store_true")
    model.add_argument("--horizon", type=int, default=1)
    model.add_argument("--train-days", type=int, default=252)
    model.add_argument("--test-days", type=int, default=21)
    model.add_argument("--purge-days", type=int, default=1)
    model.add_argument("--n-jobs", type=int, default=4)

    model_backtest = sub.add_parser(
        "model-backtest",
        help="backtest out-of-sample model scores from a predictions CSV",
    )
    model_backtest.add_argument(
        "--input", type=Path, required=True, help="original price panel CSV"
    )
    model_backtest.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="predictions.csv from factorlab model",
    )
    model_backtest.add_argument(
        "--output", type=Path, default=Path("model_backtest_artifacts")
    )
    model_backtest.add_argument("--quantile", type=float, default=0.2)
    model_backtest.add_argument("--cost-bps", type=float, default=5.0)
    model_backtest.add_argument("--min-assets", type=int, default=20)
    model_backtest.add_argument("--start-date")
    model_backtest.add_argument("--end-date")
    model_backtest.add_argument(
        "--engine", choices=["reference", "vectorbt"], default="reference"
    )
    model_backtest.add_argument("--fees", type=float, default=0.001)
    model_backtest.add_argument("--slippage", type=float, default=0.0005)

    ai = sub.add_parser("ai", help="use DeepSeek for a human-reviewed factor proposal")
    ai_sub = ai.add_subparsers(dest="ai_command", required=True)
    propose = ai_sub.add_parser("propose", help="propose one factor as validated JSON")
    propose.add_argument("--question", required=True)
    propose.add_argument(
        "--columns", default="date,ticker,close,volume,amount,turnover_pct"
    )
    propose.add_argument("--output", type=Path)
    formula = ai_sub.add_parser(
        "validate", help="validate and evaluate a safe factor formula"
    )
    formula.add_argument("--formula", required=True)
    formula.add_argument("--input", type=Path, required=True)
    formula.add_argument("--output", type=Path)
    return parser


def _default_feature_columns(frame) -> list[str]:
    prefixes = (
        "return_",
        "log_",
        "momentum_",
        "reversal_",
        "volatility_",
        "amount_",
        "amihud_",
        "spectral_",
        "turnover_",
    )
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and frame[column].notna().any()
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            panel = generate_demo_panel(
                days=args.days, assets=args.assets, seed=args.seed
            )
            panel.to_csv(args.output, index=False)
            print(f"Wrote {len(panel):,} rows to {args.output}")
            return 0
        if args.command == "quality":
            panel = load_panel(args.input)
            result = audit_panel(
                panel,
                config=QualityConfig(
                    jump_threshold=args.jump_threshold,
                    stale_days=args.stale_days,
                    quarantine_price_jumps=args.quarantine_jumps,
                ),
            )
            write_quality_artifacts(result, str(args.output), source=str(args.input))
            print(
                f"Quality audit: clean={len(result.cleaned):,}, "
                f"quarantine={len(result.quarantine):,}, issues={len(result.issues):,}"
            )
            return 0
        if args.command == "features":
            panel = load_panel(args.input)
            lookbacks = tuple(
                int(value.strip())
                for value in args.lookbacks.split(",")
                if value.strip()
            )
            frame = build_features(
                panel,
                lookbacks=lookbacks,
                include_fourier=not args.no_fourier,
                fft_window=args.fft_window,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"Wrote {len(frame):,} feature rows to {args.output}")
            return 0
        if args.command == "model":
            panel = load_panel(args.input)
            lookbacks = tuple(
                int(value.strip())
                for value in args.lookbacks.split(",")
                if value.strip()
            )
            frame = build_features(
                panel,
                lookbacks=lookbacks,
                include_fourier=not args.no_fourier,
                fft_window=args.fft_window,
            )
            feature_columns = (
                [
                    value.strip()
                    for value in args.feature_columns.split(",")
                    if value.strip()
                ]
                if args.feature_columns
                else _default_feature_columns(frame)
            )
            result = walk_forward_alpha(
                frame,
                feature_columns,
                config=ModelConfig(
                    model=args.model,
                    horizon=args.horizon,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    purge_days=args.purge_days,
                    n_jobs=args.n_jobs,
                ),
            )
            args.output.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(args.output / "predictions.csv", index=False)
            result.feature_importance.to_csv(
                args.output / "feature_importance.csv", index=False
            )
            result.folds.to_csv(args.output / "folds.csv", index=False)
            (args.output / "metrics.json").write_text(
                json.dumps(
                    {"metrics": result.metrics, "config": result.model_config}, indent=2
                ),
                encoding="utf-8",
            )
            write_model_plots(result, args.output)
            print(f"Wrote model artifacts to {args.output}; metrics={result.metrics}")
            return 0
        if args.command == "model-backtest":
            panel = load_panel(args.input)
            predictions = pd.read_csv(args.predictions)
            required = {"date", "ticker", "alpha_score"}
            missing_predictions = required - set(predictions.columns)
            if missing_predictions:
                raise ValueError(
                    f"predictions is missing columns: {', '.join(sorted(missing_predictions))}"
                )
            predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
            predictions["ticker"] = predictions["ticker"].astype(str)
            panel["ticker"] = panel["ticker"].astype(str)
            predictions = predictions[
                ["date", "ticker", "alpha_score"]
            ].drop_duplicates(["date", "ticker"])
            panel = panel.merge(predictions, on=["date", "ticker"], how="left")
            quality_result = audit_panel(panel)
            write_quality_artifacts(
                quality_result, str(args.output), source=str(args.input)
            )
            if args.engine == "vectorbt":
                prices, target_weights = target_weights_from_scores(
                    quality_result.cleaned,
                    quantile=args.quantile,
                    min_assets=args.min_assets,
                )
                vector_result = run_vectorbt(
                    prices,
                    target_weights,
                    fees=args.fees,
                    slippage=args.slippage,
                )
                (args.output / "vectorbt_stats.json").write_text(
                    json.dumps(vector_result.stats.to_dict(), default=str, indent=2),
                    encoding="utf-8",
                )
                print(f"Wrote VectorBT stats to {args.output}")
            else:
                result = run_research(
                    quality_result.cleaned,
                    factor="column",
                    raw_column="alpha_score",
                    backtest=BacktestConfig(
                        quantile=args.quantile,
                        cost_bps=args.cost_bps,
                        min_assets=args.min_assets,
                    ),
                    analysis_start=args.start_date,
                    analysis_end=args.end_date,
                )
                output = write_artifacts(
                    result, args.output, source=f"model predictions: {args.predictions}"
                )
                print(f"Wrote model backtest artifacts to {output}")
                print(f"Net total return: {result.metrics.get('total_return')}")
            return 0
        if args.command == "ai":
            if args.ai_command == "validate":
                formula_text = validate_formula(args.formula)
                panel = load_panel(args.input)
                values = evaluate_formula(panel, formula_text)
                output_frame = panel[["date", "ticker"]].copy()
                output_frame["alpha_score"] = values
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    output_frame.to_csv(args.output, index=False)
                    print(f"Wrote formula scores to {args.output}")
                else:
                    print(output_frame.head(20).to_string(index=False))
                return 0
            client = client_from_environment()
            if client is None:
                raise RuntimeError("DEEPSEEK_API_KEY is not set")
            columns = [
                value.strip() for value in args.columns.split(",") if value.strip()
            ]
            proposal = client.propose_factor(args.question, columns)
            rendered = json.dumps(proposal.as_dict(), ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
                print(f"Wrote factor proposal to {args.output}")
            else:
                print(rendered)
            return 0
        if args.command == "analyze":
            panel = load_panel(args.input)
            quality_result = audit_panel(panel)
            write_quality_artifacts(
                quality_result, str(args.output), source=str(args.input)
            )
            panel = quality_result.cleaned
            result = run_research(
                panel,
                factor=args.factor,
                lookback=args.lookback,
                raw_column=args.raw_column,
                direction=args.direction,
                sector_neutral=args.sector_neutral,
                backtest=BacktestConfig(
                    quantile=args.quantile,
                    cost_bps=args.cost_bps,
                    min_assets=args.min_assets,
                ),
                split_date=args.split_date,
                analysis_start=args.start_date,
                analysis_end=args.end_date,
            )
            output = write_artifacts(result, args.output, source=str(args.input))
        else:
            selected_start, selected_end = resolve_window(
                args.start_date, args.end_date
            )
            fetched = fetch_sse_panel(
                start_date=selected_start.strftime("%Y-%m-%d"),
                end_date=selected_end.strftime("%Y-%m-%d"),
                lookback=args.lookback,
                board=args.board,
                adjust=args.adjust,
                max_stocks=args.max_stocks,
                cache_dir=args.cache_dir,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
            )
            quality_result = audit_panel(fetched.panel)
            write_quality_artifacts(
                quality_result,
                str(args.output),
                source="AkShare / Shanghai Stock Exchange",
            )
            result = run_research(
                quality_result.cleaned,
                factor=args.factor,
                lookback=args.lookback,
                raw_column=args.raw_column,
                direction=args.direction,
                sector_neutral=args.sector_neutral,
                backtest=BacktestConfig(
                    quantile=args.quantile,
                    cost_bps=args.cost_bps,
                    min_assets=args.min_assets,
                ),
                split_date=args.split_date,
                analysis_start=selected_start.strftime("%Y-%m-%d"),
                analysis_end=selected_end.strftime("%Y-%m-%d"),
                market_summary=fetched.market_summary,
                data_metadata=fetched.metadata,
            )
            output = write_artifacts(
                result, args.output, source="AkShare / Shanghai Stock Exchange"
            )
            fetched.universe.to_csv(output / "universe.csv", index=False)
            fetched.errors.to_csv(output / "fetch_errors.csv", index=False)
            print(
                f"Fetched {fetched.metadata['successful_stocks']}/"
                f"{fetched.metadata['requested_stocks']} stocks; "
                f"failed: {fetched.metadata['failed_stocks']}"
            )
        print(f"Wrote research artifacts to {output}")
        print(f"Net total return: {result.metrics.get('total_return')}")
        print(f"Mean IC: {result.metrics.get('mean_ic')}")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"factorlab: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
