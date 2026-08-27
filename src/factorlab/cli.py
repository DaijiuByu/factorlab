"""Command-line interface for FactorLab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import generate_demo_panel, load_panel
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
    return parser


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
        panel = load_panel(args.input)
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
        )
        output = write_artifacts(result, args.output, source=str(args.input))
        print(f"Wrote research artifacts to {output}")
        print(f"Net total return: {result.metrics.get('total_return')}")
        print(f"Mean IC: {result.metrics.get('mean_ic')}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"factorlab: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
