"""Run the complete FactorLab workflow without a data vendor."""

from pathlib import Path

from factorlab.data import generate_demo_panel
from factorlab.report import write_artifacts
from factorlab.research import BacktestConfig, run_research


def main() -> None:
    panel = generate_demo_panel(days=756, assets=40, seed=7)
    result = run_research(
        panel,
        factor="momentum",
        lookback=20,
        sector_neutral=True,
        backtest=BacktestConfig(quantile=0.2, cost_bps=5, min_assets=20),
        split_date="2020-01-01",
    )
    output = write_artifacts(result, Path("artifacts"), source="synthetic demo")
    print(f"Wrote {output / 'report.md'}")


if __name__ == "__main__":
    main()
