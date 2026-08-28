import tempfile
import unittest
from pathlib import Path

from factorlab.data import generate_demo_panel
from factorlab.report import write_artifacts
from factorlab.research import BacktestConfig, run_research


class VisualizationTests(unittest.TestCase):
    def test_multi_stock_visualizations_are_written(self):
        panel = generate_demo_panel(days=80, assets=12, seed=21)
        result = run_research(
            panel,
            factor="momentum",
            lookback=10,
            backtest=BacktestConfig(quantile=0.2, cost_bps=5, min_assets=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_artifacts(result, output, source="test")
            expected = {
                "asset_metrics.csv",
                "asset_metrics_all_stocks.png",
                "asset_metrics_heatmap.png",
                "metric_summary.csv",
                "metric_timeseries_all_stocks.png",
                "factor_diagnostics.png",
                "portfolio_turnover_drawdown.png",
                "equity_curve.png",
                "report.md",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertEqual(len(result.asset_metrics), 12)
