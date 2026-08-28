"""Static diagnostics for multi-stock factor research results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .research import ResearchResult


METRICS = (
    ("momentum", "Momentum"),
    ("reversal", "One-day reversal"),
    ("volatility", "Annualized volatility"),
    ("turnover_pct", "Turnover (%)"),
)


def _plot_empty(axis, title: str, message: str = "No data") -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _set_sampled_ticks(axis, labels: list[str]) -> None:
    if len(labels) <= 40:
        positions = np.arange(len(labels))
    else:
        count = min(20, len(labels))
        positions = np.unique(np.linspace(0, len(labels) - 1, count).astype(int))
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [labels[index] for index in positions], rotation=75, ha="right", fontsize=7
    )


def _asset_metrics_all_stocks(result: "ResearchResult", output: Path) -> None:
    import matplotlib.pyplot as plt

    frame = result.asset_metrics.sort_values("ticker", kind="stable").reset_index(
        drop=True
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(max(13, min(28, 10 + len(frame) / 40)), 9)
    )
    if frame.empty:
        for axis, (_, title) in zip(axes.flat, METRICS):
            _plot_empty(axis, title)
    else:
        x = np.arange(len(frame))
        labels = frame["ticker"].astype(str).tolist()
        for axis, (column, title) in zip(axes.flat, METRICS):
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna().to_numpy()
            axis.scatter(
                x[valid],
                values[valid],
                s=12 if len(frame) < 100 else 5,
                alpha=0.72,
                color="#1f4e79",
            )
            if valid.any():
                axis.axhline(
                    values.mean(),
                    color="#c44e52",
                    linestyle="--",
                    linewidth=1,
                    label="cross-sectional mean",
                )
                axis.legend(fontsize=8)
            axis.set_title(f"{title} — all {len(frame)} stocks")
            axis.set_xlabel(
                "Ticker (all observations plotted; labels sampled for readability)"
            )
            axis.grid(alpha=0.2)
            _set_sampled_ticks(axis, labels)
    figure.tight_layout()
    figure.savefig(output / "asset_metrics_all_stocks.png", dpi=150)
    plt.close(figure)


def _asset_metrics_heatmap(result: "ResearchResult", output: Path) -> None:
    import matplotlib.pyplot as plt

    frame = result.asset_metrics.sort_values("ticker", kind="stable").reset_index(
        drop=True
    )
    figure_height = max(5, min(42, 3 + len(frame) * 0.12))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    if frame.empty:
        _plot_empty(axis, "Cross-sectional percentile ranks — all stocks")
    else:
        values = frame[[column for column, _ in METRICS]].apply(
            pd.to_numeric, errors="coerce"
        )
        ranked = values.rank(pct=True)
        image = axis.imshow(
            ranked.fillna(0.5).to_numpy(),
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"Cross-sectional percentile ranks — all {len(frame)} stocks")
        axis.set_xlabel("Metric")
        axis.set_xticks(np.arange(len(METRICS)))
        axis.set_xticklabels([title for _, title in METRICS], rotation=20, ha="right")
        if len(frame) <= 80:
            axis.set_yticks(np.arange(len(frame)))
            axis.set_yticklabels(frame["ticker"].astype(str), fontsize=7)
        else:
            positions = np.unique(np.linspace(0, len(frame) - 1, 24).astype(int))
            axis.set_yticks(positions)
            axis.set_yticklabels(frame.loc[positions, "ticker"].astype(str), fontsize=7)
        figure.colorbar(image, ax=axis, label="Percentile rank")
    figure.tight_layout()
    figure.savefig(output / "asset_metrics_heatmap.png", dpi=150)
    plt.close(figure)


def _metric_timeseries(result: "ResearchResult", output: Path) -> None:
    import matplotlib.pyplot as plt

    summary = result.metric_summary
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    if summary.empty:
        for axis, (_, title) in zip(axes.flat, METRICS):
            _plot_empty(axis, title)
    else:
        for axis, (metric, title) in zip(axes.flat, METRICS):
            axis.plot(
                summary["date"], summary[f"{metric}_mean"], label="mean", linewidth=1.1
            )
            axis.plot(
                summary["date"],
                summary[f"{metric}_median"],
                label="median",
                linewidth=1.1,
            )
            axis.set_title(f"{title} — cross-sectional mean and median")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.suptitle("Daily cross-sectional metric summary across all available stocks")
    figure.tight_layout()
    figure.savefig(output / "metric_timeseries_all_stocks.png", dpi=150)
    plt.close(figure)


def _factor_diagnostics(result: "ResearchResult", output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    if result.ic_by_date.empty:
        _plot_empty(axes[0], "Daily information coefficient")
    else:
        ic = result.ic_by_date.sort_values("date")
        axes[0].plot(
            ic["date"], ic["ic"], color="#4c72b0", linewidth=0.9, label="daily IC"
        )
        axes[0].plot(
            ic["date"],
            ic["ic"].rolling(20, min_periods=5).mean(),
            color="#dd8452",
            linewidth=1.5,
            label="20-day rolling mean",
        )
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].set_title(
            f"Daily information coefficient — {ic['n_assets'].max()} max stocks per date"
        )
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)
    if result.daily.empty:
        _plot_empty(axes[1], "Portfolio diagnostics")
    else:
        daily = result.daily.sort_values("date").copy()
        gross_equity = (1 + daily["gross_return"]).cumprod()
        net_equity = (1 + daily["net_return"]).cumprod()
        axes[1].plot(daily["date"], gross_equity, label="gross equity", linewidth=1.0)
        axes[1].plot(daily["date"], net_equity, label="net equity", linewidth=1.4)
        axes[1].set_title("Portfolio gross vs. net equity")
        axes[1].set_ylabel("Growth of 1.0")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "factor_diagnostics.png", dpi=150)
    plt.close(figure)


def _portfolio_diagnostics(result: "ResearchResult", output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if result.daily.empty:
        for axis in axes:
            _plot_empty(axis, "Portfolio diagnostics")
    else:
        daily = result.daily.sort_values("date").copy()
        net_equity = (1 + daily["net_return"]).cumprod()
        drawdown = net_equity / net_equity.cummax() - 1
        axes[0].bar(
            daily["date"], daily["turnover"], width=1.0, color="#55a868", alpha=0.65
        )
        axes[0].set_title("Daily portfolio turnover")
        axes[0].set_ylabel("Turnover")
        axes[0].grid(alpha=0.2)
        axes[1].fill_between(daily["date"], drawdown, 0, color="#c44e52", alpha=0.35)
        axes[1].set_title("Net portfolio drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "portfolio_turnover_drawdown.png", dpi=150)
    plt.close(figure)


def write_visualizations(
    result: "ResearchResult", output_dir: str | Path
) -> list[Path]:
    """Write multi-stock visual diagnostics and return their paths."""

    import matplotlib

    matplotlib.use("Agg")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _asset_metrics_all_stocks(result, destination)
    _asset_metrics_heatmap(result, destination)
    _metric_timeseries(result, destination)
    _factor_diagnostics(result, destination)
    _portfolio_diagnostics(result, destination)
    return [
        destination / "asset_metrics_all_stocks.png",
        destination / "asset_metrics_heatmap.png",
        destination / "metric_timeseries_all_stocks.png",
        destination / "factor_diagnostics.png",
        destination / "portfolio_turnover_drawdown.png",
    ]
