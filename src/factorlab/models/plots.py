"""Compact plots for out-of-sample Alpha model diagnostics."""

from __future__ import annotations

from pathlib import Path

from .supervised import ModelResult


def write_model_plots(result: ModelResult, output_dir: str | Path) -> list[Path]:
    """Write prediction and feature-importance plots when data is available."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        destination / "model_predictions.png",
        destination / "model_feature_importance.png",
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    if result.predictions.empty:
        axis.text(0.5, 0.5, "No out-of-sample predictions", ha="center", va="center")
        axis.set_axis_off()
    else:
        predictions = result.predictions
        axis.scatter(
            predictions["target"],
            predictions["alpha_score"],
            s=8,
            alpha=0.4,
            color="#1f4e79",
        )
        low = min(predictions["target"].min(), predictions["alpha_score"].min())
        high = max(predictions["target"].max(), predictions["alpha_score"].max())
        axis.plot(
            [low, high], [low, high], color="#c44e52", linestyle="--", linewidth=1
        )
        axis.set_xlabel("Forward return")
        axis.set_ylabel("Predicted Alpha")
        axis.set_title("Out-of-sample prediction vs. realized return")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(paths[0], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    if result.feature_importance.empty:
        axis.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        axis.set_axis_off()
    else:
        importance = (
            result.feature_importance.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .head(20)
            .sort_values("importance")
        )
        axis.barh(importance["feature"], importance["importance"], color="#4c72b0")
        axis.set_title("Mean out-of-sample feature importance")
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(paths[1], dpi=150)
    plt.close(figure)
    return paths
