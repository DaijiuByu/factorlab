"""Quantitative data quality checks for A-share research panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import validate_panel


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for warnings and quarantined observations."""

    jump_threshold: float = 0.35
    stale_days: int = 5
    quarantine_price_jumps: bool = False

    def __post_init__(self) -> None:
        if self.jump_threshold <= 0:
            raise ValueError("jump_threshold must be positive")
        if self.stale_days < 2:
            raise ValueError("stale_days must be at least 2")


@dataclass
class QualityResult:
    cleaned: pd.DataFrame
    quarantine: pd.DataFrame
    issues: pd.DataFrame
    summary: pd.DataFrame


ISSUE_COLUMNS = ["row_id", "date", "ticker", "check", "severity", "detail"]


def _issue(
    frame: pd.DataFrame,
    mask: pd.Series,
    check: str,
    severity: str,
    detail: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id in frame.index[mask.fillna(False)]:
        rows.append(
            {
                "row_id": int(row_id),
                "date": frame.at[row_id, "date"],
                "ticker": frame.at[row_id, "ticker"],
                "check": check,
                "severity": severity,
                "detail": detail,
            }
        )
    return rows


def audit_panel(
    panel: pd.DataFrame,
    *,
    config: QualityConfig | None = None,
) -> QualityResult:
    """Validate a panel and return cleaned rows plus an auditable issue table.

    Hard schema/price failures raise immediately through ``validate_panel``.
    Soft quality issues (large jumps, stale prices, OHLC contradictions and
    negative activity fields) are recorded. Price jumps are warnings by
    default because a genuine corporate action can look exactly like one.
    """

    cfg = config or QualityConfig()
    clean = validate_panel(panel)
    issues: list[dict[str, Any]] = []
    if {"open", "high", "low"}.issubset(clean.columns):
        numeric = clean[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        bad_high = numeric["high"] < numeric[["open", "close"]].max(axis=1)
        bad_low = numeric["low"] > numeric[["open", "close"]].min(axis=1)
        issues.extend(
            _issue(
                clean,
                bad_high,
                "ohlc_high",
                "quarantine",
                "high is below open or close",
            )
        )
        issues.extend(
            _issue(
                clean, bad_low, "ohlc_low", "quarantine", "low is above open or close"
            )
        )
    returns = clean.groupby("ticker", sort=False)["close"].pct_change()
    jump_severity = "quarantine" if cfg.quarantine_price_jumps else "warning"
    issues.extend(
        _issue(
            clean,
            returns.abs() > cfg.jump_threshold,
            "price_jump",
            jump_severity,
            f"absolute one-day return exceeds {cfg.jump_threshold:.1%}",
        )
    )
    stale = (
        clean["close"]
        .groupby(clean["ticker"], sort=False)
        .transform(lambda series: series.diff().eq(0).rolling(cfg.stale_days).sum())
        >= cfg.stale_days
    )
    issues.extend(
        _issue(
            clean,
            stale,
            "stale_price",
            "warning",
            f"price unchanged for {cfg.stale_days} observations",
        )
    )
    for column in ("volume", "amount", "turnover_pct"):
        if column in clean:
            negative = pd.to_numeric(clean[column], errors="coerce") < 0
            issues.extend(
                _issue(
                    clean,
                    negative,
                    f"negative_{column}",
                    "quarantine",
                    f"{column} is negative",
                )
            )

    issues_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    quarantine_ids = set(
        issues_frame.loc[issues_frame["severity"] == "quarantine", "row_id"].astype(int)
        if not issues_frame.empty
        else []
    )
    quarantine = clean.loc[clean.index.isin(quarantine_ids)].copy()
    cleaned = clean.drop(index=quarantine_ids).reset_index(drop=True)
    if issues_frame.empty:
        summary = pd.DataFrame(columns=["check", "severity", "rows", "detail"])
    else:
        summary = (
            issues_frame.groupby(["check", "severity"], as_index=False)
            .agg(rows=("row_id", "nunique"), detail=("detail", "first"))
            .sort_values(["severity", "rows"], ascending=[True, False])
            .reset_index(drop=True)
        )
    return QualityResult(
        cleaned=cleaned,
        quarantine=quarantine,
        issues=issues_frame,
        summary=summary,
    )


def quality_report_markdown(result: QualityResult, *, source: str = "unknown") -> str:
    """Render a compact data quality report."""

    lines = [
        "# FactorLab data quality report",
        "",
        f"- Source: `{source}`",
        f"- Clean rows: **{len(result.cleaned):,}**",
        f"- Quarantined rows: **{len(result.quarantine):,}**",
        f"- Recorded issue rows: **{len(result.issues):,}**",
        "",
        "## Checks",
        "",
        "| Check | Severity | Rows | Detail |",
        "|---|---|---:|---|",
    ]
    if result.summary.empty:
        lines.append("| none | — | 0 | No soft quality issues detected |")
    else:
        for _, row in result.summary.iterrows():
            lines.append(
                f"| {row['check']} | {row['severity']} | {int(row['rows'])} | {row['detail']} |"
            )
    lines += [
        "",
        "Rows marked `warning` remain in the cleaned panel. Rows marked",
        "`quarantine` are excluded from the cleaned panel and written separately",
        "for review. A price jump is a warning by default because corporate actions",
        "and genuine news can create legitimate large moves.",
        "",
    ]
    return "\n".join(lines)


def write_quality_artifacts(
    result: QualityResult,
    output_dir: str,
    *,
    source: str = "unknown",
) -> None:
    """Write the cleaned data and all quality tables to a directory."""

    from pathlib import Path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.cleaned.to_csv(destination / "cleaned_panel.csv", index=False)
    result.quarantine.to_csv(destination / "quarantine.csv", index=False)
    result.issues.to_csv(destination / "quality_issues.csv", index=False)
    result.summary.to_csv(destination / "quality_summary.csv", index=False)
    (destination / "quality_report.md").write_text(
        quality_report_markdown(result, source=source), encoding="utf-8"
    )
