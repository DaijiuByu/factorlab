"""Reproducibility metadata for FactorLab experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .data import validate_panel


def panel_fingerprint(panel: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of a canonicalized panel."""

    clean = validate_panel(panel)
    payload = pd.util.hash_pandas_object(clean, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def build_run_manifest(
    panel: pd.DataFrame,
    *,
    config: dict[str, Any],
    source: str,
    data_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for one research run."""

    clean = validate_panel(panel)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": source,
        "rows": int(len(clean)),
        "assets": int(clean["ticker"].nunique()),
        "date_start": clean["date"].min().strftime("%Y-%m-%d") if not clean.empty else None,
        "date_end": clean["date"].max().strftime("%Y-%m-%d") if not clean.empty else None,
        "panel_sha256": panel_fingerprint(clean),
        "config": config,
        "data_metadata": data_metadata or {},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    revision = _git_revision()
    if revision:
        manifest["git_revision"] = revision
    return manifest


def write_run_manifest(
    destination: str | Path,
    panel: pd.DataFrame,
    *,
    config: dict[str, Any],
    source: str,
    data_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_manifest.json`` and return its path."""

    path = Path(destination) / "run_manifest.json"
    path.write_text(
        json.dumps(
            build_run_manifest(
                panel,
                config=config,
                source=source,
                data_metadata=data_metadata,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
