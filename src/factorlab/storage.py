"""Optional columnar storage helpers with explicit dependency errors."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV or Parquet based on suffix and return the resolved path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(destination, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Parquet support requires the optional fast dependency") from exc
    else:
        frame.to_csv(destination, index=False)
    return destination


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet without silently changing the selected format."""

    source = Path(path)
    if source.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(source)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Parquet support requires the optional fast dependency") from exc
    return pd.read_csv(source)


def query_duckdb(path: str | Path, sql: str) -> pd.DataFrame:
    """Run a read-only DuckDB query against a CSV/Parquet dataset."""

    if not sql.lstrip().lower().startswith(("select", "with", "describe", "summarize")):
        raise ValueError("query_duckdb only permits read-only statements")
    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("sql must contain one statement without comments")
    if "{dataset}" not in sql:
        raise ValueError("sql must reference the {dataset} placeholder")
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB support requires the optional fast dependency") from exc
    source = str(Path(path).resolve()).replace("'", "''")
    relation = f"read_parquet('{source}')" if Path(path).suffix.lower() == ".parquet" else f"read_csv_auto('{source}')"
    # The relation is embedded only after resolving the local path; SQL text
    # remains read-only by the prefix gate above.
    return duckdb.sql(sql.replace("{dataset}", relation)).df()
