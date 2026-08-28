"""AkShare-backed Shanghai Stock Exchange data ingestion.

The public AkShare APIs are imported lazily so the existing CSV workflow keeps
working in environments that do not install the live-data dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import validate_panel


@dataclass
class SSEFetchResult:
    panel: pd.DataFrame
    market_summary: pd.DataFrame
    universe: pd.DataFrame
    errors: pd.DataFrame
    metadata: dict[str, Any]


def resolve_window(
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    months: int = 1,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a calendar-date window, defaulting to the most recent month."""

    end = (
        pd.Timestamp(end_date).normalize()
        if end_date
        else pd.Timestamp.today().normalize()
    )
    start = (
        pd.Timestamp(start_date).normalize()
        if start_date
        else end - pd.DateOffset(months=months)
    )
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return start, end


def _akshare() -> Any:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "AkShare is not installed; run `python -m pip install -e .` first"
        ) from exc
    return ak


def fetch_sse_summary() -> pd.DataFrame:
    """Fetch the latest Shanghai exchange market overview snapshot."""

    summary = _akshare().stock_sse_summary()
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise RuntimeError("stock_sse_summary returned no data")
    return summary.copy()


def _fetch_stock_list(ak: Any, board: str) -> pd.DataFrame:
    api_board = {"main": "主板A股", "star": "科创板"}[board]
    function = ak.stock_info_sh_name_code
    try:
        result = function(symbol=api_board)
    except TypeError:
        # Compatibility with older AkShare releases that used indicator=.
        result = function(indicator=api_board)
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise RuntimeError(f"stock_info_sh_name_code returned no data for {api_board}")
    return result.copy()


def fetch_sse_universe(board: str = "all") -> pd.DataFrame:
    """Fetch current SSE A-share symbols from the official exchange list API."""

    if board not in {"all", "main", "star"}:
        raise ValueError("board must be all, main, or star")
    ak = _akshare()
    boards = ["main", "star"] if board == "all" else [board]
    frames = []
    for board_name in boards:
        frame = _fetch_stock_list(ak, board_name)
        code_column = next(
            (
                name
                for name in ("证券代码", "代码", "公司代码", "END_SHARE_CODE")
                if name in frame
            ),
            None,
        )
        name_column = next(
            (
                name
                for name in ("证券简称", "简称", "公司简称", "COMPANY_ABBR")
                if name in frame
            ),
            None,
        )
        listing_column = next(
            (name for name in ("上市日期", "LISTING_DATE") if name in frame), None
        )
        if code_column is None:
            raise RuntimeError(
                f"unrecognized stock list columns: {list(frame.columns)}"
            )
        normalized = pd.DataFrame(
            {"ticker": frame[code_column].astype(str).str.extract(r"(\d{6})")[0]}
        )
        normalized["name"] = frame[name_column].astype(str) if name_column else ""
        normalized["board"] = "主板A股" if board_name == "main" else "科创板"
        normalized["listed_date"] = (
            pd.to_datetime(frame[listing_column], errors="coerce")
            if listing_column
            else pd.NaT
        )
        frames.append(normalized)
    universe = pd.concat(frames, ignore_index=True)
    universe = (
        universe.dropna(subset=["ticker"])
        .drop_duplicates("ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    if universe.empty:
        raise RuntimeError("no valid six-digit SSE stock codes were found")
    return universe


def _normalize_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "股票代码": "ticker",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_pct",
    }
    frame = raw.rename(columns=rename).copy()
    if "date" not in frame or "close" not in frame:
        raise RuntimeError(
            f"unrecognized history columns for {ticker}: {list(raw.columns)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = ticker
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover_pct"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    return (
        frame.sort_values("date", kind="stable")
        .drop_duplicates(["date", "ticker"])
        .reset_index(drop=True)
    )


def _read_cached(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, ticker: str
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = _normalize_history(pd.read_csv(path), ticker)
    except (OSError, ValueError, RuntimeError):
        return None
    if frame.empty or frame["date"].min() > start or frame["date"].max() < end:
        return None
    return frame


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def fetch_sse_panel(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback: int = 20,
    board: str = "all",
    adjust: str = "hfq",
    max_stocks: int | None = None,
    cache_dir: str | Path = ".factorlab_cache",
    sleep_seconds: float = 0.15,
    retries: int = 2,
) -> SSEFetchResult:
    """Fetch a warm-started SSE daily panel for a selected calendar window.

    ``stock_sse_summary`` is fetched once as the latest market snapshot. The
    individual rows come from ``stock_zh_a_hist``. Each symbol is cached and a
    failed symbol is recorded so one bad response does not discard the run.
    """

    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be empty, qfq, or hfq")
    if max_stocks is not None and max_stocks < 1:
        raise ValueError("max_stocks must be positive")
    if sleep_seconds < 0 or retries < 0:
        raise ValueError("sleep_seconds and retries must be non-negative")
    selected_start, selected_end = resolve_window(start_date, end_date)
    warmup_start = selected_start - pd.Timedelta(days=max(lookback * 3, 60))
    ak = _akshare()
    summary = fetch_sse_summary()
    universe = fetch_sse_universe(board)
    if max_stocks is not None:
        universe = universe.head(max_stocks).copy()

    cache_root = Path(cache_dir)
    rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    requested = len(universe)
    for index, item in universe.iterrows():
        ticker = str(item["ticker"])
        cache_file = cache_root / f"{ticker}_{adjust or 'raw'}.csv"
        history = _read_cached(cache_file, warmup_start, selected_end, ticker)
        last_error = ""
        if history is None:
            for attempt in range(retries + 1):
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=ticker,
                        period="daily",
                        start_date=warmup_start.strftime("%Y%m%d"),
                        end_date=selected_end.strftime("%Y%m%d"),
                        adjust=adjust,
                    )
                    history = _normalize_history(raw, ticker)
                    if history.empty:
                        raise RuntimeError("empty history")
                    _write_cache(cache_file, history)
                    break
                except (
                    Exception
                ) as exc:  # AkShare propagates provider-specific exceptions.
                    last_error = str(exc)
                    history = None
                    if attempt < retries:
                        time.sleep(min(2.0, 0.5 * (2**attempt)))
            if sleep_seconds and index + 1 < requested:
                time.sleep(sleep_seconds)
        if history is None:
            errors.append(
                {
                    "ticker": ticker,
                    "name": str(item.get("name", "")),
                    "error": last_error[-500:],
                }
            )
            continue
        history["name"] = str(item.get("name", ""))
        history["board"] = str(item.get("board", ""))
        history["listed_date"] = item.get("listed_date", pd.NaT)
        rows.append(history)

    if not rows:
        raise RuntimeError("no stock history was fetched successfully")
    panel = validate_panel(pd.concat(rows, ignore_index=True))
    panel = panel.loc[
        (panel["date"] >= warmup_start) & (panel["date"] <= selected_end)
    ].reset_index(drop=True)
    return SSEFetchResult(
        panel=panel,
        market_summary=summary,
        universe=universe,
        errors=pd.DataFrame(errors, columns=["ticker", "name", "error"]),
        metadata={
            "provider": "akshare",
            "summary_interface": "stock_sse_summary",
            "universe_interface": "stock_info_sh_name_code",
            "history_interface": "stock_zh_a_hist",
            "selected_start": selected_start.strftime("%Y-%m-%d"),
            "selected_end": selected_end.strftime("%Y-%m-%d"),
            "warmup_start": warmup_start.strftime("%Y-%m-%d"),
            "board": board,
            "adjust": adjust,
            "requested_stocks": requested,
            "successful_stocks": len(rows),
            "failed_stocks": len(errors),
        },
    )
