import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from factorlab.akshare_data import fetch_sse_panel, resolve_window


class FakeAkShare:
    def stock_sse_summary(self):
        return pd.DataFrame(
            {
                "项目": ["报告时间", "换手率"],
                "股票": ["20260827", "1.2"],
                "科创板": ["20260827", "1.5"],
                "主板": ["20260827", "1.1"],
            }
        )

    def stock_info_sh_name_code(self, symbol=None, indicator=None):
        board = symbol or indicator
        code = "600000" if board == "主板A股" else "688001"
        return pd.DataFrame(
            {
                "证券代码": [code],
                "证券简称": ["测试股票"],
                "上市日期": ["2000-01-01"],
            }
        )

    def stock_zh_a_hist(
        self, symbol, period, start_date, end_date, adjust, timeout=None
    ):
        dates = pd.bdate_range("2026-05-01", periods=50)
        return pd.DataFrame(
            {
                "日期": dates,
                "股票代码": symbol,
                "开盘": range(10, 60),
                "收盘": range(10, 60),
                "最高": range(10, 60),
                "最低": range(10, 60),
                "成交量": [100] * 50,
                "成交额": [1000] * 50,
                "换手率": [1.0] * 50,
            }
        )


class AkShareDataTests(unittest.TestCase):
    def test_window_defaults_to_one_calendar_month(self):
        start, end = resolve_window("2026-07-15", "2026-08-15")
        self.assertEqual(start, pd.Timestamp("2026-07-15"))
        self.assertEqual(end, pd.Timestamp("2026-08-15"))

    def test_fetch_normalizes_realistic_akshare_columns_and_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("factorlab.akshare_data._akshare", return_value=FakeAkShare()):
                result = fetch_sse_panel(
                    start_date="2026-05-20",
                    end_date="2026-06-30",
                    lookback=20,
                    board="main",
                    max_stocks=1,
                    cache_dir=Path(directory),
                    sleep_seconds=0,
                    retries=0,
                )
            self.assertEqual(result.metadata["successful_stocks"], 1)
            self.assertEqual(result.panel["ticker"].unique().tolist(), ["600000"])
            self.assertIn("turnover_pct", result.panel)
            self.assertTrue((Path(directory) / "600000_hfq.csv").is_file())
            self.assertFalse(result.market_summary.empty)

    def test_falls_back_to_sina_shape_when_primary_history_fails(self):
        class PrimaryFails(FakeAkShare):
            def stock_zh_a_hist(self, *args, **kwargs):
                raise ConnectionError("primary source unavailable")

            def stock_zh_a_daily(self, symbol, start_date, end_date, adjust):
                dates = pd.bdate_range("2026-05-01", periods=50)
                return pd.DataFrame(
                    {
                        "date": dates,
                        "open": [10.0] * 50,
                        "high": [11.0] * 50,
                        "low": [9.0] * 50,
                        "close": [10.0] * 50,
                        "volume": [100] * 50,
                        "amount": [1000] * 50,
                        "outstanding_share": [100_000] * 50,
                        "turnover": [0.001] * 50,
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            with patch("factorlab.akshare_data._akshare", return_value=PrimaryFails()):
                result = fetch_sse_panel(
                    start_date="2026-05-20",
                    end_date="2026-06-30",
                    lookback=20,
                    board="main",
                    max_stocks=1,
                    cache_dir=Path(directory),
                    sleep_seconds=0,
                    retries=0,
                )
            self.assertEqual(result.metadata["successful_stocks"], 1)
            self.assertEqual(
                result.metadata["history_sources_used"], ["stock_zh_a_daily"]
            )
            self.assertAlmostEqual(float(result.panel["turnover_pct"].iloc[0]), 0.1)
