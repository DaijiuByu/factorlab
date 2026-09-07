import tempfile
import unittest
from pathlib import Path

import pandas as pd

from factorlab.storage import query_duckdb, read_dataset, write_dataset


class StorageTests(unittest.TestCase):
    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(pd.DataFrame({"a": [1, 2]}), Path(directory) / "x.csv")
            pd.testing.assert_frame_equal(read_dataset(path), pd.DataFrame({"a": [1, 2]}))

    def test_duckdb_query_is_read_only_gated(self):
        with self.assertRaises(ValueError):
            query_duckdb("missing.csv", "DELETE FROM {dataset}")
        with self.assertRaises(ValueError):
            query_duckdb("missing.csv", "SELECT * FROM missing.csv")
