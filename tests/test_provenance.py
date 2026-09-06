import unittest

from factorlab.data import generate_demo_panel
from factorlab.provenance import build_run_manifest, panel_fingerprint


class ProvenanceTests(unittest.TestCase):
    def test_panel_fingerprint_is_stable_for_row_order(self):
        panel = generate_demo_panel(days=35, assets=8, seed=12)
        shuffled = panel.sample(frac=1.0, random_state=3).reset_index(drop=True)
        self.assertEqual(panel_fingerprint(panel), panel_fingerprint(shuffled))

    def test_manifest_contains_research_identity(self):
        panel = generate_demo_panel(days=35, assets=8, seed=13)
        manifest = build_run_manifest(
            panel,
            config={"factor": "momentum", "lookback": 5},
            source="unit-test",
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["rows"], len(panel))
        self.assertEqual(manifest["assets"], 8)
        self.assertEqual(len(manifest["panel_sha256"]), 64)

