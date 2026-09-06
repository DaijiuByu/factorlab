import json
import tempfile
import unittest
from pathlib import Path

from factorlab.data import generate_demo_panel
from factorlab.experiment import ExperimentSpec, load_experiment_spec, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_spec_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            ExperimentSpec.from_dict({"input": "panel.csv", "unexpected": True})

    def test_json_spec_runs_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.csv"
            generate_demo_panel(days=60, assets=10, seed=12).to_csv(panel_path, index=False)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": str(panel_path),
                        "output": str(root / "artifacts"),
                        "factor": "momentum",
                        "lookback": 5,
                        "quantile": 0.2,
                        "cost_bps": 5.0,
                        "min_assets": 8,
                    }
                ),
                encoding="utf-8",
            )
            spec = load_experiment_spec(config_path)
            result = run_experiment(spec)
            self.assertFalse(result.daily.empty)
            self.assertTrue((root / "artifacts" / "run_manifest.json").exists())
