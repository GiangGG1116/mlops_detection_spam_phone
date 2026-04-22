from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.core.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "pipeline.yaml"
            cfg_path.write_text("{}\n", encoding="utf-8")
            cfg = load_config(cfg_path)

            self.assertGreater(cfg.settings.min_train_rows, 0)
            self.assertGreaterEqual(cfg.settings.min_auc_to_promote, 0.0)
            self.assertTrue(str(cfg.data.run_manifest_dir).endswith("data/runs"))
            self.assertTrue(cfg.mlflow.enabled)
            self.assertEqual(cfg.api.port, 8000)

    def test_load_config_custom_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "pipeline.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    data:
                      run_manifest_dir: data/custom_runs
                    settings:
                      min_train_rows: 50
                      min_auc_to_promote: 0.7
                    mlflow:
                      enabled: false
                      experiment_name: custom_exp
                    api:
                      host: 127.0.0.1
                      port: 9000
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            cfg = load_config(cfg_path)
            self.assertEqual(cfg.settings.min_train_rows, 50)
            self.assertAlmostEqual(cfg.settings.min_auc_to_promote, 0.7)
            self.assertTrue(str(cfg.data.run_manifest_dir).endswith("data/custom_runs"))
            self.assertFalse(cfg.mlflow.enabled)
            self.assertEqual(cfg.mlflow.experiment_name, "custom_exp")
            self.assertEqual(cfg.api.host, "127.0.0.1")
            self.assertEqual(cfg.api.port, 9000)


if __name__ == "__main__":
    unittest.main()
