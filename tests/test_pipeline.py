from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline import SpamDetectionPipeline
from src.core.config import load_config


class PipelineInitTests(unittest.TestCase):
    """Test SpamDetectionPipeline constructor and basic attributes."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        cfg_path = Path(self.tmp) / "pipeline.yaml"
        cfg_path.write_text("{}\n", encoding="utf-8")
        self.cfg = load_config(cfg_path)

    def test_init_with_config_only(self) -> None:
        pipeline = SpamDetectionPipeline(self.cfg)
        self.assertIsNotNone(pipeline.run_id)
        self.assertIsInstance(pipeline.run_id, str)

    def test_init_with_custom_run_id(self) -> None:
        pipeline = SpamDetectionPipeline(self.cfg, run_id="test_run_001")
        self.assertEqual(pipeline.run_id, "test_run_001")

    def test_init_without_args_uses_defaults(self) -> None:
        # Should not raise — config defaults kick in
        pipeline = SpamDetectionPipeline()
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.run_id)

    def test_dirs_created_on_init(self) -> None:
        """_prepare_dirs should create required directories relative to config paths."""
        pipeline = SpamDetectionPipeline(self.cfg)
        # candidates and archive dirs should exist after init
        self.assertTrue(pipeline.config.model.candidates_dir.exists())
        self.assertTrue(pipeline.config.model.archive_dir.exists())


class EmitRunManifestTests(unittest.TestCase):
    """Test emit_run_manifest writes correctly structured JSON files."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        cfg_path = self.tmp / "pipeline.yaml"
        # Point run_manifest_dir inside our temp dir so no global side effects
        cfg_path.write_text(
            f"data:\n  run_manifest_dir: {self.tmp / 'runs'}\n",
            encoding="utf-8",
        )
        self.cfg = load_config(cfg_path)
        self.pipeline = SpamDetectionPipeline(self.cfg, run_id="test_manifest")

    def test_emit_success_manifest(self) -> None:
        result = {"predictions": 42}
        manifest_path = self.pipeline.emit_run_manifest(
            command="infer", status="success", result=result
        )
        self.assertTrue(manifest_path.exists())
        with manifest_path.open() as f:
            data = json.load(f)
        self.assertEqual(data["run_id"], "test_manifest")
        self.assertEqual(data["command"], "infer")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["result"]["predictions"], 42)

    def test_emit_failed_manifest(self) -> None:
        error = RuntimeError("something went wrong")
        manifest_path = self.pipeline.emit_run_manifest(
            command="train", status="failed", error=error
        )
        self.assertTrue(manifest_path.exists())
        with manifest_path.open() as f:
            data = json.load(f)
        self.assertEqual(data["status"], "failed")
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "RuntimeError")

    def test_manifest_filename_includes_run_id_and_command(self) -> None:
        manifest_path = self.pipeline.emit_run_manifest("dataset", status="success")
        self.assertIn("test_manifest", manifest_path.name)
        self.assertIn("dataset", manifest_path.name)


if __name__ == "__main__":
    unittest.main()
