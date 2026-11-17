import csv
import json
from pathlib import Path
import tempfile
import unittest
from shoplifting.data_config import active_dataset, rebase_manifest
from scripts.run_pipeline import build_commands


class ActiveDatasetTests(unittest.TestCase):
    def fixture(self, root):
        (root / "data").mkdir()
        config = {"schema_version": 1, "name": "fixture", "kind": "ucf_crime", "manifest": "data/manifest.csv", "provenance": "data/provenance.json", "source_dir": "data/sources", "clips_dir": "data/clips", "annotations": "data/annotations.txt", "feature_dir": "artifacts/features"}
        (root / "data/active_dataset.json").write_text(json.dumps(config))
        return config

    def test_relocation_preserves_labels_and_original_source_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            clip = root / "data/clips/Normal/example.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"fixture")
            manifest = root / config["manifest"]
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "path", "label", "group"])
                writer.writeheader()
                writer.writerow({"id": "one", "path": "/another/laptop/example.mp4", "label": 0, "group": "ucf:one.mp4"})
            rows = rebase_manifest(manifest, root)
            self.assertEqual(rows[0]["path"], str(clip.resolve()))
            self.assertEqual(rows[0]["group"], "ucf:one.mp4")
            self.assertEqual(rows[0]["label"], "0")

    def test_ledger_cannot_reference_files_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            config["manifest"] = "../outside.csv"
            (root / "data/active_dataset.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                active_dataset(root)

    def test_audited_manifest_skips_generic_group_resetting_preparation(self):
        root = Path("/project")
        commands = build_commands(project_root=root, interpreter="python", data_dir=root / "data/clips", manifest=root / "data/manifest.csv", features=root / "artifacts/features", candidate=root / "artifacts/candidate", manifest_ready=True)
        self.assertEqual(len(commands), 2)
        self.assertIn("shoplifting.extract", commands[0][1])
        self.assertNotIn("shoplifting.prepare", str(commands))

    def test_context_reproduction_preserves_ordered_features_and_validated_comparison(self):
        root = Path("/project")
        commands = build_commands(project_root=root, interpreter="python", data_dir=root / "data/clips", manifest=root / "manifest.csv", features=root / "features", candidate=root / "candidate", manifest_ready=True, head_type="temporal_gru", hidden_dim=64, feature_config={"window_seconds": 2, "stride_seconds": 1, "max_windows": 0}, frozen_transfer=root / "baseline/model.pt")
        self.assertEqual(len(commands), 3)
        self.assertIn("--max-windows", commands[0][1]); self.assertIn("0", commands[0][1])
        self.assertIn("temporal_gru", commands[1][1]); self.assertIn("64", commands[1][1])
        self.assertIn("select_context_model.py", commands[2][1][1])


if __name__ == "__main__":
    unittest.main()
