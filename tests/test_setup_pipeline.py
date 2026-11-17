"""Offline checks for setup diagnostics and non-destructive pipeline orchestration."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_pipeline as pipeline
from scripts import verify_project as verifier


class VerificationTests(unittest.TestCase):
    def test_missing_environment_and_imports_give_actionable_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = [{"name": "torch", "status": "missing", "detail": "not installed", "action": "Run SETUP.command"}]
            with patch.object(verifier, "dependency_status", return_value=missing):
                result = verifier.inspect_project(root, dependencies_only=True)
            self.assertFalse(result["ready"])
            self.assertTrue(any("SETUP.command" in item for item in result["problems"]))
            self.assertFalse((root / ".venv").exists())

    def test_inference_does_not_require_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = {"name": "local_file", "status": "ready", "detail": "verified", "action": None}
            with patch.object(verifier.sys, "prefix", str(root / ".venv")), \
                 patch.object(verifier, "check_model", return_value=ready), \
                 patch.object(verifier, "check_encoder", return_value=ready):
                result = verifier.inspect_project(root, check_dependencies=False)
            self.assertTrue(result["ready"])
            self.assertTrue(result["inference_ready"])
            self.assertFalse(result["training_ready"])
            self.assertEqual(result["data"]["counts"]["normal"], 0)
            self.assertEqual(result["problems"], [])

    def test_label_detection_prioritises_non_shoplifting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("Non-Shoplifting/negative.mp4", "Shoplifting/positive.mp4", "Unknown/unlabelled.mp4"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")
            status = verifier.inspect_data(root)
            self.assertTrue(status["labelled_videos_available"])
            self.assertEqual(status["counts"], {"normal": 1, "positive": 1, "unlabelled": 1})

    def test_missing_encoder_and_incomplete_archive_fail_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(verifier.check_encoder(root / "missing.pth")["status"], "missing")
            archive = root / "Dataset.zip"
            archive.write_bytes(b"incomplete")
            with patch.object(pipeline.subprocess, "run") as command:
                with self.assertRaisesRegex(ValueError, "complete Dataset.zip"):
                    pipeline.verify_local_archive(archive)
                command.assert_not_called()


class PipelineTests(unittest.TestCase):
    def make_root(self, temporary):
        root = Path(temporary) / "project with spaces"
        interpreter = root / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("test interpreter placeholder")
        current = root / "artifacts/model"
        current.mkdir(parents=True)
        (current / "model.pt").write_bytes(b"original working model")
        return root.resolve()

    def runtime_mocks(self):
        stack = ExitStack()
        stack.enter_context(patch.object(pipeline.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "{}", "")))
        stack.enter_context(patch.object(pipeline, "check_encoder", return_value={"status": "ready"}))
        stack.enter_context(patch.object(pipeline, "inspect_data", return_value={"labelled_videos_available": True}))
        return stack

    @staticmethod
    def write_candidate(command):
        if "shoplifting.train" in command:
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True)
            for name in ("model.pt", "metrics.json", "split_manifest.json", "test_predictions.csv", "mean_logistic.pkl"):
                (destination / name).write_bytes(b"new candidate artifact")

    def test_commands_keep_paths_literal_and_archive_offline_by_default(self):
        root = Path("/tmp/project with spaces")
        data = root / "Normal; $(touch unrelated)"
        arguments = dict(project_root=root, interpreter=str(root / ".venv/bin/python"), data_dir=data,
                         manifest=root / "data/manifest.csv", features=root / "artifacts/features",
                         candidate=root / "artifacts/candidate", extract_archive=True)
        commands = pipeline.build_commands(**arguments)
        self.assertIn("--offline", commands[0][1])
        self.assertIn(str(data), commands[1][1])
        self.assertTrue(all(isinstance(command, list) for _, command in commands))
        download = pipeline.build_commands(**arguments, allow_download=True)
        self.assertNotIn("--offline", download[0][1])

    def test_successful_default_run_preserves_working_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            observed = []
            def fake_step(title, command, cwd, log):
                observed.append((command, cwd))
                self.write_candidate(command)
            with self.runtime_mocks(), patch.object(pipeline, "run_step", side_effect=fake_step):
                result = pipeline.run_pipeline(project_root=root, epochs=2)
            self.assertEqual((root / "artifacts/model/model.pt").read_bytes(), b"original working model")
            self.assertTrue((root / "artifacts/model-new/model.pt").is_file())
            provenance = json.loads((root / "artifacts/model-new/provenance.json").read_text())
            self.assertEqual(provenance["artifact_sha256"]["model.pt"], pipeline.hashlib.sha256(b"new candidate artifact").hexdigest())
            self.assertTrue(Path(provenance["training_log"]).is_file())
            self.assertEqual(result["model_directory"], str(root / "artifacts/model-new"))
            self.assertEqual(len(observed), 3)
            self.assertTrue(all(cwd == root for _, cwd in observed))
            self.assertTrue(all(command[0] == str(root / ".venv/bin/python") for command, _ in observed))
            self.assertTrue(all("download_data.py" not in " ".join(command) for command, _ in observed))

    def test_training_failure_never_replaces_current_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            def failing_step(title, command, cwd, log):
                if "shoplifting.train" in command:
                    raise subprocess.CalledProcessError(2, command)
            with self.runtime_mocks(), patch.object(pipeline, "run_step", side_effect=failing_step):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.run_pipeline(project_root=root)
            self.assertEqual((root / "artifacts/model/model.pt").read_bytes(), b"original working model")
            self.assertFalse((root / "artifacts/model-new").exists())

    def test_output_and_log_paths_cannot_overwrite_working_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            with self.runtime_mocks(), patch.object(pipeline, "run_step") as execute:
                with self.assertRaisesRegex(ValueError, "separate from artifacts/model"):
                    pipeline.run_pipeline(project_root=root, output=root / "artifacts/model")
                with self.assertRaisesRegex(ValueError, "Log path must be outside"):
                    pipeline.run_pipeline(project_root=root, log_path=root / "artifacts/model/model.pt")
                execute.assert_not_called()
            self.assertEqual((root / "artifacts/model/model.pt").read_bytes(), b"original working model")

    def test_missing_local_data_does_not_implicitly_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            with self.runtime_mocks(), \
                 patch.object(pipeline, "inspect_data", return_value={"labelled_videos_available": False}), \
                 patch.object(pipeline, "run_step") as execute:
                with self.assertRaisesRegex(ValueError, "explicitly use --download-data"):
                    pipeline.run_pipeline(project_root=root)
                execute.assert_not_called()

    def test_explicit_promotion_keeps_previous_model_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            with self.runtime_mocks(), patch.object(pipeline, "run_step", side_effect=lambda title, command, cwd, log: self.write_candidate(command)):
                result = pipeline.run_pipeline(project_root=root, promote=True)
            self.assertEqual((root / "artifacts/model/model.pt").read_bytes(), b"new candidate artifact")
            self.assertEqual((Path(result["backup"]) / "model.pt").read_bytes(), b"original working model")
            self.assertFalse((root / "artifacts/model-new").exists())

    def test_promotion_rolls_back_failed_candidate_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_root(temporary)
            candidate = root / "artifacts/candidate"
            candidate.mkdir()
            (candidate / "model.pt").write_bytes(b"candidate")
            original_rename = Path.rename
            def fail_candidate(path, destination):
                if path == candidate:
                    raise OSError("simulated rename failure")
                return original_rename(path, destination)
            with patch.object(Path, "rename", new=fail_candidate):
                with self.assertRaisesRegex(OSError, "simulated"):
                    pipeline.promote_candidate(candidate, root / "artifacts/model")
            self.assertEqual((root / "artifacts/model/model.pt").read_bytes(), b"original working model")
            self.assertTrue((candidate / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
