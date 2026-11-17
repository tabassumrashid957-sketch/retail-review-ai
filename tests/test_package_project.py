"""Small offline fixtures validate complete real-data exports."""
import csv
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts.package_project import ARCHIVE_ROOT, REQUIRED_FILES, build_package


def fixture_project(root):
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    data = root / "data/real_retail/ucf_crime"
    for relative in ["source_videos/source.mp4", "clips/Normal/normal.mp4", "clips/Shoplifting/event.mp4", "provenance.json", "annotations.txt", "sources.json", "prepare_metadata.json"]:
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    ledger = {"name": "UCF-Crime retail subset", "kind": "ucf_crime", "manifest": "data/real_retail/ucf_crime/manifest.csv", "provenance": "data/real_retail/ucf_crime/provenance.json", "source_dir": "data/real_retail/ucf_crime/source_videos", "clips_dir": "data/real_retail/ucf_crime/clips", "annotations": "data/real_retail/ucf_crime/annotations.txt", "feature_dir": "artifacts/features"}
    (root / "data/active_dataset.json").write_text(json.dumps(ledger))
    with (data / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "path", "label", "group"])
        writer.writeheader()
        writer.writerows([{"id": "normal", "path": "clips/Normal/normal.mp4", "label": "0", "group": "ucf:source.mp4"}, {"id": "event", "path": "clips/Shoplifting/event.mp4", "label": "1", "group": "ucf:source.mp4"}])
    features = root / "artifacts/features"
    features.mkdir(parents=True, exist_ok=True)
    (features / "example.npz").write_bytes(b"features")
    (features / "index.json").write_text(json.dumps({"complete": True, "expected_video_count": 1, "records": [{"id": "example", "feature_file": "example.npz"}]}))
    (root / "examples").mkdir()
    (root / "examples/demo.mp4").write_bytes(b"example")
    for name in ["SETUP.command", "BUILD_MODEL.command", "START_UI.command"]:
        (root / name).chmod(0o755)


class PackageProjectTests(unittest.TestCase):
    def test_only_ui_entry_is_packaged_even_when_old_launchers_and_configs_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            stale = ["START_HERE.command", "RUN_DEMO.command", "run_demo.py",
                     "shoplifting/launcher.py", "tests/test_launcher.py",
                     ".run/Shoplifting_Demo.run.xml", ".run/Choose_Video.run.xml",
                     ".run/Other.run.xml"]
            for name in stale:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"obsolete entry\n")
            with zipfile.ZipFile(build_package(root)) as archive:
                names = set(archive.namelist())
                for name in stale:
                    self.assertNotIn(f"{ARCHIVE_ROOT}/{name}", names)
                configurations = {name for name in names if "/.run/" in name}
                self.assertEqual(configurations, {f"{ARCHIVE_ROOT}/.run/Retail_Review_UI.run.xml"})
                for name in ["START_UI.command", "run_ui.py", "detect.py", "shoplifting/predict.py"]:
                    self.assertIn(f"{ARCHIVE_ROOT}/{name}", names)

    def test_ui_sources_are_included_but_uploaded_clips_and_reviews_are_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_project(root)
            private=root/"outputs/ui/private-case";private.mkdir(parents=True)
            (private/"input.mp4").write_bytes(b"private video")
            (private/"job.json").write_text('{"reviews":"private notes"}')
            with zipfile.ZipFile(build_package(root)) as archive:
                names=archive.namelist()
                self.assertIn(f"{ARCHIVE_ROOT}/ui/index.html",names)
                self.assertIn(f"{ARCHIVE_ROOT}/ui/app.js",names)
                self.assertFalse(any("/outputs/ui/" in name for name in names))
                self.assertEqual(stat.S_IMODE(archive.getinfo(f"{ARCHIVE_ROOT}/START_UI.command").external_attr >> 16),0o755)
    def test_relocated_manifest_uses_local_active_clips_without_rewriting_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            manifest = root / "data/real_retail/ucf_crime/manifest.csv"
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["path"] = "/another/computer/project/" + row["path"]
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "path", "label", "group"])
                writer.writeheader()
                writer.writerows(rows)
            original = manifest.read_bytes()
            output = build_package(root)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read(f"{ARCHIVE_ROOT}/data/real_retail/ucf_crime/manifest.csv"), original)
            self.assertEqual(manifest.read_bytes(), original)

    def test_real_data_models_pycharm_and_launcher_modes_included_without_old_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            extras = {"exports/old/README.md": b"old", "downloads/clone/detect.py": b"old", "data/raw/mnnit/old.mp4": b"staged", ".idea/workspace.xml": b"private", ".venv/site.py": b"environment", "shoplifting/__pycache__/old.pyc": b"cache", ".run/notes.xml": b"not configuration", ".run/nested/Other.run.xml": b"not canonical", "outputs/example/prediction.json": b"{}", "outputs/example/annotated-video.mp4": b"example", "LICENSE": b"credits", "examples/README.md": b"how to watch", "report_support/latex/figure01.png": b"source figure", "report_support/qa/page-1.png": b"QA render", "report_support/preview.png": b"QA preview", "report_support/latex/main.aux": b"compiler intermediary"}
            for relative, content in extras.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            portable_inspections = [".idea/inspectionProfiles/Project_Default.xml", ".idea/inspectionProfiles/profiles_settings.xml"]
            for relative in portable_inspections:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("<component/>\n")
            before = (root / "data/active_dataset.json").read_bytes()
            output = build_package(root)
            self.assertEqual(output, (root / "exports/Final_AI_Complete.zip").resolve())
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertEqual(names, sorted(names))
                for name in REQUIRED_FILES:
                    self.assertIn(f"{ARCHIVE_ROOT}/{name}", names)
                for name in ["data/real_retail/ucf_crime/source_videos/source.mp4", "data/real_retail/ucf_crime/clips/Normal/normal.mp4", "data/real_retail/ucf_crime/prepare_metadata.json", "artifacts/features/example.npz", "outputs/example/annotated-video.mp4", "examples/demo.mp4", "examples/README.md", "report_support/latex/figure01.png", "LICENSE"]:
                    self.assertIn(f"{ARCHIVE_ROOT}/{name}", names)
                self.assertFalse(any(any(part in name for part in ["/exports/", "/downloads/", "/.venv/", "/__pycache__/", "/data/raw/mnnit/"]) for name in names))
                self.assertEqual({name for name in names if "/.idea/" in name}, {f"{ARCHIVE_ROOT}/{name}" for name in portable_inspections})
                self.assertNotIn(f"{ARCHIVE_ROOT}/.run/notes.xml", names)
                self.assertNotIn(f"{ARCHIVE_ROOT}/.run/nested/Other.run.xml", names)
                for name in ["report_support/qa/page-1.png", "report_support/preview.png", "report_support/latex/main.aux"]:
                    self.assertNotIn(f"{ARCHIVE_ROOT}/{name}", names)
                for name in ["START_UI.command", "SETUP.command", "BUILD_MODEL.command"]:
                    self.assertEqual(stat.S_IMODE(archive.getinfo(f"{ARCHIVE_ROOT}/{name}").external_attr >> 16), 0o755)
                self.assertEqual(archive.read(f"{ARCHIVE_ROOT}/data/active_dataset.json"), before)
                self.assertIsNone(archive.testzip())
            self.assertEqual((root / "exports/old/README.md").read_bytes(), b"old")

    def test_missing_required_inputs_do_not_replace_existing_archive(self):
        for missing in ["artifacts/model/model.pt", "artifacts/model/mil_model.pt", "run_ui.py", ".run/Retail_Review_UI.run.xml", "data/active_dataset.json", "data/real_retail/ucf_crime/clips/Normal/normal.mp4", "artifacts/features/example.npz", "examples/demo.mp4", "report_support/Report_Outline.docx"]:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture_project(root)
                (root / missing).unlink()
                output = root / "previous.zip"
                output.write_bytes(b"preserve")
                with self.assertRaises(ValueError):
                    build_package(root, output)
                self.assertEqual(output.read_bytes(), b"preserve")

    def test_symlinked_configuration_and_data_are_rejected(self):
        for link in [".run/Retail_Review_UI.run.xml", "data/real_retail/ucf_crime/source_videos/external.mp4"]:
            with self.subTest(link=link), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "project"
                fixture_project(root)
                outside = base / "external"
                outside.write_bytes(b"private")
                (root / link).unlink(missing_ok=True)
                (root / link).symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    build_package(root)

    def test_active_ledger_cannot_point_outside_real_data_or_to_staged_data(self):
        for value in ["../outside.csv", "/outside.csv", "data/raw/mnnit/manifest.csv"]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture_project(root)
                path = root / "data/active_dataset.json"
                ledger = json.loads(path.read_text())
                ledger["manifest"] = value
                path.write_text(json.dumps(ledger))
                with self.assertRaises(ValueError):
                    build_package(root)

    def test_feature_index_escape_or_interruption_is_rejected(self):
        for index in [{"complete": False, "records": []}, {"records": [{"feature_file": "../escape.npz"}]}, {"expected_video_count": 3, "records": [{"feature_file": "example.npz"}]}]:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture_project(root)
                (root / "artifacts/features/index.json").write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    build_package(root)

    def test_output_cannot_overwrite_data_or_enter_old_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            with self.assertRaises(ValueError):
                build_package(root, root / "data/real_retail/new.zip")
            with self.assertRaises(ValueError):
                build_package(root, root / "exports/old/project.zip")

    def test_crc_failure_keeps_previous_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            output = root / "previous.zip"
            output.write_bytes(b"preserve")
            with patch("scripts.package_project.zipfile.ZipFile.testzip", return_value="bad"):
                with self.assertRaisesRegex(ValueError, "CRC"):
                    build_package(root, output)
            self.assertEqual(output.read_bytes(), b"preserve")
            self.assertFalse(list(root.glob(".final-ai-package-*.zip")))


if __name__ == "__main__":
    unittest.main()
