import csv
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from scripts.prepare_ucf_crime import candidate_windows, complement_intervals, file_sha256, parse_annotations, prepare_dataset


def make_video(path, *, frames=120, fps=10, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Video writer unavailable for fixture")
    for index in range(frames):
        frame = np.full((48, 64, 3), (index + offset) % 220, dtype=np.uint8)
        frame[:, (index % 50):(index % 50) + 10, 1] = 230
        writer.write(frame)
    writer.release()


class UCFPreparationTests(unittest.TestCase):
    def test_official_one_based_inclusive_annotation_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.txt"
            path.write_text("Shoplifting001_x264.mp4 Shoplifting 11 20 -1 -1\nRobbery001.mp4 Robbery 1 4 -1 -1\n")
            self.assertEqual(parse_annotations(path), {"Shoplifting001_x264.mp4": [(10, 20)]})
            path.write_text("Shoplifting001_x264.mp4 Shoplifting 0 20 -1 -1\n")
            with self.assertRaises(ValueError):
                parse_annotations(path)

    def test_normal_complement_excludes_annotation_and_guard(self):
        self.assertEqual(complement_intervals([(40, 80)], 120, 10), [(0, 30), (90, 120)])
        self.assertEqual(candidate_windows([(0, 30)], 20, 10), [(0, 20), (20, 30)])

    def test_grouping_balancing_guard_boundaries_and_cached_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            videos, output = base / "source", base / "derived"
            annotations = base / "annotations.txt"
            lines, hashes = [], {}
            for index in range(6):
                name = f"Shoplifting{index:03d}_x264.mp4"
                make_video(videos / name, offset=index * 10)
                hashes[name] = file_sha256(videos / name)
                lines.append(f"{name} Shoplifting 41 80 -1 -1")
            annotations.write_text("\n".join(lines) + "\n")
            manifest = base / "manifest.csv"
            result = prepare_dataset(videos, annotations, output, manifest, clip_seconds=2, max_per_class=2)
            self.assertEqual(result["source_count"], 6)
            self.assertEqual(result["normal_clips"], result["positive_clips"])
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({row["group"] for row in rows}), 6)
            for group in {row["group"] for row in rows}:
                members = [row for row in rows if row["group"] == group]
                self.assertEqual(sorted(float(row["duration"]) for row in members if row["label"] == "0"), sorted(float(row["duration"]) for row in members if row["label"] == "1"))
            for row in rows:
                start, end = int(row["start_frame"]), int(row["end_frame"])
                if row["label"] == "1":
                    self.assertGreaterEqual(start, 43)
                    self.assertLessEqual(end, 77)
                else:
                    self.assertTrue(end <= 30 or start >= 90)
            clip_mtimes = {row["id"]: Path(row["path"]).stat().st_mtime_ns for row in rows}
            repeated = prepare_dataset(videos, annotations, output, manifest, clip_seconds=2, max_per_class=2)
            self.assertEqual(repeated["cached_clips_reused"], len(rows))
            self.assertEqual(clip_mtimes, {row["id"]: Path(row["path"]).stat().st_mtime_ns for row in rows})
            self.assertEqual(hashes, {name: file_sha256(videos / name) for name in hashes})
            with self.assertRaisesRegex(ValueError, "settings changed"):
                prepare_dataset(videos, annotations, output, manifest, clip_seconds=3, max_per_class=2)

    def test_short_annotated_event_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            make_video(source / "Shoplifting001_x264.mp4")
            annotations = base / "annotations.txt"
            annotations.write_text("Shoplifting001_x264.mp4 Shoplifting 41 60 -1 -1\n")
            result = prepare_dataset(source, annotations, base / "clips", base / "manifest.csv", min_source_groups=1)
            self.assertEqual(result["positive_clips"], 1)
            positives = [row for row in result["clips"] if row["label"] == 1]
            self.assertAlmostEqual(positives[0]["duration"], 1.4)

    def test_wrong_variant_and_missing_sources_fail_before_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            make_video(source / "Shoplifting001_x264.mp4")
            annotations = base / "annotations.txt"
            annotations.write_text("Shoplifting001_x264.mp4 Shoplifting 41 200 -1 -1\n")
            with self.assertRaisesRegex(ValueError, "actual frames"):
                prepare_dataset(source, annotations, base / "clips", base / "manifest.csv", min_source_groups=1)
            self.assertFalse((base / "manifest.csv").exists())
            annotations.write_text("Shoplifting002_x264.mp4 Shoplifting 41 60 -1 -1\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                prepare_dataset(source, annotations, base / "clips", base / "manifest.csv", min_source_groups=1)

    def test_insufficient_independent_sources_and_source_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            make_video(source / "Shoplifting001_x264.mp4")
            annotations = base / "annotations.txt"
            annotations.write_text("Shoplifting001_x264.mp4 Shoplifting 41 80 -1 -1\n")
            with self.assertRaisesRegex(ValueError, "at least 6"):
                prepare_dataset(source, annotations, base / "clips", base / "manifest.csv")
            with self.assertRaisesRegex(ValueError, "separate siblings"):
                prepare_dataset(source, annotations, source, base / "manifest.csv", min_source_groups=1)


if __name__ == "__main__":
    unittest.main()
