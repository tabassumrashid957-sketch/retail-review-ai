"""Prepare source-grouped retail action clips from official UCF-Crime annotations.

Both labels are derived from the same retail source videos. Original annotation
indices are 1-based inclusive; generated clip indices are 0-based half-open.
This creates a custom experiment from originally annotated UCF test videos, not
an official UCF-Crime benchmark train/test protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data/real_retail/ucf_crime"
FIELDNAMES = ["id", "path", "label", "group", "split", "duration", "fps", "source_video", "start_frame", "end_frame", "source_sha256", "clip_sha256"]


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Convert official 1-based inclusive event bounds to [start-1, end)."""
    result = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 2 or fields[1].lower() != "shoplifting":
            continue
        name = fields[0]
        if Path(name).name != name or not name.lower().endswith(".mp4"):
            raise ValueError(f"Unsafe or unsupported video name on annotation line {number}: {name}")
        if name in result:
            raise ValueError(f"Duplicate annotation row for {name}")
        try:
            coordinates = [int(value) for value in fields[2:]]
        except ValueError as error:
            raise ValueError(f"Non-integer annotation on line {number}") from error
        if len(coordinates) % 2:
            raise ValueError(f"Unpaired frame annotation on line {number}")
        intervals = []
        for start, end in zip(coordinates[::2], coordinates[1::2]):
            if (start, end) == (-1, -1):
                continue
            if start < 1 or end < start:
                raise ValueError(f"Invalid 1-based inclusive event on line {number}: {start}, {end}")
            intervals.append((start - 1, end))
        if not intervals:
            raise ValueError(f"No positive event annotation for {name}")
        result[name] = merge_intervals(intervals)
    if not result:
        raise ValueError("No Shoplifting rows found in the annotation file.")
    return result


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(events, total_frames: int, margin: int) -> list[tuple[int, int]]:
    excluded = merge_intervals([(max(0, start - margin), min(total_frames, end + margin)) for start, end in events])
    result, cursor = [], 0
    for start, end in excluded:
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_frames:
        result.append((cursor, total_frames))
    return result


def candidate_windows(intervals, width: int, minimum: int) -> list[tuple[int, int]]:
    """Non-overlapping full windows, plus an eligible short remainder."""
    result = []
    for start, end in intervals:
        cursor = start
        while cursor + width <= end:
            result.append((cursor, cursor + width))
            cursor += width
        if end - cursor >= minimum:
            result.append((cursor, end))
    return result


def evenly_spaced(values, count: int):
    if count < 1 or count > len(values):
        raise ValueError("Invalid number of temporal windows selected.")
    if count == len(values):
        return list(values)
    if count == 1:
        return [values[(len(values) - 1) // 2]]
    return [values[round(position * (len(values) - 1) / (count - 1))] for position in range(count)]


def video_details(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid source video metadata: {path}")
        return {"fps": fps, "frames": frames, "width": width, "height": height}
    finally:
        capture.release()


def write_clip(source: Path, target: Path, start: int, end: int, details: dict) -> None:
    """Decode the requested source frames; never modify or overwrite a source."""
    if source.resolve() == target.resolve():
        raise ValueError("A derived clip cannot overwrite its source video.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".partial.mp4")
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), details["fps"], (details["width"], details["height"]))
    count = 0
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise ValueError(f"Could not open source/MP4 writer for {source.name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Failed decoding {source.name} at frame {index}")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    if count != end - start:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Derived clip has an unexpected frame count for {source.name}")
    decoded = video_details(temporary)
    if decoded["frames"] != count:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Encoded clip frame count mismatch for {source.name}")
    temporary.replace(target)


def prepare_dataset(videos_dir: Path, annotations_file: Path, output_dir: Path, manifest_path: Path, *,
                    clip_seconds: float = 4.0, min_clip_seconds: float = 1.0,
                    max_per_class: int = 8, guard_seconds: float = 0.25,
                    negative_guard_seconds: float = 1.0, seed: int = 42,
                    min_source_groups: int = 6, allow_missing_sources: bool = False) -> dict:
    videos_dir, annotations_file, output_dir, manifest_path = [Path(path).expanduser().resolve() for path in (videos_dir, annotations_file, output_dir, manifest_path)]
    clip_seconds, min_clip_seconds, guard_seconds, negative_guard_seconds = [float(value) for value in (clip_seconds, min_clip_seconds, guard_seconds, negative_guard_seconds)]
    if not all(math.isfinite(value) for value in (clip_seconds, min_clip_seconds, guard_seconds, negative_guard_seconds)):
        raise ValueError("Temporal settings must be finite.")
    if not 0 < min_clip_seconds <= clip_seconds or guard_seconds < 0 or negative_guard_seconds < 0 or max_per_class < 1 or min_source_groups < 1:
        raise ValueError("Invalid clip duration, guard, class cap, or minimum source-group count.")
    if output_dir == videos_dir or output_dir in videos_dir.parents or videos_dir in output_dir.parents:
        raise ValueError("Raw source-video and derived-clip directories must be separate siblings.")
    if manifest_path.is_relative_to(videos_dir):
        raise ValueError("The training manifest cannot be written among raw source videos.")
    annotations = parse_annotations(annotations_file)
    missing = [name for name in annotations if not (videos_dir / name).is_file()]
    if missing and not allow_missing_sources:
        raise ValueError("Annotated source videos are missing; finish the download first:\n" + "\n".join(missing))
    config = {"schema_version": 1, "clip_seconds": clip_seconds, "minimum_clip_seconds": min_clip_seconds,
              "max_per_class_per_source": max_per_class, "positive_interior_guard_seconds": guard_seconds,
              "negative_event_guard_seconds": negative_guard_seconds, "seed": seed,
              "selection": "deterministic evenly spaced non-overlapping windows; downstream seed is recorded",
              "annotation_convention": "official 1-based inclusive [start,end] converted to Python 0-based half-open [start-1,end)",
              "duration_matched_pairs": True}
    plans, skipped, source_hashes = [], [], {}
    for name, events in sorted(annotations.items()):
        source = videos_dir / name
        if not source.is_file():
            skipped.append({"source_video": name, "reason": "source missing"})
            continue
        details = video_details(source)
        if any(start < 0 or end > details["frames"] for start, end in events):
            raise ValueError(f"Annotation exceeds actual frames for {name}: {events}; video has {details['frames']} frames. Check that this is the original video variant.")
        source_hashes[name] = file_sha256(source)
        fps = details["fps"]
        width = max(1, round(clip_seconds * fps))
        minimum = max(1, math.ceil(min_clip_seconds * fps))
        positive_guard = math.ceil(guard_seconds * fps)
        negative_guard = math.ceil(negative_guard_seconds * fps)
        positive_ranges = [(start + positive_guard, end - positive_guard) for start, end in events if end - start - 2 * positive_guard >= minimum]
        positive = candidate_windows(positive_ranges, width, minimum)
        normal_ranges = complement_intervals(events, details["frames"], negative_guard)
        normal = candidate_windows(normal_ranges, width, minimum)
        count = min(max_per_class, len(positive), len(normal))
        if not count:
            skipped.append({"source_video": name, "reason": "insufficient eligible positive or normal context", "positive_candidates": len(positive), "normal_candidates": len(normal)})
            continue
        positive, normal = evenly_spaced(positive, count), evenly_spaced(normal, count)
        for positive_window, normal_window in zip(positive, normal):
            # Match each normal clip's duration to its positive partner so the
            # label cannot be predicted solely from sampling speed or length.
            common = min(positive_window[1] - positive_window[0], normal_window[1] - normal_window[0])
            for label, interval in [(0, normal_window), (1, positive_window)]:
                start = interval[0] + (interval[1] - interval[0] - common) // 2
                plans.append({"source": source, "source_video": name, "source_sha256": source_hashes[name], "details": details,
                              "label": label, "start_frame": start, "end_frame": start + common})
    groups = {plan["source_video"] for plan in plans}
    if len(groups) < min_source_groups:
        raise ValueError(f"Only {len(groups)} eligible source-video groups are available; at least {min_source_groups} are required. No training manifest was written.")
    signature = {"config": config, "annotation_sha256": file_sha256(annotations_file), "source_hashes": source_hashes}
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "preparation-config.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != signature:
            raise ValueError("Source data, annotations or preparation settings changed. Use a new --output folder; existing derived clips were not overwritten.")
    elif any(output_dir.iterdir()):
        raise ValueError("The output directory contains files without a matching preparation configuration. Use an empty --output folder.")
    else:
        atomic_json(plan_path, signature)
    configuration_sha = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    rows, reused = [], 0
    for number, plan in enumerate(plans, 1):
        kind = "normal" if plan["label"] == 0 else "positive"
        clip_id = f"{plan['source_sha256'][:16]}-{kind}-{plan['start_frame']}-{plan['end_frame']}"
        target = output_dir / ("Normal" if plan["label"] == 0 else "Shoplifting") / f"{clip_id}.mp4"
        sidecar = target.with_suffix(".json")
        expected = {"configuration_sha256": configuration_sha, "source_sha256": plan["source_sha256"], "start_frame": plan["start_frame"], "end_frame": plan["end_frame"]}
        cached = False
        if target.is_file() and sidecar.is_file():
            previous = json.loads(sidecar.read_text())
            if any(previous.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Cached clip metadata does not match its source/configuration: {target}")
            cached = previous.get("clip_sha256") == file_sha256(target)
        if cached:
            reused += 1
            clip_sha = previous["clip_sha256"]
        else:
            write_clip(plan["source"], target, plan["start_frame"], plan["end_frame"], plan["details"])
            clip_sha = file_sha256(target)
            atomic_json(sidecar, {**expected, "clip_sha256": clip_sha})
        rows.append({"id": clip_id, "path": str(target), "label": plan["label"], "group": "ucf:" + plan["source_video"], "split": "",
                     "duration": round((plan["end_frame"] - plan["start_frame"]) / plan["details"]["fps"], 9), "fps": plan["details"]["fps"],
                     "source_video": plan["source_video"], "start_frame": plan["start_frame"], "end_frame": plan["end_frame"],
                     "source_sha256": plan["source_sha256"], "clip_sha256": clip_sha})
        print(f"[{number}/{len(plans)}] {'cached' if cached else 'prepared'} {clip_id}", flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)
    metadata = {"schema_version": 1, "complete": True, "config": config, "source_directory": str(videos_dir), "annotations": str(annotations_file),
                "annotation_sha256": signature["annotation_sha256"], "manifest": str(manifest_path), "clip_directory": str(output_dir),
                "source_count": len(groups), "annotated_source_count": len(annotations), "clip_count": len(rows), "normal_clips": sum(row["label"] == 0 for row in rows),
                "positive_clips": sum(row["label"] == 1 for row in rows), "cached_clips_reused": reused, "skipped_sources": skipped,
                "grouping": "All derived clips from one original source video share one group and must stay in one training/validation/test split.",
                "protocol": "Custom source-video-group-disjoint experiment constructed from the originally annotated UCF-Crime test subset. Not an official UCF-Crime benchmark result.",
                "normal_label": "Outside official annotated anomaly intervals, with temporal guard margins; this does not independently verify intent or payment.",
                "positive_label": "Inside official author-annotated Shoplifting intervals; not an independent finding of theft.",
                "geography": "Store identities/countries and representative worldwide coverage are not established by this preparation.",
                "source_hashes": source_hashes, "clips": rows}
    atomic_json(output_dir.parent / "prepare_metadata.json", metadata)
    print(f"Prepared {len(rows)} balanced clips from {len(groups)} source videos. Manifest: {manifest_path}", flush=True)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=BASE / "source_videos")
    parser.add_argument("--annotations", type=Path, default=BASE / "annotations.txt")
    parser.add_argument("--output", type=Path, default=BASE / "clips")
    parser.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--min-clip-seconds", type=float, default=1.0)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--negative-guard-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-source-groups", type=int, default=6)
    parser.add_argument("--allow-missing-sources", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepare_dataset(args.videos, args.annotations, args.output, args.manifest,
                        clip_seconds=args.clip_seconds, min_clip_seconds=args.min_clip_seconds,
                        max_per_class=args.max_per_class, guard_seconds=args.guard_seconds,
                        negative_guard_seconds=args.negative_guard_seconds, seed=args.seed,
                        min_source_groups=args.min_source_groups, allow_missing_sources=args.allow_missing_sources)
    except (ValueError, OSError, cv2.error, json.JSONDecodeError) as error:
        print(f"Preparation stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
