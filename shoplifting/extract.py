"""Extract frozen action features, preserving all windows of each source video."""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .backbone import VideoEncoder
from .prepare import sha256
from .video import read_window, video_info, window_bounds


def valid_cache(path, bounds, config, source_sha256):
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (
                saved["features"].dtype == np.float32
                and saved["features"].shape == (len(bounds), 512)
                and np.isfinite(saved["features"]).all()
                and saved["starts"].shape == (len(bounds),)
                and saved["ends"].shape == (len(bounds),)
                and np.allclose(saved["starts"], [v[0] for v in bounds], atol=1e-5, rtol=0)
                and np.allclose(saved["ends"], [v[1] for v in bounds], atol=1e-5, rtol=0)
                and str(saved["config"]) == json.dumps(config, sort_keys=True)
                and str(saved["source_sha256"]) == source_sha256
            )
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=24, help="0 keeps all windows; other values sample across the full video.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default="artifacts/torch")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("manifest is empty")
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("manifest has duplicate video IDs; regenerate it before extracting features")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"backbone": "r3d_18", "weights": "KINETICS400_V1", "num_frames": 16, "window_seconds": args.window_seconds, "stride_seconds": args.stride_seconds, "max_windows": args.max_windows, "feature_dim": 512}
    index_path = args.output / "index.json"
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if previous.get("feature_config") != config:
            raise SystemExit("Feature settings changed. Use a different --output folder to avoid mixing cached features.")
    # Write settings before expensive work; each NPZ is saved atomically for safe resuming.
    records = []
    encoder = None
    began = time.monotonic()
    for count, row in enumerate(rows, 1):
        path = Path(row["path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        if row["label"] not in {"0", "1"}:
            raise SystemExit(f"Invalid label for {path}: expected 0 or 1.")
        file_id = row["id"]
        if not file_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in file_id):
            raise SystemExit("Video IDs must contain only letters, digits, hyphens and underscores.")
        destination = args.output / f"{file_id}.npz"
        source_digest = sha256(path)
        info = video_info(path)
        bounds = window_bounds(info["duration"], args.window_seconds, args.stride_seconds, args.max_windows)
        cached = False
        if destination.exists():
            cached = valid_cache(destination, bounds, config, source_digest)
        if not cached:
            if encoder is None:
                encoder = VideoEncoder(args.device, args.cache_dir, args.threads)
            capture = cv2.VideoCapture(str(path))
            values = []
            try:
                for begin in range(0, len(bounds), args.batch_size):
                    clips = [read_window(capture, start, end, info["fps"], info["frames"], config["num_frames"]) for start, end in bounds[begin:begin + args.batch_size]]
                    values.append(encoder.encode(clips))
            finally:
                capture.release()
            temporary = args.output / f"{file_id}.partial.npz"
            np.savez_compressed(temporary, features=np.concatenate(values).astype(np.float32), starts=np.array([v[0] for v in bounds], dtype=np.float32), ends=np.array([v[1] for v in bounds], dtype=np.float32), config=json.dumps(config, sort_keys=True), source_sha256=source_digest)
            temporary.replace(destination)
        record = {"id": file_id, "path": str(path.resolve()), "label": int(row["label"]), "group": row.get("group") or file_id, "feature_file": destination.name, "source_sha256": source_digest, "duration": info["duration"]}
        if row.get("split"):
            record["split"] = row["split"]
        records.append(record)
        temporary_index = args.output / "index.partial.json"
        temporary_index.write_text(json.dumps({"schema_version": 1, "expected_video_count": len(rows), "complete": len(records) == len(rows), "feature_config": config, "records": records}, indent=2) + "\n")
        temporary_index.replace(index_path)
        print(f"[{count}/{len(rows)}] {'cached' if cached else 'encoded'} {path.name}: {len(bounds)} windows; elapsed {time.monotonic()-began:.0f}s", flush=True)
    print(f"Features ready: {index_path}")


if __name__ == "__main__":
    main()
