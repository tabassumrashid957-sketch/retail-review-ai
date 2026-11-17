"""Make an auditable video manifest; exact duplicate files share a split group."""
import argparse
import csv
import hashlib
from pathlib import Path
import re
from .video import video_info

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def infer_label(path, root):
    # The negative class must take priority over strings such as "Non-Shoplifting".
    components = list(reversed(path.relative_to(root).parts))
    for text in components:
        lower = text.lower()
        if re.search(r"(?:^|[^a-z])normal(?:$|[^a-z])|non[ _-]?(?:shoplift|theft|steal)|no[ _-]?theft", lower):
            return 0
        if re.search(r"shoplift|stealing|theft|abnormal", lower):
            return 1
    raise ValueError(f"Cannot infer class from {path}. Put videos under Normal/ and Shoplifting/ folders.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/mnnit"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()
    paths = sorted(path for path in args.data.rglob("*") if path.suffix.lower() in EXTENSIONS and not path.name.startswith("."))
    if not paths:
        raise SystemExit(f"No videos found in {args.data}. Run scripts/download_data.py first.")
    rows, rejected, known = [], [], {}
    for path in paths:
        try:
            label = infer_label(path, args.data)
            info = video_info(path)
            digest = sha256(path)
            if digest in known:
                if known[digest] != label:
                    raise ValueError(f"An identical video occurs with conflicting labels: {path}")
                print(f"Skipping exact duplicate: {path}", flush=True)
                continue
            known[digest] = label
            rows.append({"id": digest[:16], "path": str(path.resolve()), "label": label, "group": digest, "split": "", "duration": round(info["duration"], 6), "fps": round(info["fps"], 6)})
        except ValueError as exc:
            rejected.append(str(exc))
    if rejected:
        raise SystemExit("Manifest was not written because videos need attention:\n" + "\n".join(rejected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "path", "label", "group", "split", "duration", "fps"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["label"] == label for row in rows) for label in (0, 1)}
    print(f"Saved {len(rows)} unique videos to {args.output}; normal={counts[0]}, concealment-labelled={counts[1]}")
    print("Groups currently identify source files, not actors. If actor/session IDs are known, set the group column accordingly BEFORE training.")


if __name__ == "__main__":
    main()
