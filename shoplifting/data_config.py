"""Project-relative dataset paths, including after the project is moved."""
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def active_dataset(root=PROJECT_ROOT):
    root = Path(root).resolve()
    ledger = root / "data/active_dataset.json"
    if not ledger.is_file():
        return None
    config = json.loads(ledger.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported active dataset metadata version.")
    for key in ("manifest", "provenance", "source_dir", "clips_dir", "annotations", "feature_dir"):
        value = config.get(key)
        if not isinstance(value, str) or Path(value).is_absolute() or not (root / value).resolve().is_relative_to(root):
            raise ValueError(f"Active dataset {key} must be a path inside the project.")
    return config


def local_clip_path(record, root=PROJECT_ROOT, config=None):
    root = Path(root).resolve()
    config = config or active_dataset(root)
    original = Path(record["path"])
    candidates = []
    if config:
        category = "Normal" if int(record["label"]) == 0 else "Shoplifting"
        candidates.append(root / config["clips_dir"] / category / original.name)
    if not original.is_absolute():
        candidates.append(root / original)
    elif original.is_relative_to(root):
        candidates.append(original)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Local video is missing: {original.name}. Restore the project's prepared video clips.")


def rebase_manifest(manifest, root=PROJECT_ROOT):
    """Rebase only file paths; labels, source groups and temporal provenance remain intact."""
    manifest = Path(manifest)
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        rows = list(reader)
    if not names or not rows:
        raise ValueError("Dataset manifest is empty.")
    config = active_dataset(root)
    changed = False
    for row in rows:
        local = str(local_clip_path(row, root, config))
        if row["path"] != local:
            row["path"] = local
            changed = True
    if changed:
        temporary = manifest.with_suffix(".rebased.csv")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(manifest)
    return rows
