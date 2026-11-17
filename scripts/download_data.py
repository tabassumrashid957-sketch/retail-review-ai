"""Download and verify the public MNNIT staged shoplifting dataset, version 1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

URL = "https://data.mendeley.com/public-files/datasets/r3yjf35hzr/files/9353d364-5aba-4e22-8d5c-bdd844132912/file_downloaded"
SHA256 = "57657afeff84c44482327c46fdbb1f45b22e9f045bf44cdd31e368b07bebc93c"
SIZE = 760562047


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--offline", action="store_true", help="Use the existing archive only; never download anything.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "Dataset.zip"
    if not archive.exists() or archive.stat().st_size != SIZE:
        if args.offline:
            raise SystemExit(f"A complete local Dataset.zip is required in {args.output}. No download was attempted.")
        partial = args.output / "Dataset.zip.part"
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), URL], check=True)
        if partial.stat().st_size != SIZE:
            raise SystemExit("Download size mismatch; rerun to resume the partial download.")
        os.replace(partial, archive)
    print("Verifying the dataset SHA-256...", flush=True)
    if digest(archive) != SHA256:
        raise SystemExit("Dataset checksum mismatch. Remove Dataset.zip and download again.")
    destination = args.output / "mnnit"
    marker = destination / ".extracted"
    extracted_ok = marker.exists() and marker.read_text().strip() == SHA256
    with zipfile.ZipFile(archive) as bundle:
        extracted_ok = extracted_ok and all(
            (destination / member.filename).is_file() and (destination / member.filename).stat().st_size == member.file_size
            for member in bundle.infolist() if not member.is_dir()
        )
    if not extracted_ok:
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            root = destination.resolve()
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise SystemExit("Unsafe archive path rejected.")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise SystemExit("Archive contains a symbolic link; extraction stopped.")
            bundle.extractall(destination)
        marker.write_text(SHA256 + "\n")
    provenance = {"dataset": "Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad", "authors": ["Mohd. Aquib Ansari", "Dushyant Kumar Singh"], "version": 1, "doi": "10.17632/r3yjf35hzr.1", "source": "https://data.mendeley.com/datasets/r3yjf35hzr/1", "license": "CC BY 4.0", "download_url": URL, "sha256": SHA256, "bytes": SIZE, "description": "Staged normal and concealment videos. Labels do not establish unpaid exits or identify people."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified dataset: {destination}", flush=True)
    print("Video folders:")
    counts = {}
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}:
            folder = str(path.parent.relative_to(destination))
            counts[folder] = counts.get(folder, 0) + 1
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
