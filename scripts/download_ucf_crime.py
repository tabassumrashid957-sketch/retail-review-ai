"""Download only the 21 annotated retail videos from UCF's official ZIP.

The ZIP is over 100 GB. HTTP byte ranges retrieve selected members only; the
server must honour the exact range. Every extracted MP4 is checked against its
ZIP size/CRC and an independently recorded SHA-256 before it is published.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import struct
import time
import urllib.request
import zlib

import certifi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
ARCHIVE_SIZE = 102957372377
ANNOTATION_URL = "https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/master/Temporal_Anomaly_Annotation.txt"
DEFAULT_DIRECTORY = PROJECT_ROOT / "data/real_retail/ucf_crime"
TRANSFER_CHUNK = 4 * 1024 * 1024


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_range(start: int, end: int, *, opener=urllib.request.urlopen):
    if not 0 <= start <= end < ARCHIVE_SIZE:
        raise ValueError("Invalid official archive byte range")
    request = urllib.request.Request(ARCHIVE_URL, headers={
        "Range": f"bytes={start}-{end}", "User-Agent": "RetailResearchDownloader/1.0",
        "Accept-Encoding": "identity", "Cache-Control": "no-cache",
    })
    response = opener(request, timeout=45, context=ssl.create_default_context(cafile=certifi.where()))
    expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise ValueError("The official server did not honour the requested byte range; refusing a full-archive download.")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != end - start + 1:
        response.close()
        raise ValueError("Byte-range Content-Length does not match the selected member.")
    return response


def payload_start(header: bytes, record: dict) -> int:
    if len(header) != 30:
        raise ValueError("Truncated ZIP local header")
    signature, version, flags, compression, modified_time, modified_date, crc, packed_size, size, filename_length, extra_length = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50 or flags & 1 or compression != 8 or record["compression"] != 8:
        raise ValueError("Unexpected or encrypted ZIP member header")
    if not 0 < filename_length <= 4096:
        raise ValueError("Invalid ZIP filename length")
    return int(record["header_offset"]) + 30 + filename_length + extra_length


def validate_record(record: dict) -> None:
    filename = record.get("file", "")
    if not re.fullmatch(r"Shoplifting\d{3}_x264\.mp4", filename):
        raise ValueError(f"Unsafe or unexpected video filename: {filename!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("mirror_sha256", "")):
        raise ValueError(f"Expected SHA-256 missing for {filename}")
    if not re.fullmatch(r"[0-9a-f]{8}", record.get("crc32", "")):
        raise ValueError(f"Expected CRC32 missing for {filename}")
    if int(record["size"]) <= 0 or int(record["compressed_size"]) <= 0:
        raise ValueError("Invalid ZIP member size")


def download_compressed(beginning: int, size: int, path: Path, *, opener, retries: int) -> None:
    """Spool complete small ranges, retaining progress if a connection closes early."""
    completed = path.stat().st_size if path.exists() else 0
    if completed > size:
        path.unlink()
        completed = 0
    with path.open("ab") as stream:
        while completed < size:
            count = min(TRANSFER_CHUNK, size - completed)
            last_error = None
            for attempt in range(retries):
                try:
                    with open_range(beginning + completed, beginning + completed + count - 1, opener=opener) as response:
                        pieces, received = [], 0
                        while received < count:
                            block = response.read(count - received)
                            if not block:
                                raise ValueError("Selected byte-range transfer was truncated.")
                            pieces.append(block)
                            received += len(block)
                    data = b"".join(pieces)
                    break
                except (OSError, ValueError, http.client.HTTPException) as error:
                    last_error = error
                    if attempt + 1 < retries:
                        time.sleep(min(2 ** attempt, 4))
            else:
                raise RuntimeError(f"Could not retrieve compressed bytes {completed}-{completed+count}: {last_error}")
            stream.write(data)
            stream.flush()
            completed += count


def download_member(record: dict, destination: Path, *, opener=urllib.request.urlopen, retries: int = 3) -> dict:
    validate_record(record)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / record["file"]
    expected_hash = record["mirror_sha256"]
    if output.is_file() and output.stat().st_size == record["size"] and checksum(output) == expected_hash:
        return {"file": output.name, "bytes": output.stat().st_size, "sha256": expected_hash,
                "crc32": record["crc32"], "cached": True}
    temporary = output.with_suffix(".mp4.part")
    compressed_path = output.with_suffix(".deflate.part")
    last_error = None
    for attempt in range(retries):
        try:
            header_offset = int(record["header_offset"])
            with open_range(header_offset, header_offset + 29, opener=opener) as response:
                header = response.read(31)
            beginning = payload_start(header, record)
            compressed_size = int(record["compressed_size"])
            download_compressed(beginning, compressed_size, compressed_path, opener=opener, retries=retries)
            packed_read, bytes_written, crc = 0, 0, 0
            digest = hashlib.sha256()
            decoder = zlib.decompressobj(-15)
            with compressed_path.open("rb") as response, temporary.open("wb") as stream:
                while packed_read < compressed_size:
                    block = response.read(min(1024 * 1024, compressed_size - packed_read))
                    if not block:
                        raise ValueError("The selected compressed member was truncated.")
                    packed_read += len(block)
                    data = decoder.decompress(block)
                    bytes_written += len(data)
                    if bytes_written > record["size"]:
                        raise ValueError("Decompressed member exceeds its verified size.")
                    digest.update(data)
                    crc = zlib.crc32(data, crc)
                    stream.write(data)
                tail = decoder.flush()
                bytes_written += len(tail)
                digest.update(tail)
                crc = zlib.crc32(tail, crc)
                stream.write(tail)
            if not decoder.eof or decoder.unused_data or bytes_written != record["size"]:
                raise ValueError("The decompressed video does not match its ZIP metadata.")
            if f"{crc & 0xffffffff:08x}" != record["crc32"] or digest.hexdigest() != expected_hash:
                compressed_path.unlink(missing_ok=True)
                raise ValueError("Video CRC32 or SHA-256 verification failed.")
            os.replace(temporary, output)
            compressed_path.unlink(missing_ok=True)
            return {"file": output.name, "bytes": bytes_written, "sha256": digest.hexdigest(),
                    "crc32": record["crc32"], "cached": False}
        except (OSError, ValueError, zlib.error, http.client.HTTPException, RuntimeError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Could not obtain verified {record['file']}: {last_error}")


def fetch_annotations(directory: Path, records: list[dict]) -> Path:
    output = directory / "annotations.txt"
    if output.exists():
        text = output.read_text()
    else:
        request = urllib.request.Request(ANNOTATION_URL, headers={"User-Agent": "RetailResearchDownloader/1.0"})
        with urllib.request.urlopen(request, timeout=45, context=ssl.create_default_context(cafile=certifi.where())) as response:
            text = response.read(1024 * 1024).decode("utf-8-sig")
    if not all(record["file"] in text for record in records):
        raise ValueError("The official annotations do not include all selected source filenames.")
    output.write_text(text, encoding="utf-8")
    return output


def download(directory: Path = DEFAULT_DIRECTORY, *, workers: int = 3) -> dict:
    directory = Path(directory).resolve()
    records = json.loads((directory / "sources.json").read_text())
    if len(records) != 21 or len({record["file"] for record in records}) != 21:
        raise ValueError("Expected the verified list of 21 annotated shoplifting sources.")
    if not 1 <= workers <= 3:
        raise ValueError("Use between one and three download workers.")
    annotation = fetch_annotations(directory, records)
    destination = directory / "source_videos"
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(download_member, record, destination): record for record in records}
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(records)}] verified {result['file']} ({result['bytes']:,} bytes){' cached' if result['cached'] else ''}", flush=True)
    provenance = {
        "dataset": "UCF-Crime, temporally annotated Shoplifting subset",
        "authors": ["Waqas Sultani", "Chen Chen", "Mubarak Shah"],
        "source_page": "https://www.crcv.ucf.edu/projects/real-world/",
        "official_archive_url": ARCHIVE_URL, "official_archive_bytes": ARCHIVE_SIZE,
        "download_method": "Selective HTTP byte-range retrieval of official ZIP members; no mirror video bytes used",
        "annotation_url": ANNOTATION_URL, "annotation_sha256": checksum(annotation),
        "annotation_frame_convention": "1-based inclusive; official MATLAB evaluator uses GT(start_frame:end_frame)=1",
        "source_partition": "Original UCF-Crime test partition (annotated Shoplifting videos only)",
        "experiment_scope": "Any train/validation/test re-split of these sources is a custom retail-subset experiment, not official UCF-Crime benchmark performance.",
        "real_world_claim": "The dataset authors describe these as real-world surveillance videos. Individual incidents and payment/intent were not independently verified.",
        "geographic_coverage": "Country/store metadata is not documented in checked release materials; no worldwide representative coverage is claimed.",
        "license": "No explicit dataset redistribution licence was found in the checked primary release materials; do not label this data CC BY or MIT.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_count": len(results), "downloaded_video_bytes": sum(row["bytes"] for row in results),
        "files": sorted(results, key=lambda row: row["file"]),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified all {len(results)} official real-store source videos: {destination}", flush=True)
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        download(args.output, workers=args.workers)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Download stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
