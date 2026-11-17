import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from scripts.download_ucf_crime import ARCHIVE_SIZE, download_compressed, download_member, open_range, payload_start


class Response(io.BytesIO):
    def __init__(self, data, start, end, *, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{ARCHIVE_SIZE}", "Content-Length": str(len(data))}


def fixture():
    data = b"video fixture" * 10000
    encoder = zlib.compressobj(wbits=-15)
    compressed = encoder.compress(data) + encoder.flush()
    record = {"file": "Shoplifting001_x264.mp4", "compression": 8, "header_offset": 100,
              "compressed_size": len(compressed), "size": len(data), "crc32": f"{zlib.crc32(data):08x}",
              "mirror_sha256": hashlib.sha256(data).hexdigest()}
    filename = record["file"].encode()
    header = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, 0, 0, 0, len(filename), 0)
    body_start = 100 + 30 + len(filename)
    def opener(request, **kwargs):
        beginning, ending = map(int, request.headers["Range"].split("=")[1].split("-"))
        if beginning == 100:
            return Response(header, beginning, ending)
        assert beginning == body_start
        return Response(compressed, beginning, ending)
    return data, record, header, opener


class DownloadTests(unittest.TestCase):
    def test_compressed_transfer_resumes_and_retries_only_complete_chunks(self):
        payload = bytes(range(200))
        calls = []
        failed_once = False
        def opener(request, **kwargs):
            nonlocal failed_once
            beginning, ending = map(int, request.headers["Range"].split("=")[1].split("-"))
            calls.append((beginning, ending))
            data = payload[beginning-1000:ending-1000+1]
            response = Response(data, beginning, ending)
            if not failed_once:
                failed_once = True
                response = Response(data[:10], beginning, ending)
                response.headers["Content-Length"] = str(ending-beginning+1)
            return response
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resumable.part"
            path.write_bytes(payload[:50])
            with patch("scripts.download_ucf_crime.TRANSFER_CHUNK", 64), patch("scripts.download_ucf_crime.time.sleep"):
                download_compressed(1000, len(payload), path, opener=opener, retries=2)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(calls[:2], [(1050,1113),(1050,1113)])

    def test_verified_extract_and_cached_reuse(self):
        data, record, header, opener = fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = download_member(record, output, opener=opener, retries=1)
            self.assertFalse(result["cached"])
            self.assertEqual((output / record["file"]).read_bytes(), data)
            def forbidden(*args, **kwargs):
                self.fail("Cached video must not be downloaded again")
            self.assertTrue(download_member(record, output, opener=forbidden)["cached"])

    def test_wrong_hash_does_not_publish(self):
        data, record, header, opener = fixture()
        record["mirror_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                download_member(record, output, opener=opener, retries=1)
            self.assertFalse((output / record["file"]).exists())
            self.assertEqual(list(output.iterdir()), [])

    def test_non_range_response_is_rejected_before_reading(self):
        response = Response(b"whole archive", 0, 12, status=200)
        with self.assertRaises(ValueError):
            open_range(0, 12, opener=lambda *args, **kwargs: response)
        self.assertTrue(response.closed)

    def test_unsafe_filename_is_rejected(self):
        data, record, header, opener = fixture()
        record["file"] = "../Shoplifting001_x264.mp4"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                download_member(record, Path(directory), opener=opener, retries=1)

    def test_truncated_or_wrong_header_is_rejected(self):
        data, record, header, opener = fixture()
        with self.assertRaises(ValueError):
            payload_start(header[:10], record)
        with self.assertRaises(ValueError):
            payload_start(b"x" * 30, record)


if __name__ == "__main__":
    unittest.main()
