#!/usr/bin/env python3

import base64
import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("rebuild-wheel-record.py")
SPEC = importlib.util.spec_from_file_location("rebuild_wheel_record", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
REBUILD_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REBUILD_MODULE)


def wheel_hash(data):
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


class RebuildWheelRecordTest(unittest.TestCase):
    def test_rebuilds_stale_dist_info_paths_and_hashes(self):
        actual_dist_info = "torch-2.12.0+new.dist-info"
        stale_dist_info = "torch-2.12.0+old.dist-info"
        files = {
            "torch/__init__.py": b"__version__ = '2.12.0+new'\n",
            f"{actual_dist_info}/METADATA": (
                b"Metadata-Version: 2.1\nName: torch\nVersion: 2.12.0+new\n"
            ),
            f"{actual_dist_info}/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: record-test\n"
                b"Root-Is-Purelib: true\nTag: py3-none-any\n"
            ),
            f"{actual_dist_info}/licenses/LICENSE": b"fixture license\n",
        }
        record_path = f"{actual_dist_info}/RECORD"

        rows = []
        for name, data in sorted(files.items()):
            recorded_name = name
            if name.endswith("/licenses/LICENSE"):
                recorded_name = f"{stale_dist_info}/licenses/LICENSE"
            rows.append((recorded_name, wheel_hash(data), str(len(data))))
        rows.append((record_path, "", ""))

        record_buffer = io.StringIO(newline="")
        csv.writer(record_buffer, lineterminator="\n").writerows(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            wheel_path = Path(temp_dir) / "torch-2.12.0+new-py3-none-any.whl"
            with zipfile.ZipFile(
                wheel_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
                archive.writestr(
                    record_path, record_buffer.getvalue().encode("utf-8")
                )

            REBUILD_MODULE.rebuild_wheel_record(wheel_path)

            with zipfile.ZipFile(wheel_path, "r") as archive:
                names = archive.namelist()
                self.assertEqual(len(names), len(set(names)))
                rebuilt_rows = list(
                    csv.reader(
                        io.StringIO(
                            archive.read(record_path).decode("utf-8"),
                            newline="",
                        )
                    )
                )

                recorded_paths = {row[0] for row in rebuilt_rows}
                self.assertNotIn(
                    f"{stale_dist_info}/licenses/LICENSE", recorded_paths
                )
                self.assertIn(
                    f"{actual_dist_info}/licenses/LICENSE", recorded_paths
                )

                for name, expected_hash, expected_size in rebuilt_rows:
                    if name == record_path:
                        self.assertEqual(expected_hash, "")
                        self.assertEqual(expected_size, "")
                        continue
                    data = archive.read(name)
                    self.assertEqual(wheel_hash(data), expected_hash)
                    self.assertEqual(str(len(data)), expected_size)


if __name__ == "__main__":
    unittest.main()
