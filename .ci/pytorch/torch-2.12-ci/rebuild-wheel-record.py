#!/usr/bin/env python3

import argparse
import base64
import copy
import csv
import hashlib
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import List, Set, Tuple


CHUNK_SIZE = 1024 * 1024


def validate_archive_path(name: str) -> None:
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or ".." in path.parts:
        raise RuntimeError(f"unsafe wheel archive path: {name!r}")


def record_hash(digest: bytes) -> str:
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def find_record(archive: zipfile.ZipFile) -> Tuple[str, zipfile.ZipInfo]:
    names: Set[str] = set()
    dist_info_dirs: Set[str] = set()

    for info in archive.infolist():
        validate_archive_path(info.filename)
        if info.filename in names:
            raise RuntimeError(f"duplicate wheel archive path: {info.filename}")
        names.add(info.filename)

        top_level = info.filename.split("/", 1)[0]
        if top_level.endswith(".dist-info"):
            dist_info_dirs.add(top_level)

    if len(dist_info_dirs) != 1:
        found = ", ".join(sorted(dist_info_dirs)) or "none"
        raise RuntimeError(f"expected one .dist-info directory, found: {found}")

    dist_info_dir = next(iter(dist_info_dirs))
    record_path = f"{dist_info_dir}/RECORD"
    record_infos = [
        info for info in archive.infolist() if info.filename == record_path
    ]
    if len(record_infos) != 1:
        raise RuntimeError(
            f"expected one {record_path}, found {len(record_infos)}"
        )
    return record_path, record_infos[0]


def build_record(
    archive: zipfile.ZipFile, record_path: str
) -> Tuple[bytes, Set[str]]:
    signature_paths = {f"{record_path}.jws", f"{record_path}.p7s"}
    rows: List[Tuple[str, str, str]] = []
    expected_paths: Set[str] = {record_path}

    for info in archive.infolist():
        name = info.filename
        if name == record_path or name in signature_paths or name.endswith("/"):
            continue

        digest = hashlib.sha256()
        size = 0
        with archive.open(info, "r") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)

        rows.append((name, record_hash(digest.digest()), str(size)))
        expected_paths.add(name)

    rows.sort(key=lambda row: row[0])
    rows.append((record_path, "", ""))
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    return record_buffer.getvalue().encode("utf-8"), expected_paths


def replace_record(
    wheel_path: Path,
    record_path: str,
    record_data: bytes,
    expected_paths: Set[str],
) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{wheel_path.name}.",
        suffix=".tmp.whl",
        dir=str(wheel_path.parent),
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)

    try:
        shutil.copy2(str(wheel_path), str(temp_path))
        with zipfile.ZipFile(temp_path, "a", allowZip64=True) as archive:
            current_record_path, current_record_info = find_record(archive)
            if current_record_path != record_path:
                raise RuntimeError("wheel RECORD path changed during rebuild")

            archive.filelist.remove(current_record_info)
            del archive.NameToInfo[record_path]
            new_record_info = copy.copy(current_record_info)
            archive.writestr(new_record_info, record_data)

        verify_record_paths(temp_path, record_path, expected_paths)
        os.replace(str(temp_path), str(wheel_path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def verify_record_paths(
    wheel_path: Path, record_path: str, expected_paths: Set[str]
) -> None:
    with zipfile.ZipFile(wheel_path, "r") as archive:
        current_record_path, _ = find_record(archive)
        if current_record_path != record_path:
            raise RuntimeError("rebuilt wheel RECORD path is inconsistent")

        record_rows = list(
            csv.reader(
                io.StringIO(archive.read(record_path).decode("utf-8"), newline="")
            )
        )
        recorded_paths = {row[0] for row in record_rows if len(row) == 3}
        if len(recorded_paths) != len(record_rows):
            raise RuntimeError(
                "rebuilt wheel RECORD contains invalid or duplicate rows"
            )
        if recorded_paths != expected_paths:
            missing = sorted(expected_paths - recorded_paths)
            unexpected = sorted(recorded_paths - expected_paths)
            raise RuntimeError(
                f"rebuilt RECORD path mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )


def rebuild_wheel_record(wheel_path: Path) -> None:
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise RuntimeError(f"wheel does not exist: {wheel_path}")

    with zipfile.ZipFile(wheel_path, "r") as archive:
        record_path, _ = find_record(archive)
        record_data, expected_paths = build_record(archive, record_path)

    replace_record(wheel_path, record_path, record_data, expected_paths)
    print(
        f"Rebuilt {record_path} with {len(expected_paths)} entries in "
        f"{wheel_path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild wheel RECORD after CIUpload changes dist-info metadata."
    )
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    rebuild_wheel_record(args.wheel.resolve())


if __name__ == "__main__":
    main()
