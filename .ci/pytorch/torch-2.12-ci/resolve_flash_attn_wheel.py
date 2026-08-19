#!/usr/bin/env python3
"""Resolve a Torch-specific Flash Attention wheel from a PEP 503 index."""

import argparse
import hashlib
import os
import re
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence
from urllib.parse import (
    parse_qs,
    unquote,
    urldefrag,
    urljoin,
    urlsplit,
)
from urllib.request import urlopen


PACKAGE_NAME = "flash-attn"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class FlashAttnWheelError(RuntimeError):
    """Raised when a compatible, verifiable wheel cannot be downloaded."""


class WheelCandidate(NamedTuple):
    filename: str
    version: str
    url: str
    platform_tag: str


class _SimpleIndexParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.links = []  # type: List[str]

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href and "data-yanked" not in attributes:
            self.links.append(href)


def _normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_index_url(index_url):
    root = index_url.rstrip("/") + "/"
    path_parts = [part for part in urlsplit(root).path.split("/") if part]
    if path_parts and _normalize_package_name(path_parts[-1]) == PACKAGE_NAME:
        return root
    return urljoin(root, PACKAGE_NAME + "/")


def _current_cpython_tag():
    if sys.implementation.name != "cpython":
        raise FlashAttnWheelError(
            "Flash Attention test wheels require CPython, but this interpreter is "
            + sys.implementation.name
        )
    return "cp{}{}".format(sys.version_info.major, sys.version_info.minor)


def _parse_wheel(href, page_url):
    absolute_url = urljoin(page_url, href)
    filename = unquote(Path(urlsplit(absolute_url).path).name)
    if not filename.endswith(".whl") or Path(filename).name != filename:
        return None

    wheel_parts = filename[:-4].rsplit("-", 3)
    if len(wheel_parts) != 4:
        return None
    distribution_and_version, python_tags, abi_tags, platform_tags = wheel_parts
    if "-" not in distribution_and_version:
        return None
    distribution, version = distribution_and_version.split("-", 1)
    if _normalize_package_name(distribution) != PACKAGE_NAME:
        return None

    return (
        WheelCandidate(filename, version, absolute_url, platform_tags),
        python_tags.split("."),
        abi_tags.split("."),
    )


def _has_exact_torch_tag(version, torch_tag):
    _, separator, local_version = version.partition("+")
    if not separator:
        return False
    local_segments = re.split(r"[._-]+", local_version.lower())
    return torch_tag.lower() in local_segments


def _natural_version_key(version):
    parts = re.findall(r"\d+|[^\d]+", version.lower())
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts)


def _platform_priority(platform_tag):
    lowered = platform_tag.lower()
    if "manylinux" in lowered:
        return 2
    if "linux" in lowered:
        return 1
    return 0


def _read_url(url, opener):
    try:
        with opener(url) as response:
            return response.read()
    except Exception as error:
        raise FlashAttnWheelError("failed to read {}: {}".format(url, error)) from error


def _select_candidate(
    package_url,
    html,
    base_version,
    torch_tag,
    python_tag,
    abi_tag,
):
    parser = _SimpleIndexParser()
    parser.feed(html.decode("utf-8", errors="replace"))

    candidates = []
    for href in parser.links:
        parsed = _parse_wheel(href, package_url)
        if parsed is None:
            continue
        candidate, python_tags, abi_tags = parsed
        public_version = candidate.version.partition("+")[0]
        if public_version != base_version:
            continue
        if not _has_exact_torch_tag(candidate.version, torch_tag):
            continue
        if python_tag not in python_tags or abi_tag not in abi_tags:
            continue
        candidates.append(candidate)

    if not candidates:
        raise FlashAttnWheelError(
            "no {} {} wheel matched torch tag {} and ABI {}/{} at {}".format(
                PACKAGE_NAME,
                base_version,
                torch_tag,
                python_tag,
                abi_tag,
                package_url,
            )
        )

    return max(
        candidates,
        key=lambda candidate: (
            _platform_priority(candidate.platform_tag),
            _natural_version_key(candidate.version),
            candidate.filename,
        ),
    )


def _expected_sha256(url):
    fragment = urlsplit(url).fragment
    hashes = parse_qs(fragment, keep_blank_values=True).get("sha256", [])
    if len(hashes) != 1 or not _SHA256_PATTERN.fullmatch(hashes[0]):
        raise FlashAttnWheelError(
            "selected wheel URL does not contain exactly one valid sha256 fragment: "
            + url
        )
    return hashes[0].lower()


def _download_verified(candidate, destination, opener):
    expected_digest = _expected_sha256(candidate.url)
    download_url, _ = urldefrag(candidate.url)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / candidate.filename
    temporary_path = None  # type: Optional[Path]

    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=candidate.filename + ".",
            suffix=".part",
            dir=str(destination),
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            try:
                with opener(download_url) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        temporary_file.write(chunk)
            except Exception as error:
                raise FlashAttnWheelError(
                    "failed to download {}: {}".format(download_url, error)
                ) from error

        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise FlashAttnWheelError(
                "sha256 mismatch for {}: expected {}, got {}".format(
                    candidate.filename, expected_digest, actual_digest
                )
            )
        os.replace(str(temporary_path), str(output_path))
        temporary_path = None
        return output_path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def resolve_and_download(
    index_url,
    base_version,
    torch_tag,
    destination,
    python_tag=None,
    abi_tag=None,
    opener=urlopen,
):
    """Select and download the newest compatible Flash Attention wheel."""
    if not index_url:
        raise FlashAttnWheelError("index URL must not be empty")
    if not base_version:
        raise FlashAttnWheelError("base version must not be empty")
    if not re.fullmatch(r"torch\d+", torch_tag.lower()):
        raise FlashAttnWheelError("invalid torch tag: " + torch_tag)

    effective_python_tag = python_tag or _current_cpython_tag()
    effective_abi_tag = abi_tag or effective_python_tag
    package_url = _package_index_url(index_url)
    html = _read_url(package_url, opener)
    candidate = _select_candidate(
        package_url,
        html,
        base_version,
        torch_tag,
        effective_python_tag,
        effective_abi_tag,
    )
    return _download_verified(candidate, Path(destination), opener)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest Flash Attention wheel matching this CPython ABI "
            "and an exact Torch build tag"
        )
    )
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--version", required=True, dest="base_version")
    parser.add_argument("--torch-tag", required=True)
    parser.add_argument("--dest", required=True, type=Path, dest="destination")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        wheel_path = resolve_and_download(
            index_url=args.index_url,
            base_version=args.base_version,
            torch_tag=args.torch_tag,
            destination=args.destination,
        )
    except FlashAttnWheelError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1

    print("Selected Flash Attention wheel: {}".format(wheel_path), file=sys.stderr)
    print(wheel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
