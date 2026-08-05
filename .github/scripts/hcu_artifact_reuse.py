#!/usr/bin/env python3
"""Validate that a runner-local HCU wheel artifact is safe to reuse."""
# ruff: noqa: UP006, UP035

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EXACT_PATHS = {
    ".ci/pytorch/hcu-test-baseline.json",
    ".ci/pytorch/test.sh",
    ".github/labeler.yml",
    ".github/workflows/_hcu-test.yml",
    ".github/workflows/hcu-stage4.yml",
}
ALLOWED_PATH_PREFIXES = (
    ".ci/docker/hcu-ci-test/",
    ".github/actions/setup-hcu/",
    ".github/scripts/hcu_",
    ".github/scripts/test_hcu_",
    "docs/",
    "test/",
)


class ReuseError(ValueError):
    """The requested artifact cannot safely be reused."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(name: str, value: str) -> None:
    if not SHA_RE.fullmatch(value):
        raise ReuseError(f"{name} must be a 40-character lowercase git SHA")


def read_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReuseError(f"missing artifact manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise ReuseError(f"invalid artifact manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReuseError("artifact manifest root must be an object")
    return value


def read_metadata(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ReuseError(f"missing artifact metadata: {path}") from error

    metadata = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ReuseError(f"invalid artifact metadata line: {line!r}")
        if key in metadata:
            raise ReuseError(f"duplicate artifact metadata key: {key}")
        metadata[key] = value
    return metadata


def require_artifact_path(root: Path, repository: str, requested: Path) -> Path:
    if not root.is_dir():
        raise ReuseError(f"artifact root does not exist: {root}")
    if not requested.is_dir():
        raise ReuseError(f"artifact path does not exist: {requested}")

    resolved_root = root.resolve()
    repository_root = (resolved_root / repository).resolve()
    artifact_path = requested.resolve()
    try:
        common = Path(os.path.commonpath([str(repository_root), str(artifact_path)]))
    except ValueError as error:
        raise ReuseError("artifact path and root are on different filesystems") from error
    if common != repository_root:
        raise ReuseError(
            f"artifact path is outside {repository_root}: {artifact_path}"
        )
    return artifact_path


def verify_archive(artifact_path: Path) -> None:
    archive = artifact_path / "artifacts.zip"
    checksum_file = artifact_path / "artifacts.zip.sha256"
    if not archive.is_file() or not checksum_file.is_file():
        raise ReuseError("artifact archive or checksum file is missing")

    fields = checksum_file.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != "artifacts.zip":
        raise ReuseError("artifacts.zip.sha256 has an unexpected format")
    expected = fields[0]
    if not SHA256_RE.fullmatch(expected):
        raise ReuseError("artifacts.zip.sha256 contains an invalid digest")
    actual = sha256(archive)
    if actual != expected:
        raise ReuseError(
            f"artifact archive SHA256 mismatch: expected {expected}, got {actual}"
        )


def changed_paths(repo_root: Path, source_sha: str, current_sha: str) -> List[str]:
    for name, value in (("source_sha", source_sha), ("current_sha", current_sha)):
        require_sha(name, value)
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{value}^{{commit}}"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode != 0:
            raise ReuseError(f"{name} is not available in the checkout: {value}")

    output = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            source_sha,
            current_sha,
            "--",
        ],
        cwd=str(repo_root),
        universal_newlines=True,
    )
    return [line for line in output.splitlines() if line]


def is_test_only_path(path: str) -> bool:
    if path in ALLOWED_EXACT_PATHS or path.endswith(".md"):
        return True
    return path.startswith(ALLOWED_PATH_PREFIXES)


def write_outputs(path: Path, values: Dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise ReuseError(f"output {name} contains a newline")
            stream.write(f"{name}={value}\n")


def validate(args: argparse.Namespace) -> None:
    require_sha("reuse_source_sha", args.reuse_source_sha)
    require_sha("current_sha", args.current_sha)
    artifact_path = require_artifact_path(
        args.artifact_root, args.repository, args.artifact_path
    )

    if not (artifact_path / "READY").is_file():
        raise ReuseError("artifact is not marked READY")
    if (artifact_path / "INCOMPLETE").exists():
        raise ReuseError("artifact is still marked INCOMPLETE")
    verify_archive(artifact_path)

    manifest = read_json(artifact_path / "hcu-build-manifest.json")
    metadata = read_metadata(artifact_path / "run-metadata.txt")
    expected = {
        "repository": args.repository,
        "source_sha": args.reuse_source_sha,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ReuseError(
                f"manifest {name} mismatch: expected {value!r}, got {manifest.get(name)!r}"
            )
        if metadata.get(name) != value:
            raise ReuseError(
                f"metadata {name} mismatch: expected {value!r}, got {metadata.get(name)!r}"
            )

    if metadata.get("runner_name") != args.runner_name:
        raise ReuseError(
            "runner-local artifact was produced on "
            f"{metadata.get('runner_name')!r}, current runner is {args.runner_name!r}"
        )
    for field in ("build_environment", "image_digest"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ReuseError(f"manifest {field} must be a non-empty string")

    paths = changed_paths(args.repo_root, args.reuse_source_sha, args.current_sha)
    disallowed = [path for path in paths if not is_test_only_path(path)]
    if disallowed:
        preview = "\n".join(f"  - {path}" for path in disallowed[:50])
        raise ReuseError(
            "PyTorch runtime/build-affecting paths changed; a fresh wheel is required:\n"
            + preview
        )

    outputs = {
        "artifact-path": str(artifact_path),
        "runner-name": args.runner_name,
        "build-environment": str(manifest["build_environment"]),
        "image-digest": str(manifest["image_digest"]),
        "source-sha": args.reuse_source_sha,
    }
    write_outputs(args.github_output, outputs)
    print(
        "HCU artifact reuse accepted: "
        f"source={args.reuse_source_sha}, current={args.current_sha}, "
        f"test-only changed paths={len(paths)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--reuse-source-sha", required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(args)
    except (OSError, ReuseError, subprocess.CalledProcessError) as error:
        print(f"HCU artifact reuse rejected: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
