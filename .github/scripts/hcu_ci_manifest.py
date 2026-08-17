#!/usr/bin/env python3
"""Create and verify HCU CI build artifact manifests."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FIELDS = {
    "schema_version",
    "repository",
    "source_sha",
    "base_sha",
    "build_environment",
    "image_reference",
    "image_digest",
    "tool_versions",
    "wheels",
}
REQUIRED_TOOL_VERSIONS = {"compiler", "cmake", "dtk", "python"}


class ManifestError(ValueError):
    """The build artifact metadata cannot be trusted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel:
        for block in iter(lambda: wheel.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(name: str, value: str) -> None:
    if not SHA_RE.fullmatch(value):
        raise ManifestError(f"{name} must be a 40-character lowercase git SHA")


def _require_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")


def _wheel_records(dist_dir: Path) -> List[Dict[str, Any]]:
    if not dist_dir.is_dir():
        raise ManifestError(f"wheel directory does not exist: {dist_dir}")

    wheels = []
    for wheel in sorted(dist_dir.glob("*.whl")):
        if not wheel.is_file():
            continue
        wheels.append(
            {
                "name": wheel.name,
                "size": wheel.stat().st_size,
                "sha256": _sha256(wheel),
            }
        )

    if not wheels:
        raise ManifestError(f"no wheel files found in {dist_dir}")
    return wheels


def _validate_tool_versions(tool_versions: Mapping[str, str]) -> Dict[str, str]:
    if not isinstance(tool_versions, Mapping):
        raise ManifestError("tool_versions must be an object")

    missing = sorted(REQUIRED_TOOL_VERSIONS - set(tool_versions))
    if missing:
        raise ManifestError(f"tool_versions is missing: {', '.join(missing)}")

    normalized: Dict[str, str] = {}
    for name, value in tool_versions.items():
        _require_nonempty_string(f"tool_versions.{name}", value)
        normalized[str(name)] = value.strip()
    return normalized


def build_manifest(
    *,
    repository: str,
    source_sha: str,
    base_sha: str,
    build_environment: str,
    image_reference: str,
    image_digest: str,
    dist_dir: Path,
    tool_versions: Mapping[str, str],
) -> Dict[str, Any]:
    """Build metadata that binds wheels to the source and toolchain."""

    for name, value in {
        "repository": repository,
        "build_environment": build_environment,
        "image_reference": image_reference,
        "image_digest": image_digest,
    }.items():
        _require_nonempty_string(name, value)
    _require_sha("source_sha", source_sha)
    _require_sha("base_sha", base_sha)

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "build_environment": build_environment,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "tool_versions": _validate_tool_versions(tool_versions),
        "wheels": _wheel_records(dist_dir),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"manifest is not valid JSON: {path}") from error

    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be an object")
    return manifest


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        raise ManifestError(f"manifest is missing required fields: {', '.join(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema_version: {manifest['schema_version']!r}"
        )

    for name in ("repository", "build_environment", "image_reference", "image_digest"):
        _require_nonempty_string(name, manifest[name])
    _require_sha("source_sha", manifest["source_sha"])
    _require_sha("base_sha", manifest["base_sha"])
    _validate_tool_versions(manifest["tool_versions"])

    if not isinstance(manifest["wheels"], list) or not manifest["wheels"]:
        raise ManifestError("wheels must be a non-empty list")
    for wheel in manifest["wheels"]:
        if not isinstance(wheel, dict):
            raise ManifestError("each wheel record must be an object")
        for field in ("name", "sha256", "size"):
            if field not in wheel:
                raise ManifestError(f"wheel record is missing {field}")
        if Path(wheel["name"]).name != wheel["name"]:
            raise ManifestError("wheel name must not contain a path")
        _require_nonempty_string("wheel name", wheel["name"])
        if not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]):
            raise ManifestError(f"wheel {wheel['name']} has an invalid sha256")
        if not isinstance(wheel["size"], int) or wheel["size"] < 0:
            raise ManifestError(f"wheel {wheel['name']} has an invalid size")


def validate_manifest(
    manifest_path: Path,
    dist_dir: Path,
    *,
    repository: str,
    source_sha: str,
    build_environment: str,
    image_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate metadata identity and every wheel stored beside it."""

    manifest = _read_manifest(manifest_path)
    _validate_manifest_shape(manifest)

    expected = {
        "repository": repository,
        "source_sha": source_sha,
        "build_environment": build_environment,
    }
    if image_digest is not None:
        expected["image_digest"] = image_digest
    for name, value in expected.items():
        if manifest[name] != value:
            raise ManifestError(
                f"manifest {name} mismatch: expected {value!r}, got {manifest[name]!r}"
            )

    expected_wheels = {wheel["name"]: wheel for wheel in manifest["wheels"]}
    actual_wheels = {wheel["name"]: wheel for wheel in _wheel_records(dist_dir)}
    if set(expected_wheels) != set(actual_wheels):
        raise ManifestError("wheel set does not match manifest")

    for name, expected_wheel in expected_wheels.items():
        actual_wheel = actual_wheels[name]
        if actual_wheel["size"] != expected_wheel["size"]:
            raise ManifestError(f"wheel {name} size mismatch")
        if actual_wheel["sha256"] != expected_wheel["sha256"]:
            raise ManifestError(f"wheel {name} sha256 mismatch")
    return manifest


def _parse_tool_versions(values: Sequence[str]) -> Dict[str, str]:
    tool_versions: Dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name or not version:
            raise ManifestError("--tool-version must use NAME=VERSION")
        tool_versions[name] = version
    return tool_versions


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--build-environment", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    write_parser = commands.add_parser("write")
    _add_identity_arguments(write_parser)
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--dist-dir", type=Path, required=True)
    write_parser.add_argument("--base-sha", required=True)
    write_parser.add_argument("--image-reference", required=True)
    write_parser.add_argument("--image-digest", required=True)
    write_parser.add_argument("--tool-version", action="append", default=[])

    validate_parser = commands.add_parser("validate")
    _add_identity_arguments(validate_parser)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--dist-dir", type=Path, required=True)
    validate_parser.add_argument("--image-digest")

    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    try:
        if args.command == "write":
            manifest = build_manifest(
                repository=args.repository,
                source_sha=args.source_sha,
                base_sha=args.base_sha,
                build_environment=args.build_environment,
                image_reference=args.image_reference,
                image_digest=args.image_digest,
                dist_dir=args.dist_dir,
                tool_versions=_parse_tool_versions(args.tool_version),
            )
            write_manifest(args.output, manifest)
        else:
            validate_manifest(
                args.manifest,
                args.dist_dir,
                repository=args.repository,
                source_sha=args.source_sha,
                build_environment=args.build_environment,
                image_digest=args.image_digest,
            )
    except ManifestError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
