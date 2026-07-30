#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hcu_ci_manifest import (  # noqa: E402
    ManifestError,
    build_manifest,
    validate_manifest,
    write_manifest,
)


class HcuCiManifestTest(unittest.TestCase):
    SOURCE_SHA = "a" * 40
    BASE_SHA = "b" * 40

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dist_dir = self.root / "dist"
        self.dist_dir.mkdir()
        (self.dist_dir / "torch-2.12.0-cp310-cp310-linux_x86_64.whl").write_bytes(
            b"hcu-wheel"
        )
        self.manifest_path = self.root / "hcu-build-manifest.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_and_write_manifest(self) -> None:
        manifest = build_manifest(
            repository="HYGON-AI/pytorch-das",
            source_sha=self.SOURCE_SHA,
            base_sha=self.BASE_SHA,
            build_environment="linux-jammy-rocm-hcu-py3.10",
            image_reference="registry.example/hcu-ci:20260723",
            image_digest="sha256:" + "c" * 64,
            dist_dir=self.dist_dir,
            tool_versions={
                "compiler": "clang 18",
                "cmake": "3.30",
                "dtk": "7.0",
                "python": "3.10",
            },
        )
        write_manifest(self.manifest_path, manifest)

    def test_valid_manifest_matches_artifact_and_expected_identity(self) -> None:
        self.build_and_write_manifest()

        manifest = validate_manifest(
            self.manifest_path,
            self.dist_dir,
            repository="HYGON-AI/pytorch-das",
            source_sha=self.SOURCE_SHA,
            build_environment="linux-jammy-rocm-hcu-py3.10",
            image_digest="sha256:" + "c" * 64,
        )

        self.assertEqual(manifest["source_sha"], self.SOURCE_SHA)
        self.assertEqual(len(manifest["wheels"]), 1)
        self.assertEqual(manifest["wheels"][0]["name"], "torch-2.12.0-cp310-cp310-linux_x86_64.whl")

    def test_source_sha_mismatch_is_rejected(self) -> None:
        self.build_and_write_manifest()

        with self.assertRaisesRegex(ManifestError, "source_sha"):
            validate_manifest(
                self.manifest_path,
                self.dist_dir,
                repository="HYGON-AI/pytorch-das",
                source_sha="d" * 40,
                build_environment="linux-jammy-rocm-hcu-py3.10",
            )

    def test_changed_wheel_is_rejected(self) -> None:
        self.build_and_write_manifest()
        (self.dist_dir / "torch-2.12.0-cp310-cp310-linux_x86_64.whl").write_bytes(
            b"hcu-xheel"
        )

        with self.assertRaisesRegex(ManifestError, "sha256"):
            validate_manifest(
                self.manifest_path,
                self.dist_dir,
                repository="HYGON-AI/pytorch-das",
                source_sha=self.SOURCE_SHA,
                build_environment="linux-jammy-rocm-hcu-py3.10",
            )

    def test_extra_wheel_is_rejected(self) -> None:
        self.build_and_write_manifest()
        (self.dist_dir / "torch-extra.whl").write_bytes(b"unexpected-wheel")

        with self.assertRaisesRegex(ManifestError, "wheel set"):
            validate_manifest(
                self.manifest_path,
                self.dist_dir,
                repository="HYGON-AI/pytorch-das",
                source_sha=self.SOURCE_SHA,
                build_environment="linux-jammy-rocm-hcu-py3.10",
            )

    def test_image_digest_mismatch_is_rejected(self) -> None:
        self.build_and_write_manifest()

        with self.assertRaisesRegex(ManifestError, "image_digest"):
            validate_manifest(
                self.manifest_path,
                self.dist_dir,
                repository="HYGON-AI/pytorch-das",
                source_sha=self.SOURCE_SHA,
                build_environment="linux-jammy-rocm-hcu-py3.10",
                image_digest="sha256:" + "d" * 64,
            )

    def test_missing_required_field_is_rejected(self) -> None:
        self.build_and_write_manifest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del manifest["tool_versions"]
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ManifestError, "tool_versions"):
            validate_manifest(
                self.manifest_path,
                self.dist_dir,
                repository="HYGON-AI/pytorch-das",
                source_sha=self.SOURCE_SHA,
                build_environment="linux-jammy-rocm-hcu-py3.10",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
