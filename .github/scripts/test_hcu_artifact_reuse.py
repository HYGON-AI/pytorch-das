#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from hcu_artifact_reuse import ReuseError, validate


REPOSITORY = "HYGON-AI/pytorch-das"
RUNNER_NAME = "nmz36-hygon-hcu-torch"


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git"] + list(arguments),
        cwd=str(repo),
        universal_newlines=True,
    ).strip()


class HcuArtifactReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        git(self.repo, "init")
        git(self.repo, "config", "user.name", "HCU CI Test")
        git(self.repo, "config", "user.email", "hcu-ci-test@example.invalid")

        runtime = self.repo / "torch" / "runtime.py"
        runtime.parent.mkdir()
        runtime.write_text("runtime = 1\n", encoding="utf-8")
        test_file = self.repo / "test" / "test_hcu.py"
        test_file.parent.mkdir()
        test_file.write_text("def test_hcu(): pass\n", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-m", "base")
        self.source_sha = git(self.repo, "rev-parse", "HEAD")

        self.artifact_root = self.root / "artifacts"
        self.artifact_path = (
            self.artifact_root / REPOSITORY / "123" / "1" / "hcu-build-123-1"
        )
        self.artifact_path.mkdir(parents=True)
        archive = self.artifact_path / "artifacts.zip"
        archive.write_bytes(b"verified wheel archive")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.artifact_path / "artifacts.zip.sha256").write_text(
            f"{digest}  artifacts.zip\n", encoding="utf-8"
        )
        (self.artifact_path / "hcu-build-manifest.json").write_text(
            json.dumps(
                {
                    "repository": REPOSITORY,
                    "source_sha": self.source_sha,
                    "build_environment": "linux-rocky8-rocm-dtk2604-hcu-py3.10",
                    "image_digest": "sha256:" + "1" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.artifact_path / "run-metadata.txt").write_text(
            "\n".join(
                (
                    f"repository={REPOSITORY}",
                    f"source_sha={self.source_sha}",
                    "run_id=123",
                    "run_attempt=1",
                    f"runner_name={RUNNER_NAME}",
                    "artifact_name=hcu-build-123-1",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (self.artifact_path / "READY").touch()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def commit(self, path: str, content: str) -> str:
        destination = self.repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        git(self.repo, "add", path)
        git(self.repo, "commit", "-m", f"change {path}")
        return git(self.repo, "rev-parse", "HEAD")

    def arguments(self, current_sha: str) -> argparse.Namespace:
        return argparse.Namespace(
            artifact_root=self.artifact_root,
            artifact_path=self.artifact_path,
            repository=REPOSITORY,
            reuse_source_sha=self.source_sha,
            current_sha=current_sha,
            runner_name=RUNNER_NAME,
            repo_root=self.repo,
            github_output=self.root / "github-output.txt",
        )

    def test_accepts_test_only_change(self) -> None:
        current_sha = self.commit(
            "test/test_hcu.py",
            "def test_hcu():\n    assert True\n",
        )

        validate(self.arguments(current_sha))

        outputs = dict(
            line.split("=", 1)
            for line in (self.root / "github-output.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(outputs["source-sha"], self.source_sha)
        self.assertEqual(outputs["runner-name"], RUNNER_NAME)
        self.assertEqual(outputs["artifact-path"], str(self.artifact_path))

    def test_rejects_runtime_change(self) -> None:
        current_sha = self.commit("torch/runtime.py", "runtime = 2\n")

        with self.assertRaisesRegex(ReuseError, "fresh wheel is required"):
            validate(self.arguments(current_sha))

    def test_rejects_archive_digest_mismatch(self) -> None:
        current_sha = self.commit(
            "test/test_hcu.py",
            "def test_hcu():\n    assert True\n",
        )
        (self.artifact_path / "artifacts.zip").write_bytes(b"tampered")

        with self.assertRaisesRegex(ReuseError, "SHA256 mismatch"):
            validate(self.arguments(current_sha))

    def test_rejects_different_runner(self) -> None:
        current_sha = self.commit(
            "test/test_hcu.py",
            "def test_hcu():\n    assert True\n",
        )
        arguments = self.arguments(current_sha)
        arguments.runner_name = "another-runner"

        with self.assertRaisesRegex(ReuseError, "current runner"):
            validate(arguments)


if __name__ == "__main__":
    unittest.main()
