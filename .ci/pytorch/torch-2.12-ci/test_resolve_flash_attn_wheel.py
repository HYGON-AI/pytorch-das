#!/usr/bin/env python3

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("resolve_flash_attn_wheel.py")
SPEC = importlib.util.spec_from_file_location("resolve_flash_attn_wheel", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeOpener:
    def __init__(self, responses):
        self.responses = responses
        self.requested_urls = []

    def __call__(self, url):
        self.requested_urls.append(url)
        try:
            payload = self.responses[url]
        except KeyError as error:
            raise AssertionError(f"unexpected URL: {url}") from error
        return io.BytesIO(payload)


def wheel_link(filename, payload, digest=None):
    if digest is None:
        digest = hashlib.sha256(payload).hexdigest()
    return f'<a href="files/{filename}#sha256={digest}">{filename}</a>'


class ResolveFlashAttnWheelTest(unittest.TestCase):
    def test_selects_latest_matching_manylinux_wheel(self):
        package_url = "https://packages.example/simple/flash-attn/"
        selected_payload = b"selected torch 2.12 wheel"
        selected_filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608171654.ge8ddf7-"
            "cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
        )
        older_filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608161200.g111111-"
            "cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        newer_linux_filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608181200.g222222-"
            "cp310-cp310-linux_x86_64.whl"
        )
        wrong_torch_filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2100.2608191200.g333333-"
            "cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        wrong_abi_filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608191200.g444444-"
            "cp311-cp311-manylinux_2_28_x86_64.whl"
        )
        wrong_version_filename = (
            "flash_attn-2.8.4+das185.dtk2604.torch2120.2608191200.g555555-"
            "cp310-cp310-manylinux_2_28_x86_64.whl"
        )

        payloads = {
            selected_filename: selected_payload,
            older_filename: b"older manylinux wheel",
            newer_linux_filename: b"newer generic linux wheel",
            wrong_torch_filename: b"torch 2.10 wheel",
            wrong_abi_filename: b"cp311 wheel",
            wrong_version_filename: b"2.8.4 wheel",
        }
        html = "\n".join(
            wheel_link(filename, payload)
            for filename, payload in reversed(list(payloads.items()))
        ).encode("utf-8")
        responses = {package_url: html}
        for filename, payload in payloads.items():
            digest = hashlib.sha256(payload).hexdigest()
            responses[f"{package_url}files/{filename}"] = payload
        opener = FakeOpener(responses)

        with tempfile.TemporaryDirectory() as temp_dir:
            wheel_path = MODULE.resolve_and_download(
                index_url="https://packages.example/simple/",
                base_version="2.8.3",
                torch_tag="torch2120",
                destination=Path(temp_dir),
                python_tag="cp310",
                abi_tag="cp310",
                opener=opener,
            )

            self.assertEqual(wheel_path.name, selected_filename)
            self.assertEqual(wheel_path.read_bytes(), selected_payload)
            self.assertEqual(
                opener.requested_urls,
                [package_url, f"{package_url}files/{selected_filename}"],
            )

    def test_rejects_a_download_with_the_wrong_sha256(self):
        package_url = "https://packages.example/simple/flash-attn/"
        filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608171654.ge8ddf7-"
            "cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        html = wheel_link(filename, b"expected payload").encode("utf-8")
        opener = FakeOpener(
            {
                package_url: html,
                f"{package_url}files/{filename}": b"tampered payload",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                MODULE.FlashAttnWheelError, "sha256 mismatch"
            ):
                MODULE.resolve_and_download(
                    index_url="https://packages.example/simple/",
                    base_version="2.8.3",
                    torch_tag="torch2120",
                    destination=Path(temp_dir),
                    python_tag="cp310",
                    abi_tag="cp310",
                    opener=opener,
                )
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_requires_a_sha256_fragment(self):
        package_url = "https://packages.example/simple/flash-attn/"
        filename = (
            "flash_attn-2.8.3+das185.dtk2604.torch2120.2608171654.ge8ddf7-"
            "cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        html = f'<a href="files/{filename}">{filename}</a>'.encode("utf-8")
        opener = FakeOpener({package_url: html})

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                MODULE.FlashAttnWheelError, "valid sha256 fragment"
            ):
                MODULE.resolve_and_download(
                    index_url="https://packages.example/simple/",
                    base_version="2.8.3",
                    torch_tag="torch2120",
                    destination=Path(temp_dir),
                    python_tag="cp310",
                    abi_tag="cp310",
                    opener=opener,
                )

    def test_cli_prints_only_the_wheel_path_to_stdout(self):
        expected_path = Path("dist/flash/selected.whl")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            MODULE, "resolve_and_download", return_value=expected_path
        ) as resolve, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            return_code = MODULE.main(
                [
                    "--index-url",
                    "https://packages.example/simple/",
                    "--version",
                    "2.8.3",
                    "--torch-tag",
                    "torch2120",
                    "--dest",
                    "dist/flash",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(stdout.getvalue(), f"{expected_path}\n")
        self.assertIn("Selected Flash Attention wheel", stderr.getvalue())
        resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
