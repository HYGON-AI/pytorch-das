# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import run_tests, TestCase


_CREATION_RE = re.compile(r"CUDA stream pipe binding: device=(\d+), pipe=(\d+)")
_REUSE_WARNING = "CUDA stream count exceeded the configured pipe count"
_WORKER_TIMEOUT_SECONDS = 120
_MINIMUM_DTK_VERSION = (26, 10)
_MINIMUM_DRIVER_VERSION = (6, 4, 1)


def _set_user_assign_pipe(enabled: bool) -> None:
    command = ["hy-smi", "--setuserassignpipe", "1" if enabled else "0"]
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        stdout = getattr(error, "stdout", "")
        stderr = getattr(error, "stderr", "")
        raise RuntimeError(
            f"Failed to run {' '.join(command)}"
            f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error


def _get_dtk_version() -> tuple[int, int] | None:
    version = str(getattr(torch.version, "dtk", "") or "")
    match = re.search(r"(\d+)\.(\d+)", version)
    if match:
        return int(match.group(1)), int(match.group(2))

    hcu_version = str(getattr(torch.version, "__hcu_version__", "") or "")
    match = re.search(r"dtk(\d{2})(\d{2})", hcu_version.lower())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _get_driver_version() -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            ["hy-smi", "--showdriverversion"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    match = re.search(
        r"Driver Version:\s*(\d+)\.(\d+)\.(\d+)",
        result.stdout + result.stderr,
    )
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


_DTK_VERSION = _get_dtk_version()
_IS_SUPPORTED_DTK_BUILD = (
    _DTK_VERSION is not None and _DTK_VERSION >= _MINIMUM_DTK_VERSION
)
_DRIVER_VERSION = (
    _get_driver_version()
    if "--worker" not in sys.argv
    and _IS_SUPPORTED_DTK_BUILD
    and torch.version.hip is not None
    else None
)
_IS_SUPPORTED_DRIVER = (
    _DRIVER_VERSION is not None
    and _DRIVER_VERSION >= _MINIMUM_DRIVER_VERSION
)


def _run_nccl_worker(groups: int, devices: int) -> None:
    process_groups = []
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        init_method = f"file://{rendezvous.name}"

    try:
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, init_method=init_method
        )
        process_groups = [dist.new_group(ranks=[0]) for _ in range(groups)]
        for device_index in range(devices):
            torch.cuda.set_device(device_index)
            tensor = torch.ones(1, device=device_index)
            for group in process_groups:
                work = dist.all_reduce(tensor, group=group, async_op=True)
                work.wait()
                torch.cuda.synchronize(device_index)
    finally:
        for group in process_groups:
            dist.destroy_process_group(group)
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(rendezvous.name)
        except FileNotFoundError:
            pass


def _run_two_rank_nccl_process(
    rank: int,
    init_method: str,
    group_count: int,
    threads_per_rank: int,
) -> None:
    process_groups = []
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=2,
            init_method=init_method,
        )
        process_groups = [dist.new_group(ranks=[0, 1]) for _ in range(group_count)]

        def run_collective(group_index: int) -> None:
            torch.cuda.set_device(rank)
            tensor = torch.ones(1, device=rank)
            work = dist.all_reduce(
                tensor, group=process_groups[group_index], async_op=True
            )
            work.wait()
            torch.cuda.synchronize(rank)
            if tensor.item() != 2:
                raise AssertionError(
                    f"rank {rank}, group {group_index}: expected 2, got {tensor.item()}"
                )

        if threads_per_rank == 1:
            for group_index in range(group_count):
                run_collective(group_index)
        else:
            if threads_per_rank != group_count:
                raise AssertionError(
                    "threaded NCCL worker requires one group per thread"
                )
            barrier = Barrier(threads_per_rank)

            def run_collective_after_barrier(group_index: int) -> None:
                barrier.wait()
                run_collective(group_index)

            with ThreadPoolExecutor(max_workers=threads_per_rank) as executor:
                futures = [
                    executor.submit(run_collective_after_barrier, group_index)
                    for group_index in range(group_count)
                ]
                for future in futures:
                    future.result()
    finally:
        for group in process_groups:
            dist.destroy_process_group(group)
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_rank_nccl_worker(group_count: int, threads_per_rank: int) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        rendezvous_path = rendezvous.name

    try:
        mp.spawn(
            _run_two_rank_nccl_process,
            args=(f"file://{rendezvous_path}", group_count, threads_per_rank),
            nprocs=2,
            join=True,
        )
    finally:
        try:
            os.unlink(rendezvous_path)
        except FileNotFoundError:
            pass


def _run_stream_worker(stream_count: int, devices: int) -> None:
    streams = []
    for device_index in range(devices):
        torch.cuda.set_device(device_index)
        for _ in range(stream_count):
            stream = torch.cuda.Stream(device=device_index)
            value = torch.zeros(1, device=device_index)
            with torch.cuda.stream(stream):
                value.add_(1)
            stream.synchronize()
            if value.item() != 1:
                raise AssertionError(
                    f"device {device_index}: expected 1, got {value.item()}"
                )
            streams.append(stream)


def _run_threaded_worker(threads_per_device: int, devices: int) -> None:
    worker_count = threads_per_device * devices
    barrier = Barrier(worker_count)

    def create_stream(device_index: int) -> torch.cuda.Stream:
        torch.cuda.set_device(device_index)
        barrier.wait()
        stream = torch.cuda.Stream(device=device_index)
        value = torch.zeros(1, device=device_index)
        with torch.cuda.stream(stream):
            value.add_(1)
        stream.synchronize()
        if value.item() != 1:
            raise AssertionError(
                f"device {device_index}: expected 1, got {value.item()}"
            )
        return stream

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(create_stream, device_index)
            for device_index in range(devices)
            for _ in range(threads_per_device)
        ]
        streams = [future.result() for future in futures]

    # Keep all streams alive until every worker has completed.
    assert len(streams) == worker_count


@unittest.skipUnless(
    _IS_SUPPORTED_DTK_BUILD
    and _IS_SUPPORTED_DRIVER
    and torch.version.hip is not None
    and torch.cuda.is_available()
    and dist.is_nccl_available(),
    "CUDA stream pipe binding requires DTK 26.10, driver 6.4.1 or newer, "
    "and RCCL support",
)
class CUDAStreamBindPipeTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.addClassCleanup(_set_user_assign_pipe, False)
        _set_user_assign_pipe(True)

    def _invoke_worker(
        self,
        *,
        bind_pipe: bool,
        pipe_list: str | None,
        stream_count: int,
        devices: int = 1,
        threads_per_device: int = 0,
        use_nccl: bool = False,
        distributed_threads_per_rank: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["TORCH_STREAM_BIND_PIPE_FLAG"] = "1" if bind_pipe else "0"
        env["HSA_ENABLE_USER_PIPE"] = "1" if bind_pipe else "0"
        env['HIP_ENABLE_PIPE_SUPPORT'] = "1" if bind_pipe else "0"
        env["TORCH_CPP_LOG_LEVEL"] = "INFO"
        if pipe_list is None:
            env.pop("TORCH_STREAM_BIND_PIPE", None)
        else:
            env["TORCH_STREAM_BIND_PIPE"] = pipe_list

        return subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--stream-count",
                str(stream_count),
                "--devices",
                str(devices),
                "--threads-per-device",
                str(threads_per_device),
                "--distributed-threads-per-rank",
                str(distributed_threads_per_rank),
                "--use-nccl" if use_nccl else "--no-use-nccl",
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=_WORKER_TIMEOUT_SECONDS,
        )

    def _assert_worker_succeeded(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def _created_pipes_by_device(
        self, result: subprocess.CompletedProcess[str]
    ) -> dict[int, list[int]]:
        pipes: dict[int, list[int]] = {}
        for device, pipe in _CREATION_RE.findall(result.stdout + result.stderr):
            pipes.setdefault(int(device), []).append(int(pipe))
        return pipes

    def test_disabled_uses_regular_stream_pool(self) -> None:
        result = self._invoke_worker(
            bind_pipe=False,
            pipe_list="not-a-valid-list",
            stream_count=1,
        )

        self._assert_worker_succeeded(result)
        self.assertIsNone(_CREATION_RE.search(result.stdout + result.stderr))

    def test_default_pipe_order(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list=None,
            stream_count=4,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(
            self._created_pipes_by_device(result), {0: [0, 1, 2, 3]}
        )
        self.assertNotIn(_REUSE_WARNING, result.stderr)

    def test_pipes_are_reused(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list=None,
            stream_count=5,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(
            self._created_pipes_by_device(result), {0: [0, 1, 2, 3, 0]}
        )
        self.assertIn(_REUSE_WARNING, result.stdout + result.stderr)

    def test_configured_pipes_are_reused(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,1,2",
            stream_count=4,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(
            self._created_pipes_by_device(result), {0: [0, 1, 2, 0]}
        )
        self.assertIn(_REUSE_WARNING, result.stdout + result.stderr)

    def test_invalid_pipe_is_rejected(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,4",
            stream_count=1,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid pipe ID '4'", result.stderr)

    def test_pipe_assignment_is_thread_safe(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,1,2,3",
            stream_count=0,
            threads_per_device=8,
        )

        self._assert_worker_succeeded(result)
        bindings = self._created_pipes_by_device(result)
        self.assertEqual(
            sorted(bindings.get(0, [])),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        self.assertEqual(
            (result.stdout + result.stderr).count(_REUSE_WARNING), 1
        )

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "per-device pipe assignment requires at least two accelerators",
    )
    def test_pipe_assignment_is_per_device(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,1",
            stream_count=2,
            devices=2,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(
            self._created_pipes_by_device(result),
            {0: [0, 1], 1: [0, 1]},
        )
        self.assertNotIn(_REUSE_WARNING, result.stderr)

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "multi-device multi-thread assignment requires at least two accelerators",
    )
    def test_pipe_assignment_is_per_device_across_threads(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="1,3",
            stream_count=0,
            devices=2,
            threads_per_device=4,
        )

        self._assert_worker_succeeded(result)
        bindings = self._created_pipes_by_device(result)
        self.assertEqual(sorted(bindings.get(0, [])), [1, 1, 3, 3])
        self.assertEqual(sorted(bindings.get(1, [])), [1, 1, 3, 3])
        self.assertEqual(
            (result.stdout + result.stderr).count(_REUSE_WARNING), 2
        )

    def test_nccl_collective_uses_common_stream_pool(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,1",
            stream_count=1,
            use_nccl=True,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(self._created_pipes_by_device(result), {0: [0]})

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "two-rank NCCL communication requires at least two accelerators",
    )
    def test_two_device_nccl_single_thread(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="0,1",
            stream_count=2,
            devices=2,
            use_nccl=True,
            distributed_threads_per_rank=1,
        )

        self._assert_worker_succeeded(result)
        self.assertEqual(
            self._created_pipes_by_device(result),
            {0: [0, 1], 1: [0, 1]},
        )
        self.assertNotIn(_REUSE_WARNING, result.stdout + result.stderr)

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "two-rank threaded NCCL communication requires at least two accelerators",
    )
    def test_two_device_nccl_two_threads(self) -> None:
        result = self._invoke_worker(
            bind_pipe=True,
            pipe_list="1,3",
            stream_count=2,
            devices=2,
            use_nccl=True,
            distributed_threads_per_rank=2,
        )

        self._assert_worker_succeeded(result)
        bindings = self._created_pipes_by_device(result)
        self.assertEqual(sorted(bindings.get(0, [])), [1, 3])
        self.assertEqual(sorted(bindings.get(1, [])), [1, 3])
        self.assertNotIn(_REUSE_WARNING, result.stdout + result.stderr)


def _main() -> None:
    if "--worker" not in sys.argv:
        run_tests()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stream-count", type=int, required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--threads-per-device", type=int, default=0)
    parser.add_argument("--distributed-threads-per-rank", type=int, default=0)
    parser.add_argument(
        "--use-nccl", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.distributed_threads_per_rank:
        _run_two_rank_nccl_worker(
            args.stream_count, args.distributed_threads_per_rank
        )
    elif args.threads_per_device:
        _run_threaded_worker(args.threads_per_device, args.devices)
    elif args.use_nccl:
        _run_nccl_worker(args.stream_count, args.devices)
    else:
        _run_stream_worker(args.stream_count, args.devices)


if __name__ == "__main__":
    _main()
