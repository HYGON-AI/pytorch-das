# Owner(s): ["module: cuda"]

import os
import subprocess
import sys
import unittest

import torch

from torch.testing._internal.common_utils import TEST_CUDA, TestCase, run_tests


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestCudaPermuteCopyFastpath(TestCase):
    def _make_input(self, shape, dtype):
        numel = 1
        for size in shape:
            numel *= size
        base = torch.arange(numel, dtype=torch.float32).reshape(shape).fmod_(97)
        return base.to(dtype=dtype)

    def _dtypes(self):
        dtypes = [torch.float16, torch.float32]
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        return dtypes

    def _check_case(self, shape, perm, dtype):
        cpu_src = self._make_input(shape, dtype)
        gpu_src = cpu_src.cuda()
        ref = cpu_src.permute(perm).contiguous()

        self.assertEqual(gpu_src.permute(perm).contiguous().cpu(), ref)
        self.assertEqual(gpu_src.permute(perm).clone().cpu(), ref)

        dst = torch.empty(ref.shape, device="cuda", dtype=dtype)
        dst.copy_(gpu_src.permute(perm))
        self.assertEqual(dst.cpu(), ref)

    def _check_t_case(self, shape, dtype):
        cpu_src = self._make_input(shape, dtype)
        gpu_src = cpu_src.cuda()
        ref = cpu_src.t().contiguous()

        self.assertEqual(gpu_src.t().contiguous().cpu(), ref)
        self.assertEqual(gpu_src.t().clone().cpu(), ref)

        dst = torch.empty(ref.shape, device="cuda", dtype=dtype)
        dst.copy_(gpu_src.t())
        self.assertEqual(dst.cpu(), ref)

    def test_2d_transpose_contiguous_copy(self):
        shapes = [(8, 64), (64, 8), (64, 64)]
        for dtype in self._dtypes():
            for shape in shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    self._check_t_case(shape, dtype)

    def test_grouped_transpose_copy(self):
        cases = [
            ((64, 64, 3), (2, 0, 1)),
            ((16, 32, 8, 4), (2, 3, 0, 1)),
        ]
        for dtype in self._dtypes():
            for shape, perm in cases:
                with self.subTest(dtype=dtype, shape=shape, perm=perm):
                    self._check_case(shape, perm, dtype)

    def test_size_one_dims_copy(self):
        cases = [
            ((1, 3, 64, 64), (0, 2, 3, 1)),
            ((64, 64, 32, 1), (2, 0, 1, 3)),
        ]
        for dtype in self._dtypes():
            for shape, perm in cases:
                with self.subTest(dtype=dtype, shape=shape, perm=perm):
                    self._check_case(shape, perm, dtype)

    def test_alias_overlap_fallback(self):
        x = self._make_input((4, 4), torch.float16).cuda()
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            x.copy_(x.t())
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            x.t().copy_(x)


if __name__ == "__main__":
    run_tests()
