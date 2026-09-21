# Owner(s): ["module: sdpa"]

import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class TestFlashAttentionMasked(unittest.TestCase):
    """Regression coverage for the ROCm Cutlass masked FlashAttention APIs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU is required")

        cutlass = torch.backends.cuda._ROCmFABackends["cutlass"]
        if torch.backends.cuda.preferred_rocm_fa_library() != cutlass:
            raise unittest.SkipTest("ROCm Cutlass FlashAttention backend is required")

        required_ops = (
            "_scaled_dot_product_flash_attention_masked",
            "_scaled_dot_product_flash_attention_varlen_masked",
        )
        if any(not hasattr(torch.ops.aten, name) for name in required_ops):
            raise unittest.SkipTest(
                "masked FlashAttention ATen operators are unavailable"
            )

        # The ATen operators are built into PyTorch, while the mask entry points
        # are optional symbols in the external Cutlass FlashAttention library.
        # Probe both ABIs once so unrelated ROCm installations skip this suite.
        q = torch.zeros((1, 1, 1, 64), device="cuda", dtype=torch.bfloat16)
        mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        try:
            torch.ops.aten._scaled_dot_product_flash_attention_masked(
                q, q, q, mask, 0.0, False
            )
            torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                q.view(1, 1, 64),
                q.view(1, 1, 64),
                q.view(1, 1, 64),
                mask,
                cu_seqlens,
                cu_seqlens,
                1,
                1,
                0.0,
                False,
            )
        except RuntimeError as error:
            if "Cutlass FA" in str(error) and "unavailable" in str(error):
                raise unittest.SkipTest(str(error)) from error
            raise

        from flash_attn.version import version as flash_attn_version

        cls.supports_mask_dropout = tuple(
            int(part) for part in flash_attn_version.split(".")[:3]
        ) >= (2, 8, 4)

    @staticmethod
    def _tolerances(dtype):
        if dtype is torch.bfloat16:
            return {"atol": 3e-2, "rtol": 3e-2}
        return {"atol": 5e-3, "rtol": 5e-3}

    def test_dense_forward_backward_matches_math(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(100)
                batch, heads, q_len, kv_len, head_dim = 2, 2, 7, 9, 64
                query = torch.randn(
                    batch,
                    heads,
                    q_len,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                key = torch.randn(
                    batch,
                    heads,
                    kv_len,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                value = torch.randn_like(key, requires_grad=True)
                mask = torch.rand(
                    batch, heads, q_len, kv_len, device="cuda"
                ) > 0.25
                mask[..., 0] = True

                query_ref = query.detach().clone().requires_grad_()
                key_ref = key.detach().clone().requires_grad_()
                value_ref = value.detach().clone().requires_grad_()

                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    output = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=mask, dropout_p=0.0
                    )
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    output_ref = F.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        attn_mask=mask,
                        dropout_p=0.0,
                    )

                grad_output = torch.randn_like(output)
                grads = torch.autograd.grad(
                    output, (query, key, value), grad_output
                )
                grads_ref = torch.autograd.grad(
                    output_ref, (query_ref, key_ref, value_ref), grad_output
                )

                tolerances = self._tolerances(dtype)
                torch.testing.assert_close(output, output_ref, **tolerances)
                for grad, grad_ref in zip(grads, grads_ref):
                    torch.testing.assert_close(grad, grad_ref, **tolerances)

    def test_varlen_forward_backward_matches_math(self):
        q_lengths = (3, 5)
        kv_lengths = (4, 5)
        max_q = max(q_lengths)
        max_k = max(kv_lengths)
        heads = 2
        head_dim = 64

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(101)
                query = torch.randn(
                    sum(q_lengths),
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                key = torch.randn(
                    sum(kv_lengths),
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                value = torch.randn_like(key, requires_grad=True)
                mask = torch.rand(
                    len(q_lengths), heads, max_q, max_k, device="cuda"
                ) > 0.25
                mask[..., 0] = True
                cu_q = torch.tensor(
                    [0, q_lengths[0], sum(q_lengths)],
                    device="cuda",
                    dtype=torch.int32,
                )
                cu_k = torch.tensor(
                    [0, kv_lengths[0], sum(kv_lengths)],
                    device="cuda",
                    dtype=torch.int32,
                )

                query_ref = query.detach().clone().requires_grad_()
                key_ref = key.detach().clone().requires_grad_()
                value_ref = value.detach().clone().requires_grad_()

                output, logsumexp, rng_state = (
                    torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                        query,
                        key,
                        value,
                        mask,
                        cu_q,
                        cu_k,
                        max_q,
                        max_k,
                        0.0,
                        False,
                    )
                )

                output_ref_parts = []
                q_start = 0
                kv_start = 0
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    for batch_index, (q_len, kv_len) in enumerate(
                        zip(q_lengths, kv_lengths)
                    ):
                        q_slice = (
                            query_ref[q_start : q_start + q_len]
                            .transpose(0, 1)
                            .unsqueeze(0)
                        )
                        k_slice = (
                            key_ref[kv_start : kv_start + kv_len]
                            .transpose(0, 1)
                            .unsqueeze(0)
                        )
                        v_slice = (
                            value_ref[kv_start : kv_start + kv_len]
                            .transpose(0, 1)
                            .unsqueeze(0)
                        )
                        mask_slice = mask[
                            batch_index : batch_index + 1,
                            :,
                            :q_len,
                            :kv_len,
                        ]
                        dense_output = F.scaled_dot_product_attention(
                            q_slice,
                            k_slice,
                            v_slice,
                            attn_mask=mask_slice,
                            dropout_p=0.0,
                        )
                        output_ref_parts.append(
                            dense_output.squeeze(0).transpose(0, 1)
                        )
                        q_start += q_len
                        kv_start += kv_len

                output_ref = torch.cat(output_ref_parts, dim=0)
                grad_output = torch.randn_like(output)
                grads = torch.autograd.grad(
                    output, (query, key, value), grad_output
                )
                grads_ref = torch.autograd.grad(
                    output_ref, (query_ref, key_ref, value_ref), grad_output
                )

                self.assertEqual(
                    tuple(logsumexp.shape), (heads, sum(q_lengths))
                )
                self.assertEqual(tuple(rng_state.shape), (2,))
                tolerances = self._tolerances(dtype)
                torch.testing.assert_close(output, output_ref, **tolerances)
                for grad, grad_ref in zip(grads, grads_ref):
                    torch.testing.assert_close(grad, grad_ref, **tolerances)

    def _assert_dropout_replay(self, varlen):
        if not self.supports_mask_dropout:
            self.skipTest("masked dropout requires flash-attn >= 2.8.4")
        torch.manual_seed(2026)
        dropout_p = 0.5
        sequence_length = 16
        head_dim = 64
        unit = 1.0 / (1.0 - dropout_p)
        mask = torch.zeros(
            1,
            1,
            sequence_length,
            sequence_length,
            device="cuda",
            dtype=torch.bool,
        )
        mask[..., 0] = True

        if varlen:
            query = torch.zeros(
                sequence_length,
                1,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            key = torch.zeros_like(query)
            value = torch.ones_like(query, requires_grad=True)
            cu_seqlens = torch.tensor(
                [0, sequence_length], device="cuda", dtype=torch.int32
            )
            output = (
                torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                    query,
                    key,
                    value,
                    mask,
                    cu_seqlens,
                    cu_seqlens,
                    sequence_length,
                    sequence_length,
                    dropout_p,
                    False,
                )[0]
            )
            keep = output[..., 0] != 0
        else:
            query = torch.zeros(
                1,
                1,
                sequence_length,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            key = torch.zeros_like(query)
            value = torch.ones_like(query, requires_grad=True)
            output = torch.ops.aten._scaled_dot_product_flash_attention_masked(
                query,
                key,
                value,
                mask,
                dropout_p,
                False,
            )[0]
            keep = output[..., 0] != 0

        expected_unit = torch.full_like(output, unit)
        self.assertTrue(torch.all((output == 0) | (output == expected_unit)))
        self.assertTrue(
            torch.equal(output != 0, keep.unsqueeze(-1).expand_as(output))
        )
        self.assertTrue(bool(keep.any()))
        self.assertTrue(bool((~keep).any()))

        for query_index in range(sequence_length):
            if varlen:
                loss = output[query_index, 0].sum()
                kept = bool(keep[query_index, 0])
            else:
                loss = output[0, 0, query_index].sum()
                kept = bool(keep[0, 0, query_index])

            grad_value = torch.autograd.grad(
                loss,
                value,
                retain_graph=query_index + 1 < sequence_length,
            )[0]
            expected_grad = torch.zeros_like(value)
            if varlen:
                expected_grad[0, 0, :] = unit if kept else 0.0
            else:
                expected_grad[0, 0, 0, :] = unit if kept else 0.0
            torch.testing.assert_close(
                grad_value, expected_grad, atol=0.0, rtol=0.0
            )

    def test_dense_dropout_forward_backward_replay(self):
        self._assert_dropout_replay(varlen=False)

    def test_varlen_dropout_forward_backward_replay(self):
        self._assert_dropout_replay(varlen=True)

    def test_masked_dropout_version_gate(self):
        q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        packed_q = q.squeeze(0).transpose(0, 1).contiguous()
        dropout_p = 0.1

        params = torch.backends.cuda.SDPAParams(
            q, q, q, mask, dropout_p, False, False
        )
        self.assertEqual(
            torch.backends.cuda.can_use_flash_attention(params),
            self.supports_mask_dropout,
        )

        def dense():
            return torch.ops.aten._scaled_dot_product_flash_attention_masked(
                q, q, q, mask, dropout_p, False
            )[0]

        def varlen():
            return torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                packed_q,
                packed_q,
                packed_q,
                mask,
                cu_seqlens,
                cu_seqlens,
                8,
                8,
                dropout_p,
                False,
            )[0]

        if self.supports_mask_dropout:
            self.assertTrue(bool(torch.isfinite(dense()).all()))
            self.assertTrue(bool(torch.isfinite(varlen()).all()))
        else:
            with self.assertRaisesRegex(RuntimeError, "requires flash-attn >= 2.8.4"):
                dense()
            with self.assertRaisesRegex(RuntimeError, "requires flash-attn >= 2.8.4"):
                varlen()
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                fallback = F.scaled_dot_product_attention(
                    q, q, q, attn_mask=mask, dropout_p=dropout_p
                )
            self.assertTrue(bool(torch.isfinite(fallback).all()))
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                    F.scaled_dot_product_attention(
                        q, q, q, attn_mask=mask, dropout_p=dropout_p
                    )

    def test_meta_and_autocast(self):
        batch, heads, sequence_length, head_dim = 1, 1, 8, 64
        meta_query = torch.empty(
            batch,
            heads,
            sequence_length,
            head_dim,
            device="meta",
            dtype=torch.bfloat16,
        )
        meta_mask = torch.empty(
            batch,
            heads,
            sequence_length,
            sequence_length,
            device="meta",
            dtype=torch.bool,
        )
        dense_meta = torch.ops.aten._scaled_dot_product_flash_attention_masked(
            meta_query,
            meta_query,
            meta_query,
            meta_mask,
            0.0,
            False,
        )
        self.assertEqual(tuple(dense_meta[0].shape), tuple(meta_query.shape))
        self.assertEqual(tuple(dense_meta[1].shape), (batch, heads, sequence_length))
        self.assertEqual(tuple(dense_meta[2].shape), (2,))

        meta_varlen_query = torch.empty(
            sequence_length,
            heads,
            head_dim,
            device="meta",
            dtype=torch.bfloat16,
        )
        meta_cu = torch.empty(2, device="meta", dtype=torch.int32)
        varlen_meta = (
            torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                meta_varlen_query,
                meta_varlen_query,
                meta_varlen_query,
                meta_mask,
                meta_cu,
                meta_cu,
                sequence_length,
                sequence_length,
                0.0,
                False,
            )
        )
        self.assertEqual(
            tuple(varlen_meta[0].shape), tuple(meta_varlen_query.shape)
        )
        self.assertEqual(tuple(varlen_meta[1].shape), (batch, sequence_length))
        self.assertEqual(tuple(varlen_meta[2].shape), (2,))

        dense_query = torch.randn(
            batch,
            heads,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        dense_mask = torch.ones(
            batch,
            heads,
            sequence_length,
            sequence_length,
            device="cuda",
            dtype=torch.bool,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dense_output = (
                torch.ops.aten._scaled_dot_product_flash_attention_masked(
                    dense_query,
                    dense_query,
                    dense_query,
                    dense_mask,
                    0.0,
                    False,
                )[0]
            )
        self.assertEqual(dense_output.dtype, torch.bfloat16)

        varlen_query = torch.randn(
            sequence_length,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        cu_seqlens = torch.tensor(
            [0, sequence_length], device="cuda", dtype=torch.int32
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            varlen_output = (
                torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                    varlen_query,
                    varlen_query,
                    varlen_query,
                    dense_mask,
                    cu_seqlens,
                    cu_seqlens,
                    sequence_length,
                    sequence_length,
                    0.0,
                    False,
                )[0]
            )
        self.assertEqual(varlen_output.dtype, torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is unavailable")
    def test_inductor(self):
        batch, heads, sequence_length, head_dim = 1, 1, 8, 64
        mask = torch.ones(
            batch,
            heads,
            sequence_length,
            sequence_length,
            device="cuda",
            dtype=torch.bool,
        )
        query = torch.randn(
            batch,
            heads,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        def dense_fn(q, k, v, attn_mask):
            return torch.ops.aten._scaled_dot_product_flash_attention_masked(
                q, k, v, attn_mask, 0.0, False
            )[0]

        dense_eager = dense_fn(query, key, value, mask)
        dense_compiled = torch.compile(dense_fn, fullgraph=True)(
            query, key, value, mask
        )
        torch.testing.assert_close(dense_compiled, dense_eager)

        varlen_query = query.view(sequence_length, heads, head_dim)
        varlen_key = key.view(sequence_length, heads, head_dim)
        varlen_value = value.view(sequence_length, heads, head_dim)
        cu_seqlens = torch.tensor(
            [0, sequence_length], device="cuda", dtype=torch.int32
        )

        def varlen_fn(q, k, v, attn_mask, cu_q, cu_k):
            return (
                torch.ops.aten._scaled_dot_product_flash_attention_varlen_masked(
                    q,
                    k,
                    v,
                    attn_mask,
                    cu_q,
                    cu_k,
                    sequence_length,
                    sequence_length,
                    0.0,
                    False,
                )[0]
            )

        varlen_eager = varlen_fn(
            varlen_query,
            varlen_key,
            varlen_value,
            mask,
            cu_seqlens,
            cu_seqlens,
        )
        varlen_compiled = torch.compile(varlen_fn, fullgraph=True)(
            varlen_query,
            varlen_key,
            varlen_value,
            mask,
            cu_seqlens,
            cu_seqlens,
        )
        torch.testing.assert_close(varlen_compiled, varlen_eager)


if __name__ == "__main__":
    unittest.main()
