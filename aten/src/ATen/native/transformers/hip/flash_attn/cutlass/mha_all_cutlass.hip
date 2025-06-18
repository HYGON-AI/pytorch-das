#include <ATen/dtk_macros.h>
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/nn/functional.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <ATen/native/transformers/hip/flash_attn/flash_api.h>

#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <vector>
#include <cstdint>
#include <tuple>
#include <iostream>
#include <ATen/ops/zeros.h>

#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#endif

#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>

#include <ATen/native/transformers/hip/cutlassfa_adapter.h>

namespace pytorch_flash {


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_cutlass(const at::Tensor &q_,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
    auto* ev = std::getenv("USE_FA_BHSD");
    bool is_bhsd = ev && std::string(ev) == "1";
    at::Tensor q = q_;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto q_dtype = q.dtype();
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_fwd(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, 
                                                window_size_left, window_size_right, softcap, return_softmax, gen_, is_bhsd);
    at::Tensor out = res[0];
    at::Tensor q_padded = res[1];
    at::Tensor k_padded = res[2];
    at::Tensor v_padded = res[3];
    at::Tensor softmax_lse = res[5];
    at::Tensor rng_state = res[7];
    at::Tensor p = res[6];
    at::Tensor seed_t, offset_t;

    seed_t = at::Tensor(std::move(rng_state[0].clone()));
    offset_t = at::Tensor(std::move(rng_state[1].clone()));

    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_cutlass(const at::Tensor &q_,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_,
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {
    at::Tensor q = q_;
    c10::optional<const at::Tensor> leftpad_k_ = c10::nullopt; // batch_size
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_varlen_fwd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k,
                                                leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q, max_seqlen_k,
                                                p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, 
                                                window_size_right,softcap, return_softmax, gen_);
    
    //return {out, q_padded, k_padded, v_padded, softmax_lse, rng_state[0], rng_state[1], p};
    at::Tensor out = res[0];
    at::Tensor q_padded = res[1];
    at::Tensor k_padded = res[2];
    at::Tensor v_padded = res[3];
    at::Tensor softmax_lse = res[5];
    // FIXME: 
    at::Tensor rng_state = res[7];
    at::Tensor p = res[6];
    at::Tensor seed_t, offset_t;
    seed_t = at::Tensor(std::move(rng_state[0].clone()));
    offset_t = at::Tensor(std::move(rng_state[1].clone()));
    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};

}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd_cutlass(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {
    auto* ev = std::getenv("USE_FA_BHSD");
    bool is_bhsd = ev && std::string(ev) == "1";
    auto rng_state_ = at::empty({2}, q.options().dtype(at::kLong));
    rng_state_[0] = philox_seed;
    rng_state_[1] = philox_offset;
    c10::optional<at::Tensor> rng_state(rng_state_);
    c10::optional<at::Generator> gen_ = c10::nullopt;
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_,
                                                p_dropout, softmax_scale, is_causal, window_size_left, window_size_right,
                                                softcap, deterministic, gen_, rng_state, is_bhsd);
    at::Tensor dq = res[0];
    at::Tensor dk = res[1];
    at::Tensor dv = res[2];
    at::Tensor softmax_d = res[3];

    return { dq, dk, dv, softmax_d};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd_cutlass(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
               c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset) {
    c10::optional<at::Generator> gen_ = c10::nullopt;
    auto rng_state_ = at::empty({2}, q.options().dtype(at::kLong));
    rng_state_[0] = philox_seed;
    rng_state_[1] = philox_offset;
    c10::optional<at::Tensor> rng_state(rng_state_);

    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_varlen_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, cu_seqlens_q,
                                                cu_seqlens_k, alibi_slopes_, max_seqlen_q, max_seqlen_k,
                                                p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right,
                                                softcap, deterministic, gen_, rng_state);
    at::Tensor dq = res[0];
    at::Tensor dk = res[1];
    at::Tensor dv = res[2];
    at::Tensor softmax_d = res[3];
    return { dq, dk, dv, softmax_d};
}
}

