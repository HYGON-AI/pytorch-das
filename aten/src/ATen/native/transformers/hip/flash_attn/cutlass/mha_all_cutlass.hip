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
#include <ATen/TensorUtils.h>
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
#include <limits>

namespace pytorch_flash {


bool has_cutlass_attnmask_api() {
    static const bool available = [] {
        try {
            return sdp::cutlassfa_adapter::CUTLASSFA::get_instance().has_attnmask_api();
        } catch (const std::exception&) {
            return false;
        }
    }();
    return available;
}

bool has_cutlass_varlen_attnmask_api() {
    static const bool available = [] {
        try {
            return sdp::cutlassfa_adapter::CUTLASSFA::get_instance()
                .has_varlen_attnmask_api();
        } catch (const std::exception&) {
            return false;
        }
    }();
    return available;
}

bool supports_cutlass_attnmask_dropout() {
    static const bool supported = [] {
        try {
            return sdp::cutlassfa_adapter::CUTLASSFA::get_instance()
                .supports_attnmask_dropout();
        } catch (const std::exception&) {
            return false;
        }
    }();
    return supported;
}

void check_cutlass_attnmask_dropout(double dropout_p) {
    TORCH_CHECK(
        dropout_p == 0.0 || supports_cutlass_attnmask_dropout(),
        "Masked Flash attention with dropout_p > 0 requires flash-attn >= 2.8.4");
}

void check_attnmask_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask) {
    constexpr auto* c = "masked Flash attention";
    const at::TensorArg q{query, "query", 1}, k{key, "key", 2},
        v{value, "value", 3}, m{mask, "attn_mask", 4};
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested() && !value.is_nested() &&
        !mask.is_nested(), "Masked Flash attention requires dense inputs");
    at::checkLayout(c, {query, key, value, mask}, at::kStrided);
    for (const auto& t : {q, k, v, m}) {
        at::checkDim(c, t, 4);
    }
    TORCH_CHECK(query.is_cuda(), "Masked Flash attention requires GPU inputs");
    at::checkDeviceType(c, {key, value, mask}, query.device().type());
    at::checkAllSameGPU(c, {q, k, v, m});
    at::checkScalarTypes(c, q, {at::kHalf, at::kBFloat16});
    // Direct ATen calls bypass the SDPA selector; match the Meta input contract.
    at::checkScalarType(c, k, query.scalar_type());
    at::checkScalarType(c, v, query.scalar_type());
    at::checkScalarType(c, m, at::kBool);
    TORCH_CHECK(
        query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1,
        "Masked Flash attention requires Q/K/V last dimension stride to be 1");
    const auto batch = query.size(0);
    const auto heads = query.size(1);
    const auto kv_heads = key.size(1);
    const auto length_q = query.size(2);
    const auto length_k = key.size(2);
    const auto dim = query.size(3);
    TORCH_CHECK(
        batch > 0 && heads > 0 && kv_heads > 0 && length_q > 0 && length_k > 0 &&
        key.size(0) == batch && value.size(0) == batch &&
        value.size(1) == kv_heads && heads % kv_heads == 0 &&
        value.size(2) == length_k &&
        (dim == 64 || dim == 128) && key.size(3) == dim && value.size(3) == dim,
        "Masked Flash attention requires compatible batch/heads/lengths and head_dim 64 or 128");
    at::checkSize(c, m, {batch, heads, length_q, length_k});
    at::checkContiguous(c, m);
}

void check_varlen_attnmask_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k) {
    constexpr auto* c = "varlen masked Flash attention";
    const at::TensorArg q{query, "query", 1}, k{key, "key", 2},
        v{value, "value", 3}, m{mask, "attn_mask", 4},
        cu_q{cumulative_sequence_length_q, "cumulative_sequence_length_q", 5},
        cu_k{cumulative_sequence_length_k, "cumulative_sequence_length_k", 6};
    at::checkLayout(c, {query, key, value, mask,
        cumulative_sequence_length_q, cumulative_sequence_length_k}, at::kStrided);
    for (const auto& t : {q, k, v}) {
        at::checkDim(c, t, 3);
    }
    at::checkDim(c, m, 4);
    at::checkDim(c, cu_q, 1);
    at::checkDim(c, cu_k, 1);
    TORCH_CHECK(query.is_cuda(), "Varlen masked Flash attention requires GPU inputs");
    at::checkDeviceType(c, {key, value, mask,
        cumulative_sequence_length_q, cumulative_sequence_length_k},
        query.device().type());
    at::checkAllSameGPU(c, {q, k, v, m, cu_q, cu_k});
    at::checkScalarTypes(c, q, {at::kHalf, at::kBFloat16});
    // Varlen's direct ATen entry also has no SDPA selector before the FA ABI.
    at::checkScalarType(c, k, query.scalar_type());
    at::checkScalarType(c, v, query.scalar_type());
    at::checkScalarType(c, m, at::kBool);
    at::checkScalarType(c, cu_q, at::kInt);
    at::checkScalarType(c, cu_k, at::kInt);
    TORCH_CHECK(
        query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1,
        "Varlen masked Flash attention requires Q/K/V last dimension stride to be 1");
    const auto heads = query.size(1);
    const auto kv_heads = key.size(1);
    const auto dim = query.size(2);
    const auto batch = cumulative_sequence_length_q.numel() - 1;
    TORCH_CHECK(
        query.size(0) > 0 && key.size(0) > 0 && heads > 0 && kv_heads > 0 &&
        value.size(0) == key.size(0) && value.size(1) == kv_heads &&
        heads % kv_heads == 0 && (dim == 64 || dim == 128) &&
        key.size(2) == dim && value.size(2) == dim,
        "Varlen masked Flash attention requires compatible packed lengths, "
        "heads, and head_dim 64 or 128");
    TORCH_CHECK(batch > 0 && max_seqlen_q > 0 && max_seqlen_k > 0,
        "Varlen masked Flash attention requires a nonempty batch and positive max lengths");
    at::checkSize(c, cu_k, {batch + 1});
    at::checkSize(c, m, {batch, heads, max_seqlen_q, max_seqlen_k});
    at::checkAllContiguous(c, {m, cu_q, cu_k});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mha_fwd_attnmask_cutlass(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    double dropout_p,
    bool is_causal,
    float scale) {
    TORCH_CHECK(has_cutlass_attnmask_api(), "Compatible Cutlass FA mask API is unavailable");
    check_cutlass_attnmask_dropout(dropout_p);
    auto q = query;
    std::optional<at::Tensor> out = at::empty(q.sizes(), q.options());
    std::optional<at::Tensor> alibi;
    const std::optional<at::Tensor> aux;
    auto& fa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    auto result = fa.mha_fwd_attnmask(
        q, key, value, mask, out, alibi, dropout_p, scale, is_causal, -1, -1, 0.0f,
        -std::numeric_limits<float>::infinity(), false, std::nullopt, false, aux, aux);
    TORCH_CHECK(result.size() == 8, "Unexpected Cutlass FA mask forward return count");
    return {result[0], result[5], result[7]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mha_bwd_attnmask_cutlass(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& rng_state,
    double dropout_p,
    bool is_causal,
    float scale) {
    TORCH_CHECK(has_cutlass_attnmask_api(), "Compatible Cutlass FA mask API is unavailable");
    check_cutlass_attnmask_dropout(dropout_p);
    std::optional<at::Tensor> dq = at::empty(query.sizes(), query.options());
    std::optional<at::Tensor> dk = at::empty(key.sizes(), key.options());
    std::optional<at::Tensor> dv = at::empty(value.sizes(), value.options());
    std::optional<at::Tensor> alibi;
    std::optional<at::Tensor> state = rng_state;
    const auto& ctx = at::globalContext();
    const bool deterministic = ctx.deterministicAlgorithms();
    auto& fa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    auto result = fa.mha_bwd_attnmask(
        grad_out, query, key, value, out, logsumexp, mask, dq, dk, dv, alibi,
        dropout_p, scale, is_causal, -1, -1, 0.0f, deterministic, std::nullopt, state, false);
    TORCH_CHECK(result.size() == 4, "Unexpected Cutlass FA mask backward return count");
    return {result[0], result[1], result[2]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_attnmask_cutlass(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double dropout_p,
    bool is_causal,
    float scale) {
    TORCH_CHECK(
        has_cutlass_varlen_attnmask_api(),
        "Compatible Cutlass FA varlen mask API is unavailable");
    check_cutlass_attnmask_dropout(dropout_p);
    auto q = query;
    std::optional<at::Tensor> out = at::empty(q.sizes(), q.options());
    std::optional<at::Tensor> seqused_k;
    std::optional<at::Tensor> alibi;
    const std::optional<at::Tensor> aux;
    auto& fa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    auto result = fa.mha_varlen_fwd_attnmask(
        q,
        key,
        value,
        mask,
        out,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        seqused_k,
        alibi,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        dropout_p,
        scale,
        false,
        is_causal,
        -1,
        -1,
        0.0f,
        -std::numeric_limits<float>::infinity(),
        false,
        std::nullopt,
        aux);
    TORCH_CHECK(
        result.size() == 8,
        "Unexpected Cutlass FA varlen mask forward return count");
    return {result[0], result[5], result[7]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd_attnmask_cutlass(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& rng_state,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double dropout_p,
    bool is_causal,
    float scale) {
    TORCH_CHECK(
        has_cutlass_varlen_attnmask_api(),
        "Compatible Cutlass FA varlen mask API is unavailable");
    check_cutlass_attnmask_dropout(dropout_p);
    std::optional<at::Tensor> dq = at::empty(query.sizes(), query.options());
    std::optional<at::Tensor> dk = at::empty(key.sizes(), key.options());
    std::optional<at::Tensor> dv = at::empty(value.sizes(), value.options());
    std::optional<at::Tensor> alibi;
    std::optional<at::Tensor> state = rng_state;
    const auto& ctx = at::globalContext();
    const bool deterministic = ctx.deterministicAlgorithms();
    auto& fa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    auto result = fa.mha_varlen_bwd_attnmask(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        mask,
        dq,
        dk,
        dv,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        alibi,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        dropout_p,
        scale,
        false,
        is_causal,
        -1,
        -1,
        0.0f,
        deterministic,
        std::nullopt,
        state);
    TORCH_CHECK(
        result.size() == 4,
        "Unexpected Cutlass FA varlen mask backward return count");
    return {result[0], result[1], result[2]};
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_cutlass(const at::Tensor &q_,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        std::optional<int64_t> window_size_left,
        std::optional<int64_t> window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
    int64_t win_left  = window_size_left.value_or(-1);
    int64_t win_right = window_size_right.value_or(-1);

    auto* ev = std::getenv("USE_FA_BHSD");
    bool is_bhsd = ev && std::string(ev) == "1";
    at::Tensor q = q_;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto q_dtype = q.dtype();
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res;
    if (cfa.get_fa_changed()) {
        res = cfa.mha_fwd_new(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, 
                                                static_cast<int>(win_left), static_cast<int>(win_right), 
                                                softcap, return_softmax, gen_, is_bhsd,c10::nullopt, 0.0f);
    } else {
        res = cfa.mha_fwd(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, 
                                                static_cast<int>(win_left), static_cast<int>(win_right), 
                                                softcap, return_softmax, gen_, is_bhsd);
    }

    at::Tensor out = res[0];
    at::Tensor q_padded = res[1];
    at::Tensor k_padded = res[2];
    at::Tensor v_padded = res[3];
    at::Tensor softmax_lse = res[5];
    at::Tensor rng_state = res[7];
    at::Tensor p = res[6];
    at::Tensor seed_t, offset_t;

    seed_t = rng_state[0];
    offset_t = rng_state[1];

    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_cutlass(const at::Tensor &q_,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_,
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               std::optional<int64_t> window_size_left,
               std::optional<int64_t> window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {
    int64_t win_left  = window_size_left.value_or(-1);
    int64_t win_right = window_size_right.value_or(-1);
    at::Tensor q = q_;
    std::optional<const at::Tensor> leftpad_k_ = c10::nullopt; // batch_size
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res;
    if (cfa.get_fa_changed()) {
	const c10::optional<at::Tensor> q_descale = c10::nullopt;
	const c10::optional<at::Tensor> k_descale = c10::nullopt;
	const c10::optional<at::Tensor> v_descale = c10::nullopt;
	const c10::optional<at::Tensor> s_aux = c10::nullopt;
        res = cfa.mha_varlen_fwd_new(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k,
                                                leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q, max_seqlen_k,
                                                p_dropout, softmax_scale, zero_tensors, is_causal, static_cast<int>(win_left), 
                                                static_cast<int>(win_right),softcap, return_softmax, 
						q_descale, k_descale, v_descale, gen_, s_aux);
    } else {
        res = cfa.mha_varlen_fwd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k,
                                                leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q, max_seqlen_k,
                                                p_dropout, softmax_scale, zero_tensors, is_causal, static_cast<int>(win_left), 
                                                static_cast<int>(win_right),softcap, return_softmax, gen_);
    }

    
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
    seed_t = rng_state[0];
    offset_t = rng_state[1];
    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};

}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd_cutlass(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        std::optional<int64_t> window_size_left,
        std::optional<int64_t> window_size_right,
        const float softcap,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {
    int64_t win_left  = window_size_left.value_or(-1);
    int64_t win_right = window_size_right.value_or(-1);
    auto* ev = std::getenv("USE_FA_BHSD");
    bool is_bhsd = ev && std::string(ev) == "1";
    auto rng_state_ = at::empty({2}, q.options().dtype(at::kLong));
    rng_state_[0] = philox_seed;
    rng_state_[1] = philox_offset;
    std::optional<at::Tensor> rng_state(rng_state_);
    std::optional<at::Generator> gen_ = c10::nullopt;
    
    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_,
                                                p_dropout, softmax_scale, is_causal, static_cast<int>(win_left), static_cast<int>(win_right),
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
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               std::optional<int64_t> window_size_left,
               std::optional<int64_t> window_size_right,
               const float softcap,
               const bool deterministic,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset) {
    int64_t win_left  = window_size_left.value_or(-1);
    int64_t win_right = window_size_right.value_or(-1);
    std::optional<at::Generator> gen_ = c10::nullopt;
    auto rng_state_ = at::empty({2}, q.options().dtype(at::kLong));
    rng_state_[0] = philox_seed;
    rng_state_[1] = philox_offset;
    std::optional<at::Tensor> rng_state(rng_state_);

    auto& cfa = sdp::cutlassfa_adapter::CUTLASSFA::get_instance();
    std::vector<at::Tensor> res = cfa.mha_varlen_bwd(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, cu_seqlens_q,
                                                cu_seqlens_k, alibi_slopes_, max_seqlen_q, max_seqlen_k,
                                                p_dropout, softmax_scale, zero_tensors, is_causal, static_cast<int>(win_left), static_cast<int>(win_right),
                                                softcap, deterministic, gen_, rng_state);
    at::Tensor dq = res[0];
    at::Tensor dk = res[1];
    at::Tensor dv = res[2];
    at::Tensor softmax_d = res[3];
    return { dq, dk, dv, softmax_d};
}
}
