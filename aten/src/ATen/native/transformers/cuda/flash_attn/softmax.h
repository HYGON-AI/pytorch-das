/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace pytorch_flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            // float ori = summary(mi);
            summary(mi) = op(summary(mi), tensor(mi, ni));
            // wangaq debug
            // if (thread0()) {
            //     printf("thread_reduce_ mi:%d ni:%d %7.4f %7.4f %7.4f\n", mi, ni, ori, tensor(mi, ni), summary(mi));
            // }
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<64>::run(src(i), op);
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f \n", threadIdx.x,
        //     src(i), dst(i));
        // }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_sum_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    int tidx = threadIdx.x % 64;
    float a, b = 1.0;
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        v4f d = {0};
        a = src(i);
        d = __builtin_amdgcn_mmac_f32_16x16x4f32(a, b, d);
        dst(i) = d.x;
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f "
        //     "D:%10.4f %10.4f %10.4f %10.4f sum:%7.4f\n", threadIdx.x,
        //     a, b,
        //     d[0], d[1], d[2], d[3], dst(i));
        // }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_with_mmac_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    int tidx = threadIdx.x % 64;
    float a, b = 0.0 + (tidx / 4 == 0 && tidx / 16 == 0) + (tidx / 4 == 5 && tidx / 16 == 1) + 
        (tidx / 4 == 10 && tidx / 16 == 2) + (tidx / 4 == 15 && tidx / 16 == 3);
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        v4f d = {0};
        a = src(i) == -INFINITY ? -10000.0 : src(i);
        d = __builtin_amdgcn_mmac_f32_16x16x4f32(a, b, d);
        dst(i) = isnan(d.x) ? -INFINITY : d.x;
        dst(i) = op(dst(i), isnan(d.y) ? -INFINITY : d.y);
        dst(i) = op(dst(i), isnan(d.z) ? -INFINITY : d.z);
        dst(i) = op(dst(i), isnan(d.w) ? -INFINITY : d.w);
        // if (blockIdx.x == 0) {
        //     printf("tid:%3d A:%7.4f B:%7.4f "
        //     "D:%10.4f %10.4f %10.4f %10.4f max:%7.4f\n", threadIdx.x,
        //     a, b,
        //     d[0], d[1], d[2], d[3], dst(i));
        // }
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    #if 1
    quad_allreduce_(summary, summary, op);
    #else
    quad_allreduce_with_mmac_(summary, summary, op);
    #endif
    
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
        #if 0
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        #else
            tensor(mi, ni) = __llvm_exp2_f32(tensor(mi, ni) * scale - max_scaled);
        #endif
        }
    }
}

// Apply the exp to all the elements.
// template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
// __forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
//     static_assert(Layout0::rank == 2, "Only support 2D Tensor");
//     static_assert(Layout1::rank == 1, "Only support 1D Tensor");
//     CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
//     #pragma unroll
//     for (int mi = 0; mi < size<0>(tensor); ++mi) {
//         MaxOp<float> max_op;
//         max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
//         #pragma unroll
//         for (int ni = 1; ni < size<1>(tensor); ni++) {
//             max(mi) = max_op(max(mi), tensor(mi, ni));
//         }
//         max(mi) = Allreduce<4>::run(max(mi), max_op);
//         // If max is -inf, then all elements must have been -inf (possibly due to masking).
//         // We don't want (-inf - (-inf)) since that would give NaN.
//         const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
//         sum(mi) = 0;
//         #pragma unroll
//         for (int ni = 0; ni < size<1>(tensor); ++ni)  {
//             // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
//             // max * log_2(e)) This allows the compiler to use the ffma
//             // instruction instead of fadd and fmul separately.
//             tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
//             sum(mi) += tensor(mi, ni);
//         }
//         SumOp<float> sum_op;
//         sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
//     }
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // wangaq debug
        // __syncthreads();
        // if (thread0()) {
        //     printf("scores %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f "
        //     "%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f "
        //     "%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f "
        //     "%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n", 
        //     scores(0, 0),  scores(0, 1),  scores(0, 2),  scores(0, 3), 
        //     scores(0, 4),  scores(0, 5),  scores(0, 6),  scores(0, 7), 
        //     scores(0, 8),  scores(0, 9),  scores(0, 10), scores(0, 11), 
        //     scores(0, 12), scores(0, 13), scores(0, 14), scores(0, 15),
        //     scores(1, 0),  scores(1, 1),  scores(1, 2),  scores(1, 3), 
        //     scores(1, 4),  scores(1, 5),  scores(1, 6),  scores(1, 7), 
        //     scores(1, 8),  scores(1, 9),  scores(1, 10), scores(1, 11), 
        //     scores(1, 12), scores(1, 13), scores(1, 14), scores(1, 15)
        //     );
        // }
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            pytorch_flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            pytorch_flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            pytorch_flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            pytorch_flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
#if 0
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
#else
                float scores_scale = __llvm_exp2_f32((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
#endif
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            pytorch_flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            pytorch_flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
        
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        #if 1
        quad_allreduce_(row_sum, row_sum, sum_op);
        #else
        quad_allreduce_sum_(row_sum, row_sum, sum_op);
        #endif
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};


}  // namespace pytorch_flash
