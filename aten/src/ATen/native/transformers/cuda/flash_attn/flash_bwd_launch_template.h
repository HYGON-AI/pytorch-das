/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>
#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_bwd_preprocess_kernel.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_bwd_kernel.h>

namespace pytorch_flash {
// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#if defined(DCU_ASM)
    #define ARCH_SUPPORTS_FLASH
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__launch_bounds__(Kernel_traits::kNThreads) \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_kernel, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
    pytorch_flash::compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dk_dv_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dk_dv_trans_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dk_dv_trans_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dk_dv_trans_16x64_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dk_dv_trans_seqk_parallel_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dk_dv_trans_16x64_prefetch, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dk_dv_trans_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_loop_seqq_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dq_seqq_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_loop_16x64_seqq_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dq_seqq_parallel_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_loop_16x64_prefetch_seqq_parallel_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));  // If Is_local is true, Is_causal should be false
        pytorch_flash::compute_dq_seqq_parallel_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    pytorch_flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(const Flash_bwd_params params) {
    pytorch_flash::clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    pytorch_flash::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(const Flash_bwd_params params) {
    pytorch_flash::convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);
    // printf("run_flash_bwd_seqk_parallel: grid_m=%d, %d, %d, \n", grid_m.x, grid_m.y, grid_m.z);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // printf("flash_bwd_dot_do_o_kernel done, params.deterministic=%d, params.seqlen_q=%d, params.seqlen_k=%d, \n", 
    //     params.deterministic, params.seqlen_q, params.seqlen_k);

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    // printf("run_flash_bwd_seqk_parallel: grid_n=%d, %d, %d, \n", grid_n.x, grid_n.y, grid_n.z);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                        // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                        // If Is_local, set Is_causal to false
                        auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                            Kernel_traits, 
                            Is_dropout && !Is_softcap, Is_causal, 
                            Is_local && !Is_causal, 
                            Has_alibi, 
                            IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                            IsEvenKConst, 
                            Is_softcap>;
                        
                        kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    // if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
    //     C10_CUDA_CHECK(cudaFuncSetAttribute(
    //         kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    // }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename Kernel_traits, typename Kernel_trans_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_separate_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);
    // printf("run_flash_bwd_seqk_parallel: grid_m=%d, %d, %d, \n", grid_m.x, grid_m.y, grid_m.z);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // printf("flash_bwd_dot_do_o_kernel done, params.deterministic=%d, params.seqlen_q=%d, params.seqlen_k=%d, \n", 
    //     params.deterministic, params.seqlen_q, params.seqlen_k);

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    
    #ifdef BWDTRANS
    constexpr int smem_size_dq_dk_dv = Kernel_trans_traits::kSmemSizeTrans1colblock;
    #else
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    #endif
    constexpr int smem_size_dq = Kernel_traits::kSmemSize1rowblock;
    // printf("smem_size_dq_dk_dv = %d smem_size_dq = %d\n", smem_size_dq_dk_dv, smem_size_dq);
    // printf("run_flash_bwd_seqk_parallel: grid_n=%d, %d, %d, \n", grid_n.x, grid_n.y, grid_n.z);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                        // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                        // If Is_local, set Is_causal to false
                        #ifdef BWDTRANS
                        auto kernel = &flash_bwd_dk_dv_trans_loop_seqk_parallel_kernel<
                            Kernel_trans_traits, 
                            Is_dropout && !Is_softcap, Is_causal, 
                            Is_local && !Is_causal, 
                            Has_alibi, 
                            IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                            IsEvenKConst, 
                            Is_softcap>;
                        #else
                        auto kernel = &flash_bwd_dk_dv_loop_seqk_parallel_kernel<
                            Kernel_traits, 
                            Is_dropout && !Is_softcap, Is_causal, 
                            Is_local && !Is_causal, 
                            Has_alibi, 
                            IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                            IsEvenKConst, 
                            Is_softcap>;
                        #endif
                        
                        kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                        auto kernel_dq = flash_bwd_dq_loop_seqq_parallel_kernel<
                            Kernel_traits, 
                            Is_dropout && !Is_softcap, Is_causal, 
                            Is_local && !Is_causal, 
                            Has_alibi, 
                            IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                            IsEvenKConst, 
                            Is_softcap>;
                        kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();

                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, typename Kernel_trans_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_separate_seqk_parallel_trans(Flash_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.h, params.b);
    dim3 grid_m_do(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    // if (params.deterministic) {
    //     auto dprops = at::cuda::getCurrentDeviceProperties();
    //     gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) / (params.b * params.h);
    // }
    dim3 grid_n(num_n_block, params.h, params.b);
    // printf("run_flash_bwd_seqk_parallel: grid_m=%d, %d, %d, \n", grid_m.x, grid_m.y, grid_m.z);

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m_do, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m_do, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // printf("flash_bwd_dot_do_o_kernel done, params.deterministic=%d, params.seqlen_q=%d, params.seqlen_k=%d, \n", 
    //     params.deterministic, params.seqlen_q, params.seqlen_k);

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_MN_trans = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_trans_traits::kBlockM == 0 && params.seqlen_k % Kernel_trans_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
        // printf("is_even_MN = %d Kernel_traits::kBlockN = %d params.seqlen_k = %d\n", is_even_MN, Kernel_traits::kBlockN, params.seqlen_k);

// printf("is_even_MN = %d Kernel_traits::kBlockN = %d\n", is_even_MN, Kernel_traits::kBlockN);
    #if 1
    constexpr int smem_size_dq_dk_dv = Kernel_trans_traits::kSmemSizeTrans1colblock;
    #else
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    #endif
    constexpr int smem_size_dq = Kernel_traits::kSmemSize1rowblock;
    // printf("smem_size_dq_dk_dv = %d smem_size_dq = %d\n", smem_size_dq_dk_dv, smem_size_dq);
    // printf("run_flash_bwd_seqk_parallel: grid_n=%d, %d, %d, \n", grid_n.x, grid_n.y, grid_n.z);
    BOOL_SWITCH(is_even_MN_trans, IsEvenMNTransConst, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // // If Is_local, set Is_causal to false
                            auto kernel = &flash_bwd_dk_dv_trans_16x64_loop_seqk_parallel_kernel<
                                Kernel_trans_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNTransConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                        
                            kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                            auto kernel_dq = flash_bwd_dq_loop_16x64_seqq_parallel_kernel<
                                Kernel_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                            kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();

                        });
                    });
                });
            });
        });
    });

}


template<typename Kernel_traits, typename Kernel_trans_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_separate_prefetch(Flash_bwd_params &params, cudaStream_t stream) {
    // const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    
#ifdef NO_CAUSAL_OPT
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
#else
    const int non_causal_num_n_block =  (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    const int num_n_block = Is_causal ? (non_causal_num_n_block + 1 ) >> 1 : 
        non_causal_num_n_block;
    const int non_causal_num_m_block =  (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_m_block = Is_causal ? (non_causal_num_m_block + 1 ) >> 1 : 
        non_causal_num_m_block;
#endif
    dim3 grid_m(num_m_block, params.h, params.b);
    dim3 grid_m_do((params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM, params.b, params.h);
    dim3 grid_n(num_n_block, params.h, params.b);

    flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m_do, Kernel_traits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_MN_trans = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_trans_traits::kBlockM == 0 && params.seqlen_k % Kernel_trans_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dk_dv = Kernel_trans_traits::kSmemPrefetchSize;
    constexpr int smem_size_dq = Kernel_traits::kSmemPrefetchSize;
    BOOL_SWITCH(is_even_MN_trans, IsEvenMNTransConst, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_bwd_dk_dv_trans_16x64_prefetch<
                                Kernel_trans_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNTransConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                        
                            kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dk_dv, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                            auto kernel_dq = flash_bwd_dq_loop_16x64_prefetch_seqq_parallel_kernel<
                                Kernel_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                            kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();

                        });
                    });
                });
            });
        });
    });

}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout, Is_causal>(params, stream);
#endif
}

template<typename Kernel_dq_traits, typename Kernel_dkdv_traits, bool Is_dropout, bool Is_causal>
void run_flash_separate_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_separate_seqk_parallel<Kernel_dq_traits, Kernel_dkdv_traits, Is_dropout, Is_causal>(params, stream);
#endif
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim32..\n");
    constexpr static int Headdim = 32;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
        //     if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
        //         run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        //     } else {
        //         run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        //     }
        // } else {  // 96 KB
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // }
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 32, 4, 1, 1, 1, true, true, T>, Is_dropout, Is_causal>(params, stream);
        
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim64..\n");
    constexpr static int Headdim = 64;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    // // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // Changing AtomLayoutMdQ from 2 to 4 takes the same time
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 2, 4, 4, false, false, T>>(params, stream);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
        // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
        // if (max_smem_per_block >= 144 * 1024) {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        //     // This has a lot of register spilling
        //     // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
        // } else {
        //     // if (params.h == params.h_k) {
        //         // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream);
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        //         // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream);
        //         // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout>(params, stream);
        //     // } else {
        //     // }
        // }

        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 32, 4, 4, 2, 4, true, false, T>, Is_dropout>(params, stream);
        
        using kernel_traits = Flash_bwd_kernel_dq_16x64_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_16x64_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/128, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_bwd_separate_seqk_parallel_trans<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);

    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim96..\n");
    constexpr static int Headdim = 96;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    // // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 116 * 1024) {
        //     if constexpr(!Is_dropout) {  // 92KB
        //         run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        //     } else {  // 116 KB
        //         // This is faster for dropout since we don't have many registers to spare
        //         run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        //     }
        // } else {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // }
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936") {
            using kernel_trans_traits = Flash_bwd_kernel_trans_16x64_prefetch_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4, T, 3>;
            using kernel_traits = Flash_bwd_kernel_dq_16x64_prefetch_traits<Headdim, /*kBlockM_*/128, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T, 3>;
            run_flash_bwd_separate_prefetch<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        } else {
            using kernel_trans_traits = Flash_bwd_kernel_trans_16x64_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4,
                /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
                /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
            // run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
            if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
                using kernel_traits = Flash_bwd_kernel_dq_16x64_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4,
                /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
                /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
                run_flash_bwd_separate_seqk_parallel_trans<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
            }
            else {
                using kernel_traits = Flash_bwd_kernel_dq_16x64_traits<Headdim, /*kBlockM_*/128, /*kBlockN_*/64, /*kNWarps_*/4,
                /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
                /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
                run_flash_bwd_separate_seqk_parallel_trans<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
            }
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim160(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim160..\n");
    constexpr static int Headdim = 160;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 116 * 1024) {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // } else {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        // }
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/32, /*kNWarps_*/2,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/2, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true, T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/4,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim192..\n");
    constexpr static int Headdim = 192;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 136 * 1024) {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // } else {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // }
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/32, /*kNWarps_*/2,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/2, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/4,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim224(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim224..\n");
    constexpr static int Headdim = 224;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 136 * 1024) {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // } else {
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // }
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/32, /*kNWarps_*/2,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/2, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/4,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("run_mha_bwd_hdim256..\n");
    constexpr static int Headdim = 256;
    // int device;
    // cudaGetDevice(&device);
    // int max_smem_per_block;
    // cudaError status_ = cudaDeviceGetAttribute(
    //     &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // if (status_ != cudaSuccess) {
    //   C10_CUDA_CHECK(status_);
    // }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (max_smem_per_block >= 176 * 1024) {  // H100
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // } else if (max_smem_per_block >= 144 * 1024) {  // A100, we don't do double buffering to save smem
        //     run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout, Is_causal>(params, stream);
        // } else { // sm86 and sm89, max smem is 99 KB. Only works without dropout. V in regs and no double buffering.
        //     if constexpr (!Is_dropout) {
        //         run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, false, Is_causal>(params, stream);
        //     }
        // }
        #ifdef BWDSEPARATE
        using kernel_traits = Flash_bwd_kernel_dq_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/16, /*kNWarps_*/2,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/2, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true,  T>;
        using kernel_trans_traits = Flash_bwd_kernel_trans_traits<Headdim, /*kBlockM_*/32, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/4, /*AtomLayoutMdQ*/1,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>;
        run_flash_separate_bwd<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
        // run_flash_separate_bwd<dq_traits, Is_dropout, Is_causal>(params, stream);
        #else
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/32, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/2, /*AtomLayoutNdKV*/2, /*AtomLayoutMdQ*/4,
            /*Is_V_in_regs_*/true, /*No_double_buffer_*/true, T>, Is_dropout, Is_causal>(params, stream);
        #endif
    });
}
}