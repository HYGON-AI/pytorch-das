/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

namespace pytorch_flash {

struct Dropout {

    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                              const uint8_t p_dropout_in_uint8_t,
                              const int bid, const int hid, const int tid, const int nheads)
            : seed(seed)
            , offset(offset + (bid * nheads + hid) * 32)
            , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {
    }

    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), pytorch_flash::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            if constexpr(encode_dropout_in_sign_bit) {
                return keep ? val : -val;
            } else {
                return keep ? val : T(0);
            }
        };

        #if 1
        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            #pragma unroll
            for (int n = 0; n < size<2>(tensor); ++n, ++rowcol.y) {
                uint4 random_uint4 = pytorch_flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                // 16位类型的特殊实现：我们将阈值复制到32位值的低16位和高16位，然后使用f16x2比较指令来获取掩码。
                // 掩码的低16位将是0xffff或0x0000，高16位也将是0xffff或0x0000，这取决于随机值是否小于阈值。
                // 然后，我们在掩码和原始32位值之间进行位与运算。
                // 我们利用了浮点比较等同于整数比较的事实，因为我们比较的是其最高8位为零的无符号整数。
                #if 1
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    tensor(i, m, n) = encode_dropout(rnd_8[i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                }
                Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n));
                // if (cute::thread0()) { printf("pos2: tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                #else

                
                #endif

                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
                // }
            }
        }
        #endif
    }

    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout_continuous(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), tensor_.layout());
        
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            if constexpr(encode_dropout_in_sign_bit) {
                return keep ? val : -val;
            } else {
                return keep ? val : T(0);
            }
        };
        const int lane_id = threadIdx.x % 64;
        const int col_idx_offset = block_col_start + (lane_id / 16) * 4;
        const int stride_between_each_repeat = 16;
        const int stride_between_each_thread = 1;

        for (int i = 0; i < size<1>(tensor); ++i)
        {
            const int row_idx_base = block_row_start + i * block_row_stride;
            const int row_idx = row_idx_base;
            for (int j = 0; j < size<2>(tensor); ++j)
            {
                const int col_idx_base = col_idx_offset + j * stride_between_each_repeat;
                for (int mi = 0; mi < size<0>(tensor); ++mi)
                {
                    const int col_idx = col_idx_base + mi * stride_between_each_thread;
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
                    // {
                    //     printf("tidx = %d row_idx = %d col_idx = %d offset = %d\n", threadIdx.x, row_idx, col_idx, offset);
                    // }
                    uint2 rowcol = make_uint2(row_idx, col_idx);
                    uint4 random_uint4 = pytorch_flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                    uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                    tensor(mi, i, j) = 
                        encode_dropout(rnd_8[0] <= p_dropout_in_uint8_t, tensor(mi, i, j));
                }
            }
        }
    }


    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout_trans(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), pytorch_flash::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            if constexpr(encode_dropout_in_sign_bit) {
                return keep ? val : -val;
            } else {
                return keep ? val : T(0);
            }
        };
        const int lane_id = threadIdx.x % 64;
        const int col_idx_offset = block_col_start + (lane_id / 16) * 4;
        const int stride_between_each_repeat = 16;
        const int stride_between_each_thread = 1;
        for (int i = 0; i < size<1>(tensor); ++i)
        {
            const int row_idx_base = block_row_start + i * block_row_stride;
            const int row_idx = row_idx_base;
            for (int j = 0; j < size<2>(tensor); ++j)
            {
                const int col_idx_base = col_idx_offset + j * stride_between_each_repeat;
                for (int mi = 0; mi < size<0>(tensor); ++mi)
                {
                    const int col_idx = col_idx_base + mi * stride_between_each_thread;
                    uint2 rowcol = make_uint2(col_idx, row_idx);
                    uint4 random_uint4 = pytorch_flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                    uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                    tensor(mi, i, j) = 
                        encode_dropout(rnd_8[0] <= p_dropout_in_uint8_t, tensor(mi, i, j));
                }
            }
        }
    }





};

} // namespace pytorch_flash
