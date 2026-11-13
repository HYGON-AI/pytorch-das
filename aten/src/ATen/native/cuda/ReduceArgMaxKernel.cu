#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/native/cuda/Reduce.cuh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>

#include <thrust/pair.h>

namespace at::native {

#if defined(USE_ROCM)
namespace {

constexpr int ARGMAX_SMALL_K_MAX_K = 64;
constexpr int64_t ARGMAX_SMALL_K_MIN_ROWS = 1024;
constexpr int64_t ARGMAX_SMALL_K_LARGE_K_MIN_ROWS = 65536;

template <typename scalar_t, typename acc_t, int Threads, int TileK>
C10_LAUNCH_BOUNDS_1(Threads)
__global__ void argmax_small_k_coalesced(
    const scalar_t* input,
    int64_t* output,
    int64_t rows,
    int k) {
  __shared__ scalar_t tile[Threads * TileK];
  const int64_t block_row =
      static_cast<int64_t>(blockIdx.x) * Threads;
  const int64_t rows_in_block =
      rows - block_row < Threads
      ? rows - block_row
      : Threads;
  const int block_elements = static_cast<int>(rows_in_block) * k;
  const int64_t block_input_offset = block_row * k;

  for (int offset = threadIdx.x;
       offset < block_elements;
       offset += Threads) {
    tile[offset] = input[block_input_offset + offset];
  }
  __syncthreads();

  const int64_t row = block_row + threadIdx.x;
  if (row >= rows) {
    return;
  }
  acc_t max_value = at::numeric_limits<acc_t>::lower_bound();
  int max_index = 0;
  const int tile_row_offset = threadIdx.x * k;
#pragma unroll
  for (int index = 0; index < TileK; ++index) {
    if (index >= k) {
      continue;
    }
    const acc_t value =
        static_cast<acc_t>(tile[tile_row_offset + index]);
    const bool max_is_nan = at::_isnan(max_value);
    const bool value_is_nan = at::_isnan(value);
    const bool take_other =
        (value_is_nan && !max_is_nan) ||
        (!value_is_nan && !max_is_nan && value > max_value);
    if (take_other) {
      max_value = value;
      max_index = index;
    }
  }
  output[row] = max_index;
}

bool should_use_argmax_small_k(int64_t rows, int64_t k) {
  if (k < 2 || k > ARGMAX_SMALL_K_MAX_K) {
    return false;
  }
  return rows >= (k <= 32
      ? ARGMAX_SMALL_K_MIN_ROWS
      : ARGMAX_SMALL_K_LARGE_K_MIN_ROWS);
}

template <typename scalar_t>
bool can_use_argmax_small_k_fast_path(const TensorIterator& iter) {
  if (iter.noutputs() != 1 ||
      iter.ntensors() != 2 ||
      iter.dtype(0) != kLong ||
      iter.ndim() != 2 ||
      iter.num_reduce_dims() != 1) {
    return false;
  }

  const auto shape = iter.shape();
  const auto output_strides = iter.strides(0);
  const auto input_strides = iter.strides(1);
  const int64_t k = shape[0];
  const int64_t rows = shape[1];
  return should_use_argmax_small_k(rows, k) &&
      iter.num_output_elements() == rows &&
      iter.numel() == rows * k &&
      output_strides[0] == 0 &&
      output_strides[1] == static_cast<int64_t>(sizeof(int64_t)) &&
      input_strides[0] == static_cast<int64_t>(sizeof(scalar_t)) &&
      input_strides[1] == static_cast<int64_t>(k * sizeof(scalar_t));
}

template <typename scalar_t, typename acc_t, int Threads>
bool try_launch_argmax_small_k(TensorIterator& iter) {
  if (!can_use_argmax_small_k_fast_path<scalar_t>(iter)) {
    return false;
  }

  const auto shape = iter.shape();
  const int k = static_cast<int>(shape[0]);
  const int64_t rows = shape[1];
  const int64_t blocks = (rows + Threads - 1) / Threads;
  const int tile_k = k <= 8 ? 8 : k <= 16 ? 16 : k <= 32 ? 32 : 64;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const size_t shared_memory_bytes =
      static_cast<size_t>(Threads) * tile_k * sizeof(scalar_t);
  if (Threads > properties->maxThreadsPerBlock ||
      shared_memory_bytes > properties->sharedMemPerBlock ||
      blocks > properties->maxGridSize[0]) {
    return false;
  }

  const scalar_t* input =
      static_cast<const scalar_t*>(iter.data_ptr(1));
  int64_t* output = static_cast<int64_t*>(iter.data_ptr(0));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (k <= 8) {
    argmax_small_k_coalesced<scalar_t, acc_t, Threads, 8>
        <<<static_cast<int>(blocks), Threads, 0, stream>>>(
            input, output, rows, k);
  } else if (k <= 16) {
    argmax_small_k_coalesced<scalar_t, acc_t, Threads, 16>
        <<<static_cast<int>(blocks), Threads, 0, stream>>>(
            input, output, rows, k);
  } else if (k <= 32) {
    argmax_small_k_coalesced<scalar_t, acc_t, Threads, 32>
        <<<static_cast<int>(blocks), Threads, 0, stream>>>(
            input, output, rows, k);
  } else {
    argmax_small_k_coalesced<scalar_t, acc_t, Threads, 64>
        <<<static_cast<int>(blocks), Threads, 0, stream>>>(
            input, output, rows, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool try_launch_argmax_small_k(TensorIterator& iter) {
  switch (iter.dtype(1)) {
    case kHalf:
      return try_launch_argmax_small_k<at::Half, float, 256>(iter);
    case kBFloat16:
      return try_launch_argmax_small_k<at::BFloat16, float, 256>(iter);
    case kFloat:
      return try_launch_argmax_small_k<float, float, 128>(iter);
    case kDouble:
      return try_launch_argmax_small_k<double, double, 64>(iter);
    default:
      return false;
  }
}

} // namespace
#endif

template <typename scalar_t, typename acc_t = scalar_t>
void argmax_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMaxOps<acc_t>{},
      thrust::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::lower_bound(), 0));
};

void argmax_kernel_cuda(TensorIterator& iter) {
#if defined(USE_ROCM)
  if (try_launch_argmax_small_k(iter)) {
    return;
  }
#endif

  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in
  // float.
  if (iter.dtype(1) == kHalf) {
    argmax_kernel_cuda_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmax_kernel_cuda_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cuda", [&]() {
      argmax_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda)

} // namespace at::native
