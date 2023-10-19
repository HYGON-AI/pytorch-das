#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/layer_norm.h>

#include <type_traits>

#include <thrust/tuple.h>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/thread_constants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/native_fuse_layer_norm_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/env.h>


namespace at::native {

namespace{

constexpr int kCUDANumThreads = 256;

template<typename T, typename T_ACC> __global__
void cuComputeGradInput(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int64_t M,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    const T* gamma,
    T* grad_input)
{
  for (int i1=blockIdx.y; i1 < M; i1 += gridDim.y) {
    T_ACC sum_loss1 = T_ACC(0);
    T_ACC sum_loss2 = T_ACC(0);
    T_ACC c_mean = mean[i1];
    const T_ACC c_rstd = rstd[i1];
    const T* k_input = input + i1*N;
    const T* k_dout = dout + i1*N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      // Optimization for ROCm MI100
      for( int l = 0; l < N ; l += numx) {
        int idx = l + thrx;
        const T_ACC gamma_idx = static_cast<T_ACC>((idx<N) ? gamma[idx] : T(0));
        const T_ACC c_h = static_cast<T_ACC>((idx<N) ? k_input[idx] : T(0));
        const T_ACC c_loss = static_cast<T_ACC>((idx<N) ? k_dout[idx] : T(0));
        sum_loss1 += c_loss * gamma_idx;
        sum_loss2 += c_loss * gamma_idx * (c_h - c_mean) * c_rstd;
      }
    } else {
      for( int l = 0; l < N ; l += numx) {
        int idx = l + thrx;
        const T_ACC c_h = static_cast<T_ACC>((idx<N) ? k_input[idx] : T(0));
        const T_ACC c_loss = static_cast<T_ACC>((idx<N) ? k_dout[idx] : T(0));
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_rstd;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      alignas(sizeof(double)) extern __shared__ char shared[];
      T_ACC * buf = reinterpret_cast<T_ACC*>(&shared);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    T_ACC fH = (T_ACC)N;
    T_ACC term1 = (T_ACC(1) / fH) * c_rstd;
    T* k_grad_input = grad_input + i1*N;
    if (gamma != NULL) {
      for (int l = thrx;  l < N;  l+=numx) {
        const T_ACC c_h = static_cast<T_ACC>(k_input[l]);
        const T_ACC c_loss = static_cast<T_ACC>(k_dout[l]);
        T_ACC f_grad_input = fH * c_loss * gamma[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < N;  l+=numx) {
        const T_ACC c_h = static_cast<T_ACC>(k_input[l]);
        const T_ACC c_loss = static_cast<T_ACC>(k_dout[l]);
        T_ACC f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardSimpleCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dg == nullptr ? T_ACC(0)
                            : static_cast<T_ACC>(dY[index]) *
              (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
              static_cast<T_ACC>(rstd[i]);
      sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]);
    }
    if (dg != nullptr) {
      dg[j] = sum1;
    }
    if (db != nullptr) {
      db[j] = sum2;
    }
  }
}
template<typename T, typename T_ACC> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    T_ACC* warp_buf1,
    T_ACC* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd)
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    T curr_mean = mean[i1];
    T curr_rstd = rstd[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*N+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<N) {
        T curr_input = static_cast<T>(input[load_idx]);
        T curr_dout = static_cast<T>(dout[load_idx]);
        warp_buf1[write_idx] = curr_dout;
        warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_rstd;
      } else {
        warp_buf1[write_idx] = T(0);
        warp_buf2[write_idx] = T(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      warp_buf1[write_idx] = T(0);
      warp_buf2[write_idx] = T(0);
    }
  }
}

template<typename T, typename T_ACC> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    T_ACC* warp_buf1,
    T_ACC* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd)
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    T_ACC curr_mean = mean[i1];
    T_ACC curr_rstd = rstd[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*N+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<N) {
        T_ACC curr_input = static_cast<T_ACC>(input[load_idx]);
        T_ACC curr_dout = static_cast<T_ACC>(dout[load_idx]);
        warp_buf1[write_idx] += curr_dout;
        warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_rstd;
      }
    }
  }
}
template<typename T, typename T_ACC> __global__
void cuComputePartGradGammaBeta(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int64_t M,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    T_ACC* part_grad_gamma,
    T_ACC* part_grad_beta)
{
    const int numsegs_M = (M+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_M + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < M ? i1_beg_plus_one : M;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    alignas(sizeof(double)) extern __shared__ char shared[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&shared); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    T_ACC* warp_buf1 = (T_ACC*)buf;
    T_ACC* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,N,mean,rstd);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,N,mean,rstd);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    T_ACC acc1 = T_ACC(0);
    T_ACC acc2 = T_ACC(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      acc1 += warp_buf1[idx1];
      acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        warp_buf1[idx1] += warp_buf1[idx2];
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < N) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      part_grad_beta[blockIdx.y*N+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      part_grad_gamma[blockIdx.y*N+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}
template<typename T, typename T_ACC> __global__
void cuComputeGradGammaBeta(
    const T_ACC* part_grad_gamma,
    const T_ACC* part_grad_beta,
    const int part_size,
    const int64_t M,
    const int64_t N,
    T* grad_gamma,
    T* grad_beta)
{
    // sum partial gradients for gamma and beta
    alignas(sizeof(double)) extern __shared__ char shared[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&shared);
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < N) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      T_ACC sum_gamma = T_ACC(0);
      T_ACC sum_beta = T_ACC(0);
      const T_ACC* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * N + i2;
      const T_ACC* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * N + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*N];
        sum_beta += part_grad_beta_ptr[warp_offset*N];
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;
          buf[write_idx+nbsize3] = sum_beta;
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];
          sum_beta += buf[read_idx+nbsize3];
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        grad_beta[i2] = sum_beta;
      }
    }
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val,int max=32) {
  for (int offset = max; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  const int block_size=blockDim.x;
  const int share_size=block_size/C10_WARP_SIZE;
  val = WarpReduceSum(val);
  if(block_size==C10_WARP_SIZE)return val;
  __syncthreads();
  if (lid == 0&&wid<share_size) {
    shared[wid] = val;
  }
  __syncthreads();
  if (wid == 0&&lid<share_size) {
    val= shared[lid];
    val = WarpReduceSum<T>(val,share_size/2);
  }
  return val;
}

template <typename scalar_t,typename T_ACC>
__global__ void layer_norm_grad_input_kernel(int N,const scalar_t* dY,const scalar_t* X,const scalar_t* gamma,const T_ACC* mean,
                                         const T_ACC* rstd, scalar_t* dX)
{                                             
  __shared__ T_ACC ds_shared[8];                                                         
  __shared__ T_ACC db_shared[8];                                                         
 const int i = blockIdx.x;                                                               
  T_ACC sum1 = 0;                                                                        
  T_ACC sum2 = 0;                                                                        
  for (int j = threadIdx.x; j < N; j += blockDim.x) {                                    
    const int index = i * N + j;                                                         
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);                                  
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;      
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;                                     
  }                                                                                      
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);                                         
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);                                         
  __shared__ T_ACC b;                                                                    
  __shared__ T_ACC c;                                                                    
  if (threadIdx.x == 0) {                                                                
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);                                    
    T_ACC tmean = mean[i];                                           
    T_ACC trstd = rstd[i];                                           
    const T_ACC a = (sum2 * tmean - sum1) * trstd * trstd * trstd * s;                   
    b = a;                                                                               
    c = -a * tmean - sum2 * trstd * s;                                                   
  }                                                                                      
  __syncthreads();
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
    const T_ACC gamma_v = static_cast<T_ACC>(gamma[j]);
    dX[index] = static_cast<T_ACC>(rstd[i]) * static_cast<T_ACC>(dY[index]) * gamma_v + b * static_cast<T_ACC>(X[index]) + c;
  }
}


int get_layernorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else blocksize=128;
  return blocksize;
}
template <typename T>
void FuseLayerNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  using T_ACC = acc_type<T, true>;
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(mean.numel() == M);
  TORCH_CHECK(rstd.numel() == M);
  TORCH_CHECK(M <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0], "M should be less than maximum CUDA grid size, \
  file a support request to support bigger batches");
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "Normalized shape should have less than INT_MAX elements, \
  file a support request to support bigger normalized shapes");
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T_ACC* mean_data = mean.template data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.template data_ptr<T_ACC>();
  const T* gamma_data =
      gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int warp_size = at::cuda::warp_size();
  if (dX_data != nullptr) {
    if (M >= 32768) {
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
      const dim3 blocks1(1, std::min((uint64_t)M, maxGridY), 1);
      dim3 threads1(warp_size, 4, 1);
      threads1.y = 2; // Optimization for ROCm
      int nshared =
              threads1.y > 1 ?
              threads1.y*threads1.x*sizeof(T_ACC) :
              0;
      cuComputeGradInput<<<blocks1, threads1, nshared, cuda_stream>>>(
              dY_data,
              X_data,
              M, N,
              mean_data,
              rstd_data,
              gamma_data,
              dX_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      layer_norm_grad_input_kernel<T,T_ACC><<<M, get_layernorm_blocksize(N), 0, cuda_stream>>>(
                  N, dY_data, X_data,gamma_data,mean_data,rstd_data,dX_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data =
        dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;

    if (M < 128) {
      // For small batch size, do colwise reduce directly.
      const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T, T_ACC>
          <<<B, kCUDANumThreads, 0, cuda_stream>>>(
              M,
              N,
              dY_data,
              X_data,
              mean_data,
              rstd_data,
              dgamma_data,
              dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // For small batch size, do colwise reduce directly.
      const int part_size = warp_size;
      const dim3 threads2(warp_size, 4, 1);
      const dim3 blocks2((N + threads2.x - 1) / threads2.x, part_size, 1);
      const int nshared2_a = 2 * sizeof(T_ACC) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(T_ACC);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;

      const auto part_grad_dtype = at::toAccumulateType(X.scalar_type(), true);
      Tensor part_grad_gamma = at::empty({part_size,N}, gamma.options().dtype(part_grad_dtype));
      Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, cuda_stream>>>(
                      dY_data,
                      X_data,
                      M,N,
                      mean_data,
                      rstd_data,
                      part_grad_gamma.template data_ptr<T_ACC>(),
                      part_grad_beta.template data_ptr<T_ACC>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      const dim3 threads3(warp_size, 8, 1); // Optimization for ROCm
      const dim3 blocks3((N + threads2.x - 1) / threads2.x, 1, 1);
      const int nshared3 = threads3.x * threads3.y * sizeof(T);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, cuda_stream>>>(
                      part_grad_gamma.template data_ptr<T_ACC>(),
                      part_grad_beta.template data_ptr<T_ACC>(),
                      part_size,
                      M,N,
                      dgamma_data,
                      dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

void FuseLayerNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "FuseLayerNormBackwardKernelImpl",
      [&]() {
        FuseLayerNormBackwardKernelImplInternal<scalar_t>(
            dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

}//name space {}
void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd);
void FuseLayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd){
      LayerNormKernelImpl(X,gamma,beta,M,N,eps,Y,mean,rstd);
    }

std::tuple<Tensor, Tensor, Tensor> fuse_layer_norm_cuda(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  Tensor mean = at::empty({M}, X->options().dtype(acc_type));
  Tensor rstd = at::empty({M}, X->options().dtype(acc_type));
  // Calling the kernel for M==0 gives a CUDA error
  // See: https://github.com/pytorch/pytorch/pull/28614
  if (M > 0) {
    FuseLayerNormKernelImpl(*X, *gamma, *beta, M, N, eps, &Y, &mean, &rstd);
  }
  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  std::vector<int64_t> stat_shape;
  for (size_t idx = 0; idx < axis; ++idx) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (size_t idx = axis; idx < input.dim(); ++idx) {
    stat_shape.push_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);

  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> fuse_layer_norm_backward_cuda(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (1) {
    dX = at::native::empty_like(
        *X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (1) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (1) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (M > 0 && N > 0) {
    FuseLayerNormBackwardKernelImpl(
        dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

REGISTER_DISPATCH(FuseLayerNormKernel, &FuseLayerNormKernelImpl);

} // namespace at::native
