#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

namespace at::native {

#define MAX_BLOCK 512
#define MAX_BLOCK8 4096

bool check_exchange_last_dim(const c10::IntArrayRef& strides,const c10::IntArrayRef& sizes,int64_t n)
{
  int64_t a=1;
  int64_t length=sizes.size()-1,stridelast=-1;
  for(int64_t i=length;i>=0;i--){
    if(i!=length&&i!=n){
      if(strides[i]!=a)return false;
      a*=sizes[i];
    }
    else if(i==n){
      if(strides[i]!=1)return false;
      else stridelast=a;
      a*=sizes[length];
    }
    else if(i==length){
      a*=sizes[n];
    }

  }
  if(stridelast!=strides[length])return false;
  return true;
}

//when M<=128,very fast.example1: M=64,N=4000,0.011ms  example2: M=64,N=4000,0.007ms
__launch_bounds__(MAX_BLOCK)
__global__ void and_kernel_continuous_small(bool* result, bool* self,int64_t M,int64_t N,int64_t lines_per_block){
  int64_t tid=threadIdx.x;
  int64_t idx1=tid/M;
  int64_t idx2=tid%M;
  int64_t line=blockIdx.x*lines_per_block+idx1;
  extern __shared__ bool sresult[];
  
  if(tid>=lines_per_block*M || line>=N)return;
  if(idx2==0)sresult[idx1]=true;
  __syncthreads();
  
  if(!self[line*M+idx2]){
    sresult[idx1]=false;
  }
  __syncthreads();
  if(idx2==0){
    result[line]=sresult[idx1];
  }
}


__launch_bounds__(MAX_BLOCK)
__global__ void and_kernel_continuous_middle(bool* result, bool* self,int64_t M){
  int64_t idx=threadIdx.x;
  int64_t idx8=idx*8;
  if(idx8>=M)return;
  int64_t base_idx=blockIdx.x*M;
  __shared__ bool sresult;
  if(threadIdx.x==0)sresult=true;
  __syncthreads();
  bool* t=self+base_idx+idx8;
  if(idx8+8>M)t-=idx8+8-M;
  if(*(int64_t*)t!=0x0101010101010101){
    sresult=false;
  }
  __syncthreads();
  if(idx==0){
    result[blockIdx.x]=sresult;
  }
}

__launch_bounds__(MAX_BLOCK)
__global__ void and_kernel_continuous_big(bool* result, bool* self,int64_t M){
  int64_t idx=threadIdx.x;
  int64_t idx8=idx*8;
  if(idx8>=M)return;
  int64_t base_idx=blockIdx.x*M;
  __shared__ bool sresult;
  if(threadIdx.x==0)sresult=true;
  __syncthreads();
  for(int64_t i=idx8;i<M;i+=MAX_BLOCK8){
    bool* t=self+base_idx+i;
    if(i+8>M)t-=i+8-M;
    if(*(int64_t*)t!=0x0101010101010101){
      sresult=false;
    }
  }
  __syncthreads();
  if(idx==0){
    result[blockIdx.x]=sresult;
  }
}

//when M<=128,very fast.example1: M=64,N=4000,0.011ms  example2: M=64,N=4000,0.007ms
__launch_bounds__(MAX_BLOCK)
__global__ void any_kernel_continuous_small(bool* result, bool* self,int64_t M,int64_t N,int64_t lines_per_block){
  int64_t tid=threadIdx.x;
  int64_t idx1=tid/M;
  int64_t idx2=tid%M;
  int64_t line=blockIdx.x*lines_per_block+idx1;
  extern __shared__ bool sresult[];
  
  if(tid>=lines_per_block*M || line>=N)return;
  if(idx2==0)sresult[idx1]=false;
  __syncthreads();
  
  if(self[line*M+idx2]){
    sresult[idx1]=true;
  }
  __syncthreads();
  if(idx2==0){
    result[line]=sresult[idx1];
  }
}


__launch_bounds__(MAX_BLOCK)
__global__ void any_kernel_continuous_middle(bool* result, bool* self,int64_t M){
  int64_t idx=threadIdx.x;
  int64_t idx8=idx*8;
  if(idx8>=M)return;
  int64_t base_idx=blockIdx.x*M;
  __shared__ bool sresult;
  if(threadIdx.x==0)sresult=false;
  __syncthreads();
  bool* t=self+base_idx+idx8;
  if(idx8+8>M)t-=idx8+8-M;
  if(*(int64_t*)t){
    sresult=true;
  }
  __syncthreads();
  if(idx==0){
    result[blockIdx.x]=sresult;
  }
}

__launch_bounds__(MAX_BLOCK)
__global__ void any_kernel_continuous_big(bool* result, bool* self,int64_t M){
  int64_t idx=threadIdx.x;
  int64_t idx8=idx*8;
  if(idx8>=M)return;
  int64_t base_idx=blockIdx.x*M;
  __shared__ bool sresult;
  if(threadIdx.x==0)sresult=false;
  __syncthreads();
  for(int64_t i=idx8;i<M;i+=MAX_BLOCK8){
    bool* t=self+base_idx+i;
    if(i+8>M)t-=i+8-M;
    if(*(int64_t*)t){
      sresult=true;
    }
  }
  __syncthreads();
  if(idx==0){
    result[blockIdx.x]=sresult;
  }
}


void and_kernel_cuda(TensorIterator& iter) {
    if(kBool==iter.dtype(1)){
    const auto & output = iter.tensor(0);
    const auto & input = iter.tensor(1);
    const auto & strides_i=input.strides();
    const auto & sizes_i=input.sizes();
    const auto & sizes_o=output.sizes();
    const int64_t ndim=sizes_i.size();
    int64_t dim=-1;
    for(int64_t i=0;i<ndim;i++){
      if(sizes_i[i]!=sizes_o[i]){
        dim=i;
        break;
      }
    }
    if(dim!=-1&&check_exchange_last_dim(strides_i,sizes_i,dim)){
      int64_t N=output.numel();
      int64_t M=input.numel()/N;
      bool* output_data = output.data_ptr<bool>();
      bool* input_data = (bool *)input.data_ptr();
      if(M>4096)and_kernel_continuous_big<<<N, MAX_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M);
      else if(M>128){
        int64_t blocksize=((M/8-1)/64+1)*64;
        and_kernel_continuous_middle<<<N, blocksize, 0, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M);
      }
      else{
        int64_t lines_per_block=MAX_BLOCK/M;
        int64_t grid = (N-1)/lines_per_block+1;
        and_kernel_continuous_small<<<grid, MAX_BLOCK, lines_per_block, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M,N,lines_per_block);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return ;
    }
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc && static_cast<bool>(val));
            }),
            true);
      });
}

void or_kernel_cuda(TensorIterator& iter) {
  if(kBool==iter.dtype(1)){
    const auto & output = iter.tensor(0);
    const auto & input = iter.tensor(1);
    const auto & strides_i=input.strides();
    const auto & sizes_i=input.sizes();
    const auto & sizes_o=output.sizes();
    const int64_t ndim=sizes_i.size();
    int64_t dim=-1;
    for(int64_t i=0;i<ndim;i++){
      if(sizes_i[i]!=sizes_o[i]){
        dim=i;
        break;
      }
    }
    if(dim!=-1&&check_exchange_last_dim(strides_i,sizes_i,dim)){
      int64_t N=output.numel();
      int64_t M=input.numel()/N;
      bool* output_data = output.data_ptr<bool>();
      bool* input_data = (bool *)input.data_ptr();
      if(M>4096)any_kernel_continuous_big<<<N, MAX_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M);
      else if(M>128){
        int64_t blocksize=((M/8-1)/64+1)*64;
        any_kernel_continuous_middle<<<N, blocksize, 0, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M);
      }
      else{
        int64_t lines_per_block=MAX_BLOCK/M;
        int64_t grid = (N-1)/lines_per_block+1;
        any_kernel_continuous_small<<<grid, MAX_BLOCK, lines_per_block, at::cuda::getCurrentCUDAStream()>>>
                  (output_data,input_data,M,N,lines_per_block);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return ;
    }
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc || static_cast<bool>(val));
            }),
            false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda)
REGISTER_DISPATCH(or_stub, &or_kernel_cuda)

} // namespace at::native
