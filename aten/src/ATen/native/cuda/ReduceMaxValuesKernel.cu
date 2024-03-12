#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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
#include <ATen/native/cuda/Reduce.cuh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/core/Tensor.h>
namespace at::native {


#define VEC 4

template <typename T>
struct ValueIndex{
  T value;
  int index;
};

template <typename T>
__inline__ __device__ ValueIndex<T> WarpReduceMax(ValueIndex<T> val,int count) {
#pragma unroll
  for (int offset = count; offset > 0; offset >>= 1) {
    ValueIndex<T> val2;
    val2.value=WARP_SHFL_DOWN(val.value, offset);
    val2.index=__shfl_down(val.index, offset);
    if(val2.value>val.value)val=val2;
  }
  return val;
}


template <typename T>
__inline__ __device__ ValueIndex<T> BlockReduceMax(ValueIndex<T> val, ValueIndex<T>* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceMax(val,32);// C10_WARP_SIZE/2
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  if (wid == 0) {
    val= shared[lid];
    val = WarpReduceMax(val,blockDim.x / 128);// C10_WARP_SIZE*2
  }
  return val;
}

template <typename scalar_t>
__launch_bounds__(1024)
__global__ void max_not_contiguous_kernel(scalar_t* result, int64_t* indice, scalar_t* self,int n){
  __shared__ ValueIndex<scalar_t> sresult[16];// 1024/C10_WARP_SIZE
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using LoadI = memory::aligned_vector<int64_t, VEC>;
  ValueIndex<scalar_t> tmax[VEC];
  scalar_t self_vec[VEC];
  scalar_t tvalue[VEC];
  int64_t tindex[VEC];
  int index=threadIdx.x;
  int base_index=blockIdx.x*VEC+threadIdx.x*gridDim.x*VEC;
  int offset=gridDim.x*blockDim.x*VEC;
  *(LoadT*)self_vec=*(LoadT*)(self+base_index);

  #pragma unroll
  for(int i=0;i<VEC;i++){
    tmax[i].index=threadIdx.x;
    tmax[i].value=self_vec[i];
  }

  for(int i=base_index+offset;i<n;i+=offset){
    index+=blockDim.x;
    *(LoadT*)tvalue=*(LoadT*)(self+i);
    #pragma unroll
    for(int k=0;k<VEC;k++){
      if(tvalue[k]>tmax[k].value){
        tmax[k].value=tvalue[k];
        tmax[k].index=index;
      }
    }
  }

  #pragma unroll
  for(int i=0;i<VEC;i++){
    tmax[i]=BlockReduceMax(tmax[i],sresult);
    tvalue[i]=tmax[i].value;
    tindex[i]=tmax[i].index;
  }

  if(threadIdx.x==0){
    *(LoadT*)(result+blockIdx.x*VEC)=*(LoadT*)(tvalue);
    *(LoadI*)(indice+blockIdx.x*VEC)=*(LoadI*)(tindex);
  }
}

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter,
    func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::lower_bound());
}

void max_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cuda", [&]() {
        max_values_kernel_cuda_impl<scalar_t>(iter);
      });
}

void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "max_cuda", [&]() {
        const auto & result = iter.tensor(0);
        const auto & indice = iter.tensor(1);
        const auto & self = iter.tensor(2);
        const auto & sizes_i=self.sizes();
        const auto & sizes_o=result.sizes();
        const int ndim=sizes_i.size();
        int dim=-1;
        for(int i=0;i<ndim;i++){
          if(sizes_i[i]!=sizes_o[i]){
            dim=i;
            break;
          }
        }
        if(dim!=-1&&self.dim()==2&&self.sizes()[dim]>=1024&&self.strides()[1-dim]==1&&self.strides()[dim]%VEC==0){
          int grid=self.strides()[dim]/VEC;
          int block=1024;
          scalar_t* result_data = result.data_ptr<scalar_t>();
          int64_t* indice_data = indice.data_ptr<int64_t>();
          scalar_t* self_data = self.data_ptr<scalar_t>();
          max_not_contiguous_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                    result_data,indice_data,self_data,self.numel());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else{
          gpu_reduce_kernel<scalar_t, scalar_t>(
              iter,
              MaxOps<scalar_t>{},
              thrust::pair<scalar_t, int64_t>(
                  at::numeric_limits<scalar_t>::lower_bound(), 0));
        }
      });
}

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_all_cuda", [&] {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);

} // namespace at::native
