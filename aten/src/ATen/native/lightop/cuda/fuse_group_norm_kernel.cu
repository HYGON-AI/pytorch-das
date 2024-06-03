#include "act_group_norm.h"
int get_groupnorm_blocksize(int cols)
{
  int blocksize;
  if(cols>256)blocksize=512;
  else if(cols>128)blocksize=256;
  else if(cols>64)blocksize=128;
  else blocksize=64;
  return blocksize;
}

namespace at{
namespace native{

std::tuple<Tensor,Tensor,Tensor>fuse_group_norm_cuda(
  const Tensor& self,
  const Tensor& weight,
  const Tensor& bias,
  int64_t num_groups,
  int64_t num_channels,
  double eps,
  bool training)
{
  Tensor ret = at::empty_like(self),mean,rstd;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16, 
    self.scalar_type(),
    "fuse_group_norm_cuda",
    [&] {
      PassThrough<scalar_t> passop;
      act_groupnorm_forward_impl<scalar_t,PassThrough<scalar_t>>(self,weight,bias,num_groups,num_channels,eps,training,passop,ret,mean,rstd);
    });
  return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}  

std::tuple<Tensor,Tensor,Tensor>fuse_group_norm_backward_cuda(
  const Tensor& grad,
  const Tensor& X,
  const Tensor& mean,
  const Tensor& rstd,
  const Tensor& gamma) {
  Tensor dX = at::empty_like(X),dgamma = at::empty_like(gamma),dbeta = at::empty_like(gamma);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "fuse_group_norm_backward_cuda",
    [&] {
      PassThrough<scalar_t> passop; 
      act_groupnorm_backward_impl<scalar_t,PassThrough<scalar_t>>(grad,X,mean,rstd,gamma,passop,dX,dgamma,dbeta);
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}
}
}

