#include "group_norm_act.h"

namespace at{
namespace native{

std::tuple<Tensor,Tensor,Tensor>fuse_gn_silu_cuda(
  const Tensor& self,
  const Tensor& weight,
  const Tensor& bias,
  int64_t num_groups,
  int64_t num_channels,
  double eps,
  bool training) {
  Tensor ret = at::empty_like(self),mean,rstd;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16, 
    self.scalar_type(),
    "fuse_gn_silu_cuda",
    [&] {
      SILU<scalar_t> siluop;
      groupnorm_act_forward_impl<scalar_t,SILU<scalar_t>>(self,weight,bias,num_groups,num_channels,eps,training,siluop,ret,mean,rstd);
    });
  return std::tuple<Tensor,Tensor,Tensor>(ret,mean,rstd);
}  

std::tuple<Tensor,Tensor,Tensor>fuse_gn_silu_backward_cuda(
  const Tensor& grad,
  const Tensor& X,
  const Tensor& mean,
  const Tensor& rstd,
  const Tensor& gamma,
  const Tensor& beta) { 
  Tensor dX = at::empty_like(X),dgamma = at::empty_like(gamma),dbeta = at::empty_like(gamma);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    grad.scalar_type(),
    "fuse_gn_silu_backward_cuda",
    [&] {   
      SILU<scalar_t> siluop; 
      return groupnorm_act_backward_impl<scalar_t,SILU<scalar_t>>(grad,X,mean,rstd,gamma,beta,siluop,dX,dgamma,dbeta);
  });
  return std::tuple<Tensor,Tensor,Tensor>(dX,dgamma,dbeta);
}
}
}

