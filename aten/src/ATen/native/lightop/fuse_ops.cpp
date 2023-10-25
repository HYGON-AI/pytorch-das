#include <ATen/native/lightop/fuse_ops.h>
#include <ATen/ops/native_fuse_layer_norm.h>
#include <ATen/ops/fuse_layer_norm_native.h>
#include <ATen/ops/native_fuse_layer_norm_native.h>
#include <array>
#include <tuple>
#include <vector>

namespace at {
namespace native {

Tensor fuse_layer_norm_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape, const c10::optional<Tensor>& weight_opt /* optional */, const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  return std::get<0>(at::native_fuse_layer_norm_symint(input, normalized_shape, weight, bias, eps));
}


DEFINE_DISPATCH(FuseLayerNormKernel);
DEFINE_DISPATCH(FuseLayerNormBackwardKernel);
} // namespace native
} // namespace at
