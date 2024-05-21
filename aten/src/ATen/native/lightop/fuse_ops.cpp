#include <ATen/native/lightop/fuse_ops.h>
#include <ATen/ops/native_fuse_layer_norm.h>
#include <ATen/ops/fuse_layer_norm_native.h>
#include <ATen/ops/native_fuse_layer_norm_native.h>
#include <ATen/ops/native_fuse_bias_gelu.h>
#include <ATen/ops/native_fuse_bias_gelu_native.h>
#include <ATen/ops/native_fuse_bias_swiglu.h>
#include <ATen/ops/native_fuse_bias_swiglu_native.h>
#include <ATen/ops/native_fuse_ln_dropout_native.h>
#include <ATen/ops/native_fuse_ln_dropout.h>
#include <ATen/ops/native_fuse_ln_add_dropout_native.h>
#include <ATen/ops/native_fuse_ln_add_dropout.h>
#include <ATen/ops/native_fuse_ln_add_dropout_bias_native.h>
#include <ATen/ops/native_fuse_ln_add_dropout_bias.h>
#include <ATen/ops/native_fuse_bias_dropout_add_native.h>
#include <ATen/ops/native_fuse_bias_dropout_add.h>
#include <ATen/ops/native_fuse_dropout_add_native.h>
#include <ATen/ops/native_fuse_dropout_add.h>
#include <array>
#include <tuple>
#include <vector>

namespace at {
namespace native {

Tensor fuse_layer_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool training,
    bool /* cudnn_enable, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  return std::get<0>(at::native_fuse_layer_norm(input, weight, bias, eps, training));
}

Tensor fuse_ln_dropout(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    double p,
    bool training,
    bool /* cudnn_enable, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  return std::get<0>(at::native_fuse_ln_dropout(input, weight, bias, eps, p, training));
}

Tensor fuse_ln_add_dropout(
    const Tensor& input,
    const Tensor& other,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    double p,
    bool training,
    bool /* cudnn_enable, deprecated */) {
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;
    return std::get<0>(at::native_fuse_ln_add_dropout(input, other, weight, bias, eps, p, training));
}

Tensor fuse_ln_add_dropout_bias(
    const Tensor& input,
    const Tensor& other,
    const Tensor& attention_bias,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    double p,
    bool training,
    bool /* cudnn_enable, deprecated */) {
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias = *bias_maybe_owned;
    return std::get<0>(at::native_fuse_ln_add_dropout_bias(input, other, attention_bias, weight, bias, eps, p, training));
}

Tensor fuse_bias_dropout_add(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& other,
    double p,
    bool training) {
  return std::get<0>(at::native_fuse_bias_dropout_add(input, bias, other, p, training));
}

Tensor fuse_dropout_add(
    const Tensor& input,
    const Tensor& other,
    double p,
    bool training) {
  return std::get<0>(at::native_fuse_dropout_add(input, other, p, training));
}

DEFINE_DISPATCH(FuseLayerNormKernel);
DEFINE_DISPATCH(FuseLayerNormBackwardKernel);

Tensor fuse_bias_gelu(
  const Tensor& input,
  const Tensor& bias
)
{
  return at::native_fuse_bias_gelu(input, bias);
}
Tensor fuse_bias_swiglu(
  const Tensor& input,
  const Tensor& bias
)
{
  return at::native_fuse_bias_swiglu(input, bias);
}

} // namespace native
} // namespace at
