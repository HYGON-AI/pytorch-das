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
#include <ATen/ops/native_fuse_group_norm_native.h>
#include <ATen/ops/native_fuse_group_norm.h>
#include <ATen/ops/native_fuse_celu_gn_native.h>
#include <ATen/ops/native_fuse_celu_gn.h>
#include <ATen/ops/native_fuse_gn_silu_native.h>
#include <ATen/ops/native_fuse_gn_silu.h>
#include <ATen/ops/native_fuse_ctc_loss_native.h>
#include <ATen/ops/native_fuse_ctc_loss.h>
#include <ATen/ops/native_fuse_rmsnorm_native.h>
#include <ATen/ops/native_fuse_rmsnorm.h>
#include <ATen/ops/native_fuse_rn_dropout_native.h>
#include <ATen/ops/native_fuse_rn_dropout.h>
#include <ATen/ops/native_fuse_rn_add_dropout_native.h>
#include <ATen/ops/native_fuse_rn_add_dropout.h>
#include <ATen/ops/native_fuse_rn_add_dropout_bias_native.h>
#include <ATen/ops/native_fuse_rn_add_dropout_bias.h>
#include <array>
#include <tuple>
#include <vector>

namespace at {
namespace native {

Tensor fuse_layer_norm(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
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
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
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
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
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
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
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

Tensor fuse_celu_gn(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t num_groups,
    int64_t num_channels,
    double gn_alpha,
    double ln_epsilon,
    bool training,
    bool /* cudnn_enable, deprecated */) {
    return std::get<0>(at::native_fuse_celu_gn(input, weight, bias, num_groups, num_channels, gn_alpha, ln_epsilon, training));
}

Tensor fuse_group_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t num_groups,
    int64_t num_channels,
    double eps,
    bool training,
    bool /* cudnn_enable, deprecated */) {
    return std::get<0>(at::native_fuse_group_norm(input, weight, bias, num_groups, num_channels, eps, training));
}

Tensor fuse_gn_silu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t num_groups,
    int64_t num_channels,
    double ln_epsilon,
    bool training,
    bool /* cudnn_enable, deprecated */) {
    return std::get<0>(at::native_fuse_gn_silu(input, weight, bias, num_groups, num_channels, ln_epsilon,  training));
}

Tensor fuse_rmsnorm(
    const Tensor& self,
    const Tensor& weight,
    double ln_epsilon,
    bool training) {
    return std::get<0>(at::native_fuse_rmsnorm(self, weight, ln_epsilon, training));
}

Tensor fuse_rn_dropout(
    const Tensor& self,
    const Tensor& weight,
    double ln_epsilon,
    double p,
    bool training) {
    auto pself = self.expect_contiguous();
    auto pweight = weight.expect_contiguous();
    return std::get<0>(at::native_fuse_rn_dropout(*pself, *pweight, ln_epsilon, p, training));
}

Tensor fuse_rn_add_dropout(
    const Tensor& self,
    const Tensor& other,
    const Tensor& weight,
    double ln_epsilon,
    double p,
    bool training) {
    auto pself = self.expect_contiguous();
    auto pother = other.expect_contiguous();
    auto pweight = weight.expect_contiguous();
    return std::get<0>(at::native_fuse_rn_add_dropout(*pself, *pother, *pweight, ln_epsilon, p, training));
}

Tensor fuse_rn_add_dropout_bias(
    const Tensor& self,
    const Tensor& other,
    const Tensor& bias,
    const Tensor& weight,
    double ln_epsilon,
    double p,
    bool training) {
    auto pself = self.expect_contiguous();
    auto pother = other.expect_contiguous();
    auto pweight = weight.expect_contiguous();
    return std::get<0>(at::native_fuse_rn_add_dropout_bias(*pself, *pother, bias, *pweight, ln_epsilon, p, training));
}

Tensor fuse_ctc_loss(
    const Tensor& logits,
    const Tensor& targets,
    const Tensor& logit_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool logits_time_major,
    bool compute_sum,
    bool compute_mean,
    bool check_lengths) {
    // return logits;
    return std::get<0>(at::native_fuse_ctc_loss(logits, targets, logit_lengths, target_lengths, blank, clamp, logits_time_major, compute_sum, compute_mean, check_lengths));
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
