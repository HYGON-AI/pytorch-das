#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <optional>
#include <vector>

#include <hipblaslt/hipblaslt.h>

namespace at {
namespace hip {
namespace detail {

// PyTorch wrapper: flat tensors with int32 offsets.
void hipblaslt_grouped_gemm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    at::Tensor& out);

} // namespace detail
} // namespace hip
} // namespace at
