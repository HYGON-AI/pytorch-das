#pragma once

// Third-party C header: keep it at global scope. This file is included from
// ATen/cpu/vml.h outside namespace at::vml and opens that namespace itself.
#include <hml_vml.h>

#include <algorithm>
#include <limits>

namespace at::vml {

namespace detail {
// VML_INT is 32-bit, so large tensors must be processed in chunks; the MKL
// stub in ATen/cpu/vml.h does the same via max_mkl_ind.
template <typename scalar_t, typename fn_t>
inline void vml_call_chunked(int64_t size, const scalar_t* in, scalar_t* out, fn_t&& fn) {
  constexpr int64_t kMaxChunk = static_cast<int64_t>(std::numeric_limits<VML_INT>::max());
  int64_t offset = 0;
  while (offset < size) {
    const int64_t chunk = std::min(kMaxChunk, size - offset);
    fn(static_cast<VML_INT>(chunk), in + offset, out + offset);
    offset += chunk;
  }
}
} // namespace detail

// Hygon vML explicit specializations (vs*/vd* interface, no mode argument).
// Same mechanism as the MKL vML stub: source-level explicit calls.
#define IMPLEMENT_VML_HYGON_STUB(op, hmlop)                                        \
  template <>                                                                      \
  inline void v##op(double* out, const double* in, int64_t size) {                 \
    detail::vml_call_chunked(                                                      \
        size, in, out, [](VML_INT n, const double* i, double* o) { vd##hmlop(n, i, o); }); \
  }                                                                                \
  template <>                                                                      \
  inline void v##op(float* out, const float* in, int64_t size) {                    \
    detail::vml_call_chunked(                                                       \
        size, in, out, [](VML_INT n, const float* i, float* o) { vs##hmlop(n, i, o); });   \
  }

IMPLEMENT_VML_HYGON_STUB(sin, Sin)
IMPLEMENT_VML_HYGON_STUB(cos, Cos)
IMPLEMENT_VML_HYGON_STUB(tan, Tan)
IMPLEMENT_VML_HYGON_STUB(exp, Exp)
IMPLEMENT_VML_HYGON_STUB(log, Ln)
IMPLEMENT_VML_HYGON_STUB(log10, Log10)
IMPLEMENT_VML_HYGON_STUB(log2, Log2)
IMPLEMENT_VML_HYGON_STUB(log1p, Log1p)
IMPLEMENT_VML_HYGON_STUB(abs, Abs)
IMPLEMENT_VML_HYGON_STUB(ceil, Ceil)
IMPLEMENT_VML_HYGON_STUB(floor, Floor)
IMPLEMENT_VML_HYGON_STUB(erf, Erf)
IMPLEMENT_VML_HYGON_STUB(erfc, Erfc)
IMPLEMENT_VML_HYGON_STUB(erfinv, ErfInv)
IMPLEMENT_VML_HYGON_STUB(expm1, Expm1)
IMPLEMENT_VML_HYGON_STUB(rsqrt, InvSqrt)
IMPLEMENT_VML_HYGON_STUB(sqrt, Sqrt)
IMPLEMENT_VML_HYGON_STUB(acos, Acos)
IMPLEMENT_VML_HYGON_STUB(asin, Asin)
IMPLEMENT_VML_HYGON_STUB(atan, Atan)
IMPLEMENT_VML_HYGON_STUB(atanh, Atanh)
IMPLEMENT_VML_HYGON_STUB(tanh, Tanh)
//hpckit latest version fix 
//IMPLEMENT_VML_HYGON_STUB(round, RoundEven)

#undef IMPLEMENT_VML_HYGON_STUB

} // namespace at::vml
