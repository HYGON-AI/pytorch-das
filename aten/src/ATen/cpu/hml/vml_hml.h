#include <hml_vml.h>

// Hygon vML explicit specializations (vs*/vd* interface, no mode argument).
// Same mechanism as the MKL vML stub: source-level explicit calls.
#define IMPLEMENT_VML_HYGON_STUB(op, hmlop)                             \
  template <>                                                           \
  inline void v##op(double* out, const double* in, int64_t size) {      \
    vd##hmlop(static_cast<VML_INT>(size), in, out);                     \
  }                                                                     \
  template <>                                                           \
  inline void v##op(float* out, const float* in, int64_t size) {        \
    vs##hmlop(static_cast<VML_INT>(size), in, out);                     \
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
IMPLEMENT_VML_HYGON_STUB(round, Round)
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

#undef IMPLEMENT_VML_HYGON_STUB
