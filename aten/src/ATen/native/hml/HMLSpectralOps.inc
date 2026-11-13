
#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace at::native {

namespace {
// HFFT/FFTW plans assume 64B-aligned buffers.
// The FFTW planner is not thread-safe, so plan creation and destruction are
// serialized on this mutex; fftw_execute is thread-safe and runs unlocked.
std::mutex& hfft_planner_mutex() {
  static std::mutex m;
  return m;
}
inline bool hfft_is_aligned(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) % 64) == 0;
}
template <typename plan_t, typename create_t, typename exec_t, typename destroy_t>
void hfft_run_plan(create_t&& create, exec_t&& exec, destroy_t&& destroy, const char* what) {
  plan_t plan = nullptr;
  {
    std::lock_guard<std::mutex> lock(hfft_planner_mutex());
    plan = create();
  }
  TORCH_CHECK(plan != nullptr, "HFFT: ", what, " plan creation failed");
  exec(plan);
  {
    std::lock_guard<std::mutex> lock(hfft_planner_mutex());
    destroy(plan);
  }
}
// HFFT normalization scaling (matches pytorch fft_norm_mode semantics)
void hfft_scale_tensor(Tensor& out, int64_t signal_numel, int64_t normalization) {
  TORCH_CHECK(
      normalization >= 0 && normalization <= 2,
      "FFT normalization should be 0 (by_n), 1 (none) or 2 (by_root_n), but got: ",
      normalization);
  const auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) return;
  const double scale = (norm == fft_norm_mode::by_root_n)
      ? 1.0 / std::sqrt(static_cast<double>(signal_numel))
      : 1.0 / static_cast<double>(signal_numel);
  out.mul_(scale);
}
} // anonymous namespace

Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex(), "Only supports complex dtypes, but found: ", self.scalar_type());
  if (dim.empty()) {
    return self.clone();
  }
  // permute FFT dims to the back, make contiguous (FFTW interprets trailing rank dims)
  std::vector<int64_t> non_fft;
  for (int64_t i = 0; i < static_cast<int64_t>(self.dim()); ++i) {
    if (std::find(dim.begin(), dim.end(), i) == dim.end()) non_fft.push_back(i);
  }
  DimVector perm;
  perm.insert(perm.end(), non_fft.begin(), non_fft.end());
  perm.insert(perm.end(), dim.begin(), dim.end());
  bool perm_is_identity = true;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) { perm_is_identity = false; break; }
  }
  auto self_p = self.permute(perm).contiguous();
  if (!hfft_is_aligned(self_p.const_data_ptr())) {
    self_p = self_p.clone();  // move to a 64B-aligned buffer
  }
  auto out = at::empty(self.sizes(), self.options());
  DimVector out_perm_sizes;
  for (auto p : perm) out_perm_sizes.push_back(out.size(p));
  auto out_p = perm_is_identity ? out : at::empty(out_perm_sizes, out.options());

  const int rank = static_cast<int>(dim.size());
  std::vector<int> n(rank);
  int64_t signal_numel = 1;
  for (int i = 0; i < rank; ++i) {
    n[i] = static_cast<int>(self.size(dim[i]));
    signal_numel *= self.size(dim[i]);
  }
  if (signal_numel == 0 || self.numel() == 0) {
    // Empty input: either a transformed dim is 0 (rejected earlier by ATen for
    // signal_numel==0, kept for safety) or a batch dim is 0. In the latter case
    // howmany == 0 makes FFTW plan creation meaningless and the odist computation
    // below would divide by zero, so return an empty result directly.
    out.zero_();
    return out;
  }
  const int howmany = static_cast<int>(self_p.numel() / signal_numel);
  const int idist = static_cast<int>(signal_numel);

  if (self.scalar_type() == kComplexFloat) {
    auto* in = reinterpret_cast<fftwf_complex*>(const_cast<c10::complex<float>*>(self_p.const_data_ptr<c10::complex<float>>()));
    auto* o = reinterpret_cast<fftwf_complex*>(out_p.data_ptr<c10::complex<float>>());
    hfft_run_plan<fftwf_plan>(
        [&] {
          return fftwf_plan_many_dft(rank, n.data(), howmany, in, nullptr, 1, idist,
                                     o, nullptr, 1, idist,
                                     forward ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
        },
        [](fftwf_plan p) { fftwf_execute(p); },
        [](fftwf_plan p) { fftwf_destroy_plan(p); },
        "fftwf c2c");
  } else if (self.scalar_type() == kComplexDouble) {
    auto* in = reinterpret_cast<fftw_complex*>(const_cast<c10::complex<double>*>(self_p.const_data_ptr<c10::complex<double>>()));
    auto* o = reinterpret_cast<fftw_complex*>(out_p.data_ptr<c10::complex<double>>());
    hfft_run_plan<fftw_plan>(
        [&] {
          return fftw_plan_many_dft(rank, n.data(), howmany, in, nullptr, 1, idist,
                                    o, nullptr, 1, idist,
                                    forward ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
        },
        [](fftw_plan p) { fftw_execute(p); },
        [](fftw_plan p) { fftw_destroy_plan(p); },
        "fftw c2c");
  } else {
    TORCH_CHECK(false, "HFFT doesn't support tensors of type: ", self.scalar_type());
  }

  hfft_scale_tensor(out_p, signal_numel, normalization);

  if (!perm_is_identity) {
    DimVector inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv_perm[perm[i]] = static_cast<int64_t>(i);
    out.copy_(out_p.permute(inv_perm));
  }
  return out;
}

Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point(), "Only supports floating-point dtypes, but found: ", self.scalar_type());
  TORCH_CHECK(!dim.empty(), "_fft_r2c: expected at least one dim");
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }
  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));

  std::vector<int64_t> non_fft;
  for (int64_t i = 0; i < static_cast<int64_t>(self.dim()); ++i) {
    if (std::find(dim.begin(), dim.end(), i) == dim.end()) non_fft.push_back(i);
  }
  DimVector perm;
  perm.insert(perm.end(), non_fft.begin(), non_fft.end());
  perm.insert(perm.end(), dim.begin(), dim.end());
  bool perm_is_identity = true;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) { perm_is_identity = false; break; }
  }
  auto self_p = self.permute(perm).contiguous();
  if (!hfft_is_aligned(self_p.const_data_ptr())) {
    self_p = self_p.clone();  // move to a 64B-aligned buffer
  }
  DimVector out_perm_sizes;
  for (auto p : perm) out_perm_sizes.push_back(out.size(p));
  // FFTW r2c only writes the non-redundant half spectrum; when a full output is
  // requested the other half is rebuilt by _fft_fill_with_conjugate_symmetry_.
  if (!onesided) {
    out_perm_sizes.back() = last_dim_halfsize;
  }
  auto out_p = (perm_is_identity && onesided) ? out : at::empty(out_perm_sizes, out.options());

  const int rank = static_cast<int>(dim.size());
  std::vector<int> n(rank);
  int64_t signal_numel = 1;
  for (int i = 0; i < rank; ++i) {
    n[i] = static_cast<int>(self.size(dim[i]));
    signal_numel *= self.size(dim[i]);
  }
  if (signal_numel == 0 || self.numel() == 0) {
    // Empty input: either a transformed dim is 0 (rejected earlier by ATen for
    // signal_numel==0, kept for safety) or a batch dim is 0. In the latter case
    // howmany == 0 makes FFTW plan creation meaningless and the odist computation
    // below would divide by zero, so return an empty result directly.
    out.zero_();
    return out;
  }
  const int howmany = static_cast<int>(self_p.numel() / signal_numel);
  const int idist = static_cast<int>(signal_numel);
  const int odist = static_cast<int>(out_p.numel() / howmany);

  if (self.scalar_type() == kFloat) {
    auto* in = const_cast<float*>(self_p.const_data_ptr<float>());
    auto* o = reinterpret_cast<fftwf_complex*>(out_p.data_ptr<c10::complex<float>>());
    hfft_run_plan<fftwf_plan>(
        [&] {
          return fftwf_plan_many_dft_r2c(rank, n.data(), howmany, in, nullptr, 1, idist,
                                         o, nullptr, 1, odist, FFTW_ESTIMATE);
        },
        [](fftwf_plan p) { fftwf_execute(p); },
        [](fftwf_plan p) { fftwf_destroy_plan(p); },
        "fftwf r2c");
  } else if (self.scalar_type() == kDouble) {
    auto* in = const_cast<double*>(self_p.const_data_ptr<double>());
    auto* o = reinterpret_cast<fftw_complex*>(out_p.data_ptr<c10::complex<double>>());
    hfft_run_plan<fftw_plan>(
        [&] {
          return fftw_plan_many_dft_r2c(rank, n.data(), howmany, in, nullptr, 1, idist,
                                        o, nullptr, 1, odist, FFTW_ESTIMATE);
        },
        [](fftw_plan p) { fftw_execute(p); },
        [](fftw_plan p) { fftw_destroy_plan(p); },
        "fftw r2c");
  } else {
    TORCH_CHECK(false, "HFFT doesn't support tensors of type: ", self.scalar_type());
  }

  hfft_scale_tensor(out_p, signal_numel, normalization);

  if (!onesided) {
    DimVector inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv_perm[perm[i]] = static_cast<int64_t>(i);
    out.narrow(last_dim, 0, last_dim_halfsize).copy_(out_p.permute(inv_perm));
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  } else if (!perm_is_identity) {
    DimVector inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv_perm[perm[i]] = static_cast<int64_t>(i);
    out.copy_(out_p.permute(inv_perm));
  }
  return out;
}

Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  TORCH_CHECK(!dim.empty(), "_fft_c2r: expected at least one dim");
  auto in_sizes = self.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  auto out = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));

  std::vector<int64_t> non_fft;
  for (int64_t i = 0; i < static_cast<int64_t>(self.dim()); ++i) {
    if (std::find(dim.begin(), dim.end(), i) == dim.end()) non_fft.push_back(i);
  }
  DimVector perm;
  perm.insert(perm.end(), non_fft.begin(), non_fft.end());
  perm.insert(perm.end(), dim.begin(), dim.end());
  // HFFT c2r may overwrite the input buffer; keep the user tensor intact
  bool perm_is_identity = true;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) { perm_is_identity = false; break; }
  }
  auto self_p = self.permute(perm).contiguous().clone();
  if (!hfft_is_aligned(self_p.const_data_ptr())) {
    self_p = self_p.clone();  // move to a 64B-aligned buffer
  }
  DimVector out_perm_sizes;
  for (auto p : perm) out_perm_sizes.push_back(out.size(p));
  auto out_p = perm_is_identity ? out : at::empty(out_perm_sizes, out.options());

  const int rank = static_cast<int>(dim.size());
  std::vector<int> n(rank);
  int64_t signal_numel = 1;
  for (int i = 0; i < rank; ++i) {
    n[i] = static_cast<int>(out.size(dim[i]));
    signal_numel *= out.size(dim[i]);
  }
  if (signal_numel == 0 || self.numel() == 0) {
    // Empty input: either a transformed dim is 0 (rejected earlier by ATen for
    // signal_numel==0, kept for safety) or a batch dim is 0. In the latter case
    // howmany == 0 makes FFTW plan creation meaningless and the odist computation
    // below would divide by zero, so return an empty result directly.
    out.zero_();
    return out;
  }
  const int howmany = static_cast<int>(out_p.numel() / signal_numel);
  const int idist = static_cast<int>(signal_numel);
  const int last_n = static_cast<int>(out.size(dim.back()));
  const int odist = static_cast<int>(signal_numel / last_n * (last_n/2 + 1));

  if (self.scalar_type() == kComplexFloat) {
    auto* in = reinterpret_cast<fftwf_complex*>(const_cast<c10::complex<float>*>(self_p.const_data_ptr<c10::complex<float>>()));
    auto* o = out_p.data_ptr<float>();
    hfft_run_plan<fftwf_plan>(
        [&] {
          return fftwf_plan_many_dft_c2r(rank, n.data(), howmany, in, nullptr, 1, odist,
                                         o, nullptr, 1, idist, FFTW_ESTIMATE);
        },
        [](fftwf_plan p) { fftwf_execute(p); },
        [](fftwf_plan p) { fftwf_destroy_plan(p); },
        "fftwf c2r");
  } else if (self.scalar_type() == kComplexDouble) {
    auto* in = reinterpret_cast<fftw_complex*>(const_cast<c10::complex<double>*>(self_p.const_data_ptr<c10::complex<double>>()));
    auto* o = out_p.data_ptr<double>();
    hfft_run_plan<fftw_plan>(
        [&] {
          return fftw_plan_many_dft_c2r(rank, n.data(), howmany, in, nullptr, 1, odist,
                                        o, nullptr, 1, idist, FFTW_ESTIMATE);
        },
        [](fftw_plan p) { fftw_execute(p); },
        [](fftw_plan p) { fftw_destroy_plan(p); },
        "fftw c2r");
  } else {
    TORCH_CHECK(false, "HFFT doesn't support tensors of type: ", self.scalar_type());
  }

  hfft_scale_tensor(out_p, signal_numel, normalization);

  if (!perm_is_identity) {
    DimVector inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv_perm[perm[i]] = static_cast<int64_t>(i);
    out.copy_(out_p.permute(inv_perm));
  }
  return out;
}

} // namespace at::native
