#undef __HIP_NO_HALF_CONVERSIONS__
#include <ATen/hip/HIPContext.h>
#include <ATen/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/hip/HIPStream.h>
#include <c10/core/ScalarType.h>
#include <vector>
#include <optional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <torch/library.h>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define TORCH_HIPBLASLT_CHECK(EXPR)               \
  do {                                            \
    hipblasStatus_t __err = EXPR;                 \
    TORCH_CHECK(__err == HIPBLAS_STATUS_SUCCESS,  \
                "hipblaslt error: ",              \
                hipblasStatusToString(__err),     \
                " when calling `" #EXPR "`");     \
  } while (0)

#define TORCH_HIP_CHECK(EXPR)                     \
  do {                                            \
    hipError_t __err = EXPR;                      \
    TORCH_CHECK(__err == hipSuccess,              \
                "hip error: ",                    \
                hipGetErrorString(__err));        \
  } while (0)

namespace at {
namespace hip {
namespace detail {

namespace {
class HipBlasltHandleManager {
 public:
  static hipblasLtHandle_t GetHandle() {
    static HipBlasltHandleManager mgr;
    return mgr.handle_;
  }
 private:
  HipBlasltHandleManager() {
    TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle_));
  }
  ~HipBlasltHandleManager() {
    hipblasLtDestroy(handle_);
  }
  hipblasLtHandle_t handle_;
};

hipDataType _hip_type_from_scalar_type(c10::ScalarType t) {
  switch (t) {
    case c10::ScalarType::Half: return HIP_R_16F;
    case c10::ScalarType::BFloat16: return HIP_R_16BF;
    case c10::ScalarType::Float: return HIP_R_32F;
    case c10::ScalarType::Int: return HIP_R_32I;
    case c10::ScalarType::Char: return HIP_R_8I;
    default:
      TORCH_CHECK(false, "hipblaslt_grouped_gemm: unsupported dtype ", t);
  }
}

enum class BlasMatrixLayout {
  RowMajor,
  ColMajor,
  Unsupported,
};

BlasMatrixLayout classify_blas_matrix_layout(
    int64_t rows,
    int64_t cols,
    int64_t row_stride,
    int64_t col_stride) {
  if (rows <= 0 || cols <= 0) {
    return BlasMatrixLayout::Unsupported;
  }
  if (col_stride == 1 && row_stride >= cols) {
    return BlasMatrixLayout::RowMajor;
  }
  if (row_stride == 1 && col_stride >= rows) {
    return BlasMatrixLayout::ColMajor;
  }
  return BlasMatrixLayout::Unsupported;
}

int64_t blas_leading_dim(
    BlasMatrixLayout layout,
    int64_t row_stride,
    int64_t col_stride) {
  return layout == BlasMatrixLayout::RowMajor ? row_stride : col_stride;
}

struct HipBlasltUserArgs {
  HipBlasltUserArgs() : stream_(nullptr), raw_(nullptr), event_(nullptr) {}
  HipBlasltUserArgs(hipStream_t stream, size_t size, bool host)
      : stream_(stream), raw_(nullptr), event_(nullptr), host_(host) {
    hipblaslt_ext::UserArguments* raw_ptr = nullptr;
    if (host) {
      TORCH_HIP_CHECK(hipHostMalloc(
          &raw_ptr, size * sizeof(hipblaslt_ext::UserArguments)));
    } else {
      TORCH_HIP_CHECK(hipMalloc(
          &raw_ptr, size * sizeof(hipblaslt_ext::UserArguments)));
    }
    raw_ = raw_ptr;
    if (host) {
      TORCH_HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventBlockingSync));
    } else {
      TORCH_HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
    }
  }
  HipBlasltUserArgs(const HipBlasltUserArgs&) = delete;
  HipBlasltUserArgs(HipBlasltUserArgs&& other) noexcept
      : stream_(other.stream_), raw_(other.raw_), event_(other.event_), host_(other.host_) {
    other.stream_ = nullptr;
    other.raw_ = nullptr;
    other.event_ = nullptr;
    other.host_ = false;
  }
  HipBlasltUserArgs& operator=(const HipBlasltUserArgs&) = delete;
  HipBlasltUserArgs& operator=(HipBlasltUserArgs&& other) noexcept {
    if (this != &other) {
      free();
      stream_ = other.stream_;
      raw_ = other.raw_;
      event_ = other.event_;
      host_ = other.host_;
      other.stream_ = nullptr;
      other.raw_ = nullptr;
      other.event_ = nullptr;
      other.host_ = false;
    }
    return *this;
  }
  ~HipBlasltUserArgs() { free(); }

  hipStream_t getStream() const { return stream_; }
  hipblaslt_ext::UserArguments* getArgs() const { return raw_; }
  hipEvent_t getEvent() const { return event_; }
  void setStream(hipStream_t s) { stream_ = s; }

 private:
  void free() {
    if (raw_) {
      if (event_) {
        TORCH_HIP_CHECK(hipEventSynchronize(event_));
        TORCH_HIP_CHECK(hipEventDestroy(event_));
        event_ = nullptr;
      }
      if (host_) {
        TORCH_HIP_CHECK(hipHostFree(raw_));
      } else {
        TORCH_HIP_CHECK(hipFree(raw_));
      }
      raw_ = nullptr;
    }
  }
  hipStream_t stream_;
  hipblaslt_ext::UserArguments* raw_;
  hipEvent_t event_;
  bool host_{false};
};

struct HipBlasltUserArgsBuffer {
  HipBlasltUserArgsBuffer() = default;
  HipBlasltUserArgsBuffer(hipStream_t stream, size_t size, bool host) {
    for (int i = 0; i < 4; ++i) {
      buffer_[i] = HipBlasltUserArgs(stream, size, host);
    }
  }
  HipBlasltUserArgsBuffer(const HipBlasltUserArgsBuffer&) = delete;
  HipBlasltUserArgsBuffer(HipBlasltUserArgsBuffer&& other) noexcept {
    for (int i = 0; i < 4; ++i) {
      buffer_[i] = std::move(other.buffer_[i]);
    }
    index_ = other.index_;
  }
  HipBlasltUserArgsBuffer& operator=(const HipBlasltUserArgsBuffer&) = delete;
  HipBlasltUserArgsBuffer& operator=(HipBlasltUserArgsBuffer&& other) noexcept {
    if (this != &other) {
      for (int i = 0; i < 4; ++i) {
        buffer_[i] = std::move(other.buffer_[i]);
      }
      index_ = other.index_;
    }
    return *this;
  }
  HipBlasltUserArgs& getUserArgs() {
    HipBlasltUserArgs& args = buffer_[index_];
    if (index_ < 3) {
      ++index_;
    } else {
      index_ = 0;
    }
    return args;
  }

 private:
  int index_ = 0;
  HipBlasltUserArgs buffer_[4];
};

struct HipBlasltUserArgsCache {
  HipBlasltUserArgsCache() = default;
  HipBlasltUserArgsCache(const HipBlasltUserArgsCache&) = delete;
  HipBlasltUserArgsCache& operator=(const HipBlasltUserArgsCache&) = delete;

  HipBlasltUserArgsBuffer& getBuffer(hipStream_t stream, size_t size, bool host) {
    auto& buffers = host ? host_buffers_ : device_buffers_;
    auto it = buffers.find(size);
    if (it != buffers.end()) {
      return it->second;
    }
    return buffers
        .emplace(size, HipBlasltUserArgsBuffer{stream, size, host})
        .first->second;
  }

 private:
  std::unordered_map<size_t, HipBlasltUserArgsBuffer> host_buffers_;
  std::unordered_map<size_t, HipBlasltUserArgsBuffer> device_buffers_;
};

struct HipBlasltUserArgsCacheManager {
  static HipBlasltUserArgsCacheManager& instance() {
    static thread_local HipBlasltUserArgsCacheManager instance_;
    return instance_;
  }

  HipBlasltUserArgsCache& getCache() {
    int device_id;
    TORCH_HIP_CHECK(hipGetDevice(&device_id));
    TORCH_CHECK(0 <= device_id && device_id < (int)caches_.size(),
                "invalid HIP device ID");
    return *caches_[device_id];
  }

 private:
  HipBlasltUserArgsCacheManager() {
    int count;
    TORCH_HIP_CHECK(hipGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
      caches_.push_back(std::make_unique<HipBlasltUserArgsCache>());
    }
  }
  std::vector<std::unique_ptr<HipBlasltUserArgsCache>> caches_;
};


void hipblaslt_groupedgemm(
    const std::vector<at::Tensor>& inputA,
    const std::vector<at::Tensor>& inputB,
    std::vector<at::Tensor>& outputD,
    const std::vector<std::optional<at::Tensor>>& bias,
    bool use_bias,
    std::vector<int64_t>& m,
    std::vector<int64_t>& n,
    std::vector<int64_t>& k,
    std::vector<int64_t>& b,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    bool accumulate,
    hipStream_t stream,
    std::vector<int64_t>* lda = nullptr,
    std::vector<int64_t>* ldb = nullptr,
    std::vector<int64_t>* ldc = nullptr,
    std::vector<int64_t>* ldd = nullptr,
    std::vector<int64_t>* strideA = nullptr,
    std::vector<int64_t>* strideB = nullptr,
    std::vector<int64_t>* strideC = nullptr,
    std::vector<int64_t>* strideD = nullptr) {

  int num_groups = static_cast<int>(m.size());
  TORCH_CHECK(num_groups > 0, "hipblaslt_groupedgemm: empty problem");

  hipblasLtHandle_t handle = HipBlasltHandleManager::GetHandle();

  HipBlasltUserArgs& device_user_args =
      HipBlasltUserArgsCacheManager::instance().getCache()
          .getBuffer(stream, num_groups, false).getUserArgs();
  hipblaslt_ext::UserArguments* device_args = device_user_args.getArgs();
  hipEvent_t device_event = device_user_args.getEvent();
  hipStream_t device_stream = device_user_args.getStream();

  HipBlasltUserArgs& host_user_args =
      HipBlasltUserArgsCacheManager::instance().getCache()
          .getBuffer(stream, num_groups, true).getUserArgs();
  hipblaslt_ext::UserArguments* host_args = host_user_args.getArgs();
  hipEvent_t host_event = host_user_args.getEvent();

  const hipDataType A_type = _hip_type_from_scalar_type(inputA[0].scalar_type());
  const hipDataType B_type = _hip_type_from_scalar_type(inputB[0].scalar_type());
  const hipDataType D_type = _hip_type_from_scalar_type(outputD[0].scalar_type());
  hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;

  float one = 1.0f, zero = 0.0f;
  float beta_val = accumulate ? one : zero;
  int int_one = 1, int_zero = 0;
  bool use_int8 = (A_type == HIP_R_8I && B_type == HIP_R_8I && D_type == HIP_R_32I);
  if (use_int8) {
    TORCH_CHECK(!accumulate, "Int8 grouped gemm does not support accumulate");
    computeType = HIPBLAS_COMPUTE_32I;
  }

  std::vector<hipblaslt_ext::GemmEpilogue> epilogue{hipblaslt_ext::GemmEpilogue()};
  if (use_bias) {
    const hipDataType bias_type = _hip_type_from_scalar_type(bias[0]->scalar_type());
    TORCH_CHECK(bias_type == HIP_R_32F || bias_type == HIP_R_16BF,
                "hipblaslt_groupedgemm: bias type must be f32 or bf16");
    epilogue[0].mode = HIPBLASLT_EPILOGUE_BIAS;
    epilogue[0].bias_data_type = bias_type;
  }

  std::vector<hipblaslt_ext::GemmInputs> inputs(num_groups);
  for (int i = 0; i < num_groups; ++i) {
    TORCH_CHECK(m[i] > 0 && n[i] > 0 && k[i] > 0 && b[i] > 0,
                "hipblaslt_groupedgemm: invalid problem dimensions at group ", i);

    inputs[i].a = inputA[i].data_ptr();
    inputs[i].b = inputB[i].data_ptr();
    inputs[i].c = outputD[i].data_ptr();
    inputs[i].d = outputD[i].data_ptr();
    inputs[i].alpha = use_int8
        ? static_cast<const void*>(&int_one)
        : static_cast<const void*>(&one);
    inputs[i].beta = use_int8
        ? static_cast<const void*>(&int_zero)
        : static_cast<const void*>(&beta_val);
    inputs[i].bias = bias[i]->data_ptr();
  }

  hipblaslt_ext::GroupedGemm grouped_gemm(
      handle, transa, transb, A_type, B_type, D_type, D_type, computeType);
  if (lda != nullptr) {
    hipblaslt_ext::GemmProblemType problem_type{
        transa, transb, A_type, B_type, D_type, D_type, computeType};
    TORCH_HIPBLASLT_CHECK(grouped_gemm.setProblem(
        m, n, k, b,
        *lda, *ldb, *ldc, *ldd,
        *strideA, *strideB, *strideC, *strideD,
        epilogue, inputs, problem_type));
  } else {
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.setProblem(m, n, k, b, epilogue, inputs));
  }

  {
    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(0);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.algoGetHeuristic(1, gemm_pref, heuristic_result));
    TORCH_CHECK(!heuristic_result.empty(),
                "hipblaslt_groupedgemm: no valid algorithm found");
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.initialize(heuristic_result[0].algo, nullptr, true, stream));
  }

  TORCH_HIPBLASLT_CHECK(
      grouped_gemm.getDefaultValueForDeviceUserArguments(host_args));

  if (stream != device_stream) {
    TORCH_HIP_CHECK(hipStreamWaitEvent(stream, device_event, 0));
  }
  TORCH_HIP_CHECK(hipMemcpyAsync(
      device_args, host_args,
      num_groups * sizeof(hipblaslt_ext::UserArguments),
      hipMemcpyHostToDevice, stream));
  TORCH_HIP_CHECK(hipEventRecord(host_event, stream));

  TORCH_HIPBLASLT_CHECK(grouped_gemm.run(device_args, stream));
  device_user_args.setStream(stream);
  TORCH_HIP_CHECK(hipEventRecord(device_event, stream));
}

void hipblaslt_groupedgemm_raw(
    const std::vector<void*>& inputA,
    const std::vector<void*>& inputB,
    const std::vector<void*>& outputD,
    const std::vector<void*>& bias,
    bool use_bias,
    c10::ScalarType A_scalar_type,
    c10::ScalarType B_scalar_type,
    c10::ScalarType D_scalar_type,
    c10::ScalarType bias_scalar_type,
    std::vector<int64_t>& m,
    std::vector<int64_t>& n,
    std::vector<int64_t>& k,
    std::vector<int64_t>& b,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    bool accumulate,
    hipStream_t stream,
    std::vector<int64_t>* lda = nullptr,
    std::vector<int64_t>* ldb = nullptr,
    std::vector<int64_t>* ldc = nullptr,
    std::vector<int64_t>* ldd = nullptr,
    std::vector<int64_t>* strideA = nullptr,
    std::vector<int64_t>* strideB = nullptr,
    std::vector<int64_t>* strideC = nullptr,
    std::vector<int64_t>* strideD = nullptr) {

  int num_groups = static_cast<int>(m.size());
  TORCH_CHECK(num_groups > 0, "hipblaslt_groupedgemm: empty problem");

  hipblasLtHandle_t handle = HipBlasltHandleManager::GetHandle();

  HipBlasltUserArgs& device_user_args =
      HipBlasltUserArgsCacheManager::instance().getCache()
          .getBuffer(stream, num_groups, false).getUserArgs();
  hipblaslt_ext::UserArguments* device_args = device_user_args.getArgs();
  hipEvent_t device_event = device_user_args.getEvent();
  hipStream_t device_stream = device_user_args.getStream();

  HipBlasltUserArgs& host_user_args =
      HipBlasltUserArgsCacheManager::instance().getCache()
          .getBuffer(stream, num_groups, true).getUserArgs();
  hipblaslt_ext::UserArguments* host_args = host_user_args.getArgs();
  hipEvent_t host_event = host_user_args.getEvent();

  const hipDataType A_type = _hip_type_from_scalar_type(A_scalar_type);
  const hipDataType B_type = _hip_type_from_scalar_type(B_scalar_type);
  const hipDataType D_type = _hip_type_from_scalar_type(D_scalar_type);
  hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;

  float one = 1.0f, zero = 0.0f;
  float beta_val = accumulate ? one : zero;
  int int_one = 1, int_zero = 0;
  bool use_int8 = (A_type == HIP_R_8I && B_type == HIP_R_8I && D_type == HIP_R_32I);
  if (use_int8) {
    TORCH_CHECK(!accumulate, "Int8 grouped gemm does not support accumulate");
    computeType = HIPBLAS_COMPUTE_32I;
  }

  std::vector<hipblaslt_ext::GemmEpilogue> epilogue{hipblaslt_ext::GemmEpilogue()};
  if (use_bias) {
    const hipDataType bias_type = _hip_type_from_scalar_type(bias_scalar_type);
    TORCH_CHECK(bias_type == HIP_R_32F || bias_type == HIP_R_16BF,
                "hipblaslt_groupedgemm: bias type must be f32 or bf16");
    epilogue[0].mode = HIPBLASLT_EPILOGUE_BIAS;
    epilogue[0].bias_data_type = bias_type;
  }

  std::vector<hipblaslt_ext::GemmInputs> inputs(num_groups);
  for (int i = 0; i < num_groups; ++i) {
    TORCH_CHECK(m[i] > 0 && n[i] > 0 && k[i] > 0 && b[i] > 0,
                "hipblaslt_groupedgemm: invalid problem dimensions at group ", i);

    inputs[i].a = inputA[i];
    inputs[i].b = inputB[i];
    inputs[i].c = outputD[i];
    inputs[i].d = outputD[i];
    inputs[i].alpha = use_int8
        ? static_cast<const void*>(&int_one)
        : static_cast<const void*>(&one);
    inputs[i].beta = use_int8
        ? static_cast<const void*>(&int_zero)
        : static_cast<const void*>(&beta_val);
    inputs[i].bias = bias[i];
  }

  hipblaslt_ext::GroupedGemm grouped_gemm(
      handle, transa, transb, A_type, B_type, D_type, D_type, computeType);
  if (lda != nullptr) {
    hipblaslt_ext::GemmProblemType problem_type{
        transa, transb, A_type, B_type, D_type, D_type, computeType};
    TORCH_HIPBLASLT_CHECK(grouped_gemm.setProblem(
        m, n, k, b,
        *lda, *ldb, *ldc, *ldd,
        *strideA, *strideB, *strideC, *strideD,
        epilogue, inputs, problem_type));
  } else {
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.setProblem(m, n, k, b, epilogue, inputs));
  }

  {
    hipblaslt_ext::GemmPreference gemm_pref;
    gemm_pref.setMaxWorkspaceBytes(0);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.algoGetHeuristic(1, gemm_pref, heuristic_result));
    TORCH_CHECK(!heuristic_result.empty(),
                "hipblaslt_groupedgemm: no valid algorithm found");
    TORCH_HIPBLASLT_CHECK(
        grouped_gemm.initialize(heuristic_result[0].algo, nullptr, true, stream));
  }

  TORCH_HIPBLASLT_CHECK(
      grouped_gemm.getDefaultValueForDeviceUserArguments(host_args));

  if (stream != device_stream) {
    TORCH_HIP_CHECK(hipStreamWaitEvent(stream, device_event, 0));
  }
  TORCH_HIP_CHECK(hipMemcpyAsync(
      device_args, host_args,
      num_groups * sizeof(hipblaslt_ext::UserArguments),
      hipMemcpyHostToDevice, stream));
  TORCH_HIP_CHECK(hipEventRecord(host_event, stream));

  TORCH_HIPBLASLT_CHECK(grouped_gemm.run(device_args, stream));
  device_user_args.setStream(stream);
  TORCH_HIP_CHECK(hipEventRecord(device_event, stream));
}

}


void hipblaslt_grouped_gemm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias_opt,
    at::Tensor& out) {
  const int a_dim = mat_a.dim();
  const int b_dim = mat_b.dim();

  bool use_bias = bias_opt.has_value() && bias_opt->defined();

  at::Tensor zero_bias_per_device;
  if (!use_bias) {
    static std::mutex zb_mutex;
    static std::unordered_map<int, at::Tensor> zb_cache;
    int dev_id = mat_a.get_device();
    std::lock_guard<std::mutex> lock(zb_mutex);
    auto it = zb_cache.find(dev_id);
    if (it != zb_cache.end()) {
      zero_bias_per_device = it->second;
    } else {
      zero_bias_per_device = at::zeros({1024}, mat_a.options().dtype(at::kFloat));
      zb_cache[dev_id] = zero_bias_per_device;
    }
  }

  std::vector<void*> raw_a, raw_b, raw_out, raw_bias;
  std::vector<int64_t> m_vec, n_vec, k_vec, batch_vec;
  std::vector<int64_t> lda_vec, ldb_vec, ldc_vec, ldd_vec;
  std::vector<int64_t> strideA_vec, strideB_vec, strideC_vec, strideD_vec;

  void* bias_ptr = use_bias
      ? bias_opt->data_ptr()
      : zero_bias_per_device.data_ptr();

  if (a_dim == 2 && b_dim == 2) {
    const BlasMatrixLayout a2d_layout = classify_blas_matrix_layout(
        mat_a.size(0), mat_a.size(1), mat_a.stride(0), mat_a.stride(1));
    const BlasMatrixLayout b2d_layout = classify_blas_matrix_layout(
        mat_b.size(0), mat_b.size(1), mat_b.stride(0), mat_b.stride(1));
    TORCH_CHECK(a2d_layout != BlasMatrixLayout::Unsupported &&
                b2d_layout != BlasMatrixLayout::Unsupported,
                "hipblaslt_grouped_gemm: unsupported layout in 2d/2d");

    at::Tensor out_contig = out.contiguous();
    bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

    auto offs_cpu = offs->cpu();
    auto offs_accessor = offs_cpu.accessor<int, 1>();
    int num_groups = offs_accessor.size(0);

    raw_a.reserve(num_groups);
    raw_b.reserve(num_groups);
    raw_out.reserve(num_groups);
    raw_bias.reserve(num_groups);
    m_vec.reserve(num_groups);
    n_vec.reserve(num_groups);
    k_vec.reserve(num_groups);
    batch_vec.reserve(num_groups);
    lda_vec.reserve(num_groups);
    ldb_vec.reserve(num_groups);
    ldc_vec.reserve(num_groups);
    ldd_vec.reserve(num_groups);
    strideA_vec.reserve(num_groups);
    strideB_vec.reserve(num_groups);
    strideC_vec.reserve(num_groups);
    strideD_vec.reserve(num_groups);

    char* b_base = static_cast<char*>(mat_b.data_ptr());
    char* a_base = static_cast<char*>(mat_a.data_ptr());
    char* out_base = static_cast<char*>(out_contig.data_ptr());
    const int64_t b_elem = mat_b.element_size();
    const int64_t a_elem = mat_a.element_size();
    const int64_t out_elem = out_contig.element_size();
    const int64_t m_size = mat_a.size(0);
    const int64_t n_size = mat_b.size(1);
    const int64_t lda = blas_leading_dim(
        b2d_layout, mat_b.stride(0), mat_b.stride(1));
    const int64_t ldb = blas_leading_dim(
        a2d_layout, mat_a.stride(0), mat_a.stride(1));
    const hipblasOperation_t raw_opA =
        b2d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    const hipblasOperation_t raw_opB =
        a2d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    std::vector<int> empty_groups;
    for (int i = 0; i < num_groups; ++i) {
      int64_t start_k = (i == 0) ? 0 : offs_accessor[i - 1];
      int64_t end_k = offs_accessor[i];
      int64_t k_size = end_k - start_k;
      if (k_size <= 0) { empty_groups.push_back(i); continue; }

      raw_a.push_back(b_base + start_k * mat_b.stride(0) * b_elem);
      raw_b.push_back(a_base + start_k * mat_a.stride(1) * a_elem);
      raw_out.push_back(out_base + i * out_contig.stride(0) * out_elem);
      raw_bias.push_back(bias_ptr);
      m_vec.push_back(n_size);
      n_vec.push_back(m_size);
      k_vec.push_back(k_size);
      batch_vec.push_back(1);
      lda_vec.push_back(lda);
      ldb_vec.push_back(ldb);
      ldc_vec.push_back(n_size);
      ldd_vec.push_back(n_size);
      strideA_vec.push_back(0);
      strideB_vec.push_back(0);
      strideC_vec.push_back(0);
      strideD_vec.push_back(0);
    }

    if (!m_vec.empty()) {
      hipStream_t stream = c10::hip::getCurrentHIPStream();
      hipblaslt_groupedgemm_raw(
          raw_a, raw_b, raw_out, raw_bias,
          use_bias,
          mat_b.scalar_type(),
          mat_a.scalar_type(),
          out_contig.scalar_type(),
          use_bias ? bias_opt->scalar_type() : zero_bias_per_device.scalar_type(),
          m_vec, n_vec, k_vec, batch_vec,
          raw_opA, raw_opB,
          false, stream,
          &lda_vec, &ldb_vec, &ldc_vec, &ldd_vec,
          &strideA_vec, &strideB_vec, &strideC_vec, &strideD_vec);
    }

    // Zero only the output slices for groups skipped above (no tokens).
    for (int i : empty_groups) {
      out_contig.select(0, i).zero_();
    }

    if (need_out_copy) {
      out.copy_(out_contig);
    }
    return;
  }

  if (a_dim == 2 && b_dim == 3) {
    const BlasMatrixLayout a2d_layout = classify_blas_matrix_layout(
        mat_a.size(0), mat_a.size(1), mat_a.stride(0), mat_a.stride(1));
    const BlasMatrixLayout b3d_layout = classify_blas_matrix_layout(
        mat_b.size(1), mat_b.size(2), mat_b.stride(1), mat_b.stride(2));
    TORCH_CHECK(a2d_layout != BlasMatrixLayout::Unsupported &&
                b3d_layout != BlasMatrixLayout::Unsupported,
                "hipblaslt_grouped_gemm: unsupported layout in 2d/3d");

    at::Tensor out_contig = out.contiguous();
    bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

    auto offs_cpu = offs->cpu();
    auto offs_accessor = offs_cpu.accessor<int, 1>();
    int num_groups = offs_accessor.size(0);

    raw_a.reserve(num_groups);
    raw_b.reserve(num_groups);
    raw_out.reserve(num_groups);
    raw_bias.reserve(num_groups);
    m_vec.reserve(num_groups);
    n_vec.reserve(num_groups);
    k_vec.reserve(num_groups);
    batch_vec.reserve(num_groups);
    lda_vec.reserve(num_groups);
    ldb_vec.reserve(num_groups);
    ldc_vec.reserve(num_groups);
    ldd_vec.reserve(num_groups);
    strideA_vec.reserve(num_groups);
    strideB_vec.reserve(num_groups);
    strideC_vec.reserve(num_groups);
    strideD_vec.reserve(num_groups);

    char* b_base = static_cast<char*>(mat_b.data_ptr());
    char* a_base = static_cast<char*>(mat_a.data_ptr());
    char* out_base = static_cast<char*>(out_contig.data_ptr());
    const int64_t b_elem = mat_b.element_size();
    const int64_t a_elem = mat_a.element_size();
    const int64_t out_elem = out_contig.element_size();
    const int64_t n_size = mat_b.size(2);
    const int64_t k_size = mat_a.size(1);
    const int64_t lda = blas_leading_dim(
        b3d_layout, mat_b.stride(1), mat_b.stride(2));
    const int64_t ldb = blas_leading_dim(
        a2d_layout, mat_a.stride(0), mat_a.stride(1));
    const hipblasOperation_t raw_opA =
        b3d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    const hipblasOperation_t raw_opB =
        a2d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    for (int i = 0; i < num_groups; ++i) {
      int64_t start_m = (i == 0) ? 0 : offs_accessor[i - 1];
      int64_t end_m = offs_accessor[i];
      int64_t m_size = end_m - start_m;
      if (m_size <= 0) continue;

      raw_a.push_back(b_base + i * mat_b.stride(0) * b_elem);
      raw_b.push_back(a_base + start_m * mat_a.stride(0) * a_elem);
      raw_out.push_back(out_base + start_m * out_contig.stride(0) * out_elem);
      raw_bias.push_back(bias_ptr);
      m_vec.push_back(n_size);
      n_vec.push_back(m_size);
      k_vec.push_back(k_size);
      batch_vec.push_back(1);
      lda_vec.push_back(lda);
      ldb_vec.push_back(ldb);
      ldc_vec.push_back(n_size);
      ldd_vec.push_back(n_size);
      strideA_vec.push_back(0);
      strideB_vec.push_back(0);
      strideC_vec.push_back(0);
      strideD_vec.push_back(0);
    }

    if (!m_vec.empty()) {
      hipStream_t stream = c10::hip::getCurrentHIPStream();
      hipblaslt_groupedgemm_raw(
          raw_a, raw_b, raw_out, raw_bias,
          use_bias,
          mat_b.scalar_type(),
          mat_a.scalar_type(),
          out_contig.scalar_type(),
          use_bias ? bias_opt->scalar_type() : zero_bias_per_device.scalar_type(),
          m_vec, n_vec, k_vec, batch_vec,
          raw_opA, raw_opB,
          false, stream,
          &lda_vec, &ldb_vec, &ldc_vec, &ldd_vec,
          &strideA_vec, &strideB_vec, &strideC_vec, &strideD_vec);
    }

    if (need_out_copy) {
      out.copy_(out_contig);
    }
    return;
  }

  if (a_dim == 3 && b_dim == 2) {
    const BlasMatrixLayout a3d_layout = classify_blas_matrix_layout(
        mat_a.size(1), mat_a.size(2), mat_a.stride(1), mat_a.stride(2));
    const BlasMatrixLayout b2d_layout = classify_blas_matrix_layout(
        mat_b.size(0), mat_b.size(1), mat_b.stride(0), mat_b.stride(1));
    TORCH_CHECK(a3d_layout != BlasMatrixLayout::Unsupported &&
                b2d_layout != BlasMatrixLayout::Unsupported,
                "hipblaslt_grouped_gemm: unsupported layout in 3d/2d");

    at::Tensor out_contig = out.contiguous();
    bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

    auto offs_cpu = offs->cpu();
    auto offs_accessor = offs_cpu.accessor<int, 1>();
    int num_groups = offs_accessor.size(0);

    raw_a.reserve(num_groups);
    raw_b.reserve(num_groups);
    raw_out.reserve(num_groups);
    raw_bias.reserve(num_groups);
    m_vec.reserve(num_groups);
    n_vec.reserve(num_groups);
    k_vec.reserve(num_groups);
    batch_vec.reserve(num_groups);
    lda_vec.reserve(num_groups);
    ldb_vec.reserve(num_groups);
    ldc_vec.reserve(num_groups);
    ldd_vec.reserve(num_groups);
    strideA_vec.reserve(num_groups);
    strideB_vec.reserve(num_groups);
    strideC_vec.reserve(num_groups);
    strideD_vec.reserve(num_groups);

    char* b_base = static_cast<char*>(mat_b.data_ptr());
    char* a_base = static_cast<char*>(mat_a.data_ptr());
    char* out_base = static_cast<char*>(out_contig.data_ptr());
    const int64_t b_elem = mat_b.element_size();
    const int64_t a_elem = mat_a.element_size();
    const int64_t out_elem = out_contig.element_size();
    const int64_t m_size = mat_a.size(1);
    const int64_t n_total = mat_b.size(1);
    const int64_t k_size = mat_a.size(2);
    const int64_t lda = blas_leading_dim(
        b2d_layout, mat_b.stride(0), mat_b.stride(1));
    const int64_t ldb = blas_leading_dim(
        a3d_layout, mat_a.stride(1), mat_a.stride(2));
    const hipblasOperation_t raw_opA =
        b2d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    const hipblasOperation_t raw_opB =
        a3d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    for (int i = 0; i < num_groups; ++i) {
      int64_t start_n = (i == 0) ? 0 : offs_accessor[i - 1];
      int64_t end_n = offs_accessor[i];
      int64_t n_size = end_n - start_n;
      if (n_size <= 0) continue;

      raw_a.push_back(b_base + start_n * mat_b.stride(1) * b_elem);
      raw_b.push_back(a_base + i * mat_a.stride(0) * a_elem);
      raw_out.push_back(out_base + start_n * out_contig.stride(1) * out_elem);
      raw_bias.push_back(bias_ptr);
      m_vec.push_back(n_size);
      n_vec.push_back(m_size);
      k_vec.push_back(k_size);
      batch_vec.push_back(1);
      lda_vec.push_back(lda);
      ldb_vec.push_back(ldb);
      ldc_vec.push_back(n_total);
      ldd_vec.push_back(n_total);
      strideA_vec.push_back(0);
      strideB_vec.push_back(0);
      strideC_vec.push_back(0);
      strideD_vec.push_back(0);
    }

    if (!m_vec.empty()) {
      hipStream_t stream = c10::hip::getCurrentHIPStream();
      hipblaslt_groupedgemm_raw(
          raw_a, raw_b, raw_out, raw_bias,
          use_bias,
          mat_b.scalar_type(),
          mat_a.scalar_type(),
          out_contig.scalar_type(),
          use_bias ? bias_opt->scalar_type() : zero_bias_per_device.scalar_type(),
          m_vec, n_vec, k_vec, batch_vec,
          raw_opA, raw_opB,
          false, stream,
          &lda_vec, &ldb_vec, &ldc_vec, &ldd_vec,
          &strideA_vec, &strideB_vec, &strideC_vec, &strideD_vec);
    }

    if (need_out_copy) {
      out.copy_(out_contig);
    }
    return;
  }

  if (a_dim == 3 && b_dim == 3) {
    const BlasMatrixLayout a3d_layout = classify_blas_matrix_layout(
        mat_a.size(1), mat_a.size(2), mat_a.stride(1), mat_a.stride(2));
    const BlasMatrixLayout b3d_layout = classify_blas_matrix_layout(
        mat_b.size(1), mat_b.size(2), mat_b.stride(1), mat_b.stride(2));
    TORCH_CHECK(a3d_layout != BlasMatrixLayout::Unsupported &&
                b3d_layout != BlasMatrixLayout::Unsupported,
                "hipblaslt_grouped_gemm: unsupported layout in 3d/3d");

    at::Tensor out_contig = out.contiguous();
    bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

    int64_t batch = mat_a.size(0);

    raw_a.reserve(batch);
    raw_b.reserve(batch);
    raw_out.reserve(batch);
    raw_bias.reserve(batch);
    m_vec.reserve(batch);
    n_vec.reserve(batch);
    k_vec.reserve(batch);
    batch_vec.reserve(batch);
    lda_vec.reserve(batch);
    ldb_vec.reserve(batch);
    ldc_vec.reserve(batch);
    ldd_vec.reserve(batch);
    strideA_vec.reserve(batch);
    strideB_vec.reserve(batch);
    strideC_vec.reserve(batch);
    strideD_vec.reserve(batch);

    char* b_base = static_cast<char*>(mat_b.data_ptr());
    char* a_base = static_cast<char*>(mat_a.data_ptr());
    char* out_base = static_cast<char*>(out_contig.data_ptr());
    const int64_t b_elem = mat_b.element_size();
    const int64_t a_elem = mat_a.element_size();
    const int64_t out_elem = out_contig.element_size();
    const int64_t m_size = mat_a.size(1);
    const int64_t k_size = mat_a.size(2);
    const int64_t gemm_m = mat_b.size(2);
    const int64_t ldc_out = out_contig.size(2);
    const int64_t lda = blas_leading_dim(
        b3d_layout, mat_b.stride(1), mat_b.stride(2));
    const int64_t ldb = blas_leading_dim(
        a3d_layout, mat_a.stride(1), mat_a.stride(2));
    const hipblasOperation_t raw_opA =
        b3d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    const hipblasOperation_t raw_opB =
        a3d_layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    for (int i = 0; i < batch; ++i) {
      raw_a.push_back(b_base + i * mat_b.stride(0) * b_elem);
      raw_b.push_back(a_base + i * mat_a.stride(0) * a_elem);
      raw_out.push_back(out_base + i * out_contig.stride(0) * out_elem);
      raw_bias.push_back(bias_ptr);
      m_vec.push_back(gemm_m);
      n_vec.push_back(m_size);
      k_vec.push_back(k_size);
      batch_vec.push_back(1);
      lda_vec.push_back(lda);
      ldb_vec.push_back(ldb);
      ldc_vec.push_back(ldc_out);
      ldd_vec.push_back(ldc_out);
      strideA_vec.push_back(0);
      strideB_vec.push_back(0);
      strideC_vec.push_back(0);
      strideD_vec.push_back(0);
    }

    if (!m_vec.empty()) {
      hipStream_t stream = c10::hip::getCurrentHIPStream();
      hipblaslt_groupedgemm_raw(
          raw_a, raw_b, raw_out, raw_bias,
          use_bias,
          mat_b.scalar_type(),
          mat_a.scalar_type(),
          out_contig.scalar_type(),
          use_bias ? bias_opt->scalar_type() : zero_bias_per_device.scalar_type(),
          m_vec, n_vec, k_vec, batch_vec,
          raw_opA, raw_opB,
          false, stream,
          &lda_vec, &ldb_vec, &ldc_vec, &ldd_vec,
          &strideA_vec, &strideB_vec, &strideC_vec, &strideD_vec);
    }

    if (need_out_copy) {
      out.copy_(out_contig);
    }
    return;
  }

  // Fallback: contiguous + raw pointers for any other dimension combination
  {
    at::Tensor mat_a_contig = mat_a.contiguous();
    at::Tensor mat_b_contig = mat_b.contiguous();
    at::Tensor out_contig = out.contiguous();
    bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

    int64_t batch = mat_a_contig.dim() >= 3 ? mat_a_contig.size(0) : 1;
    int64_t m = mat_a_contig.size(-2);
    int64_t k_inner = mat_a_contig.size(-1);
    int64_t n = mat_b_contig.size(-1);

    raw_a.reserve(batch);
    raw_b.reserve(batch);
    raw_out.reserve(batch);
    raw_bias.reserve(batch);
    m_vec.reserve(batch);
    n_vec.reserve(batch);
    k_vec.reserve(batch);
    batch_vec.reserve(batch);
    lda_vec.reserve(batch);
    ldb_vec.reserve(batch);
    ldc_vec.reserve(batch);
    ldd_vec.reserve(batch);
    strideA_vec.reserve(batch);
    strideB_vec.reserve(batch);
    strideC_vec.reserve(batch);
    strideD_vec.reserve(batch);

    char* b_base = static_cast<char*>(mat_b_contig.data_ptr());
    char* a_base = static_cast<char*>(mat_a_contig.data_ptr());
    char* out_base = static_cast<char*>(out_contig.data_ptr());
    const int64_t b_elem = mat_b_contig.element_size();
    const int64_t a_elem = mat_a_contig.element_size();
    const int64_t out_elem = out_contig.element_size();

    for (int i = 0; i < batch; ++i) {
      const int64_t a_stride0 = mat_a_contig.dim() >= 3 ? mat_a_contig.stride(0) : 0;
      const int64_t b_stride0 = mat_b_contig.dim() >= 3 ? mat_b_contig.stride(0) : 0;
      const int64_t out_stride0 = out_contig.dim() >= 3 ? out_contig.stride(0) : 0;

      raw_a.push_back(b_base + i * b_stride0 * b_elem);
      raw_b.push_back(a_base + i * a_stride0 * a_elem);
      raw_out.push_back(out_base + i * out_stride0 * out_elem);
      raw_bias.push_back(bias_ptr);
      m_vec.push_back(n);
      n_vec.push_back(m);
      k_vec.push_back(k_inner);
      batch_vec.push_back(1);
      lda_vec.push_back(mat_b_contig.stride(-2));
      ldb_vec.push_back(mat_a_contig.stride(-2));
      ldc_vec.push_back(out_contig.stride(-2));
      ldd_vec.push_back(out_contig.stride(-2));
      strideA_vec.push_back(0);
      strideB_vec.push_back(0);
      strideC_vec.push_back(0);
      strideD_vec.push_back(0);
    }

    if (!m_vec.empty()) {
      hipStream_t stream = c10::hip::getCurrentHIPStream();
      hipblaslt_groupedgemm_raw(
          raw_a, raw_b, raw_out, raw_bias,
          use_bias,
          mat_b_contig.scalar_type(),
          mat_a_contig.scalar_type(),
          out_contig.scalar_type(),
          use_bias ? bias_opt->scalar_type() : zero_bias_per_device.scalar_type(),
          m_vec, n_vec, k_vec, batch_vec,
          HIPBLAS_OP_N, HIPBLAS_OP_N,
          false, stream,
          &lda_vec, &ldb_vec, &ldc_vec, &ldd_vec,
          &strideA_vec, &strideB_vec, &strideC_vec, &strideD_vec);
    }

    if (need_out_copy) {
      out.copy_(out_contig);
    }
  }
}

} // namespace detail
} // namespace hip
} // namespace at

namespace {

void groupedgemm_op_impl(
    const c10::List<at::Tensor>& group_a,
    const c10::List<at::Tensor>& group_b,
    const c10::List<at::Tensor>& group_out,
    const c10::optional<c10::List<at::Tensor>>& bias,
    bool use_bias,
    c10::List<int64_t> m,
    c10::List<int64_t> n,
    c10::List<int64_t> k,
    c10::List<int64_t> batch,
    int64_t transa,
    int64_t transb,
    bool accumulate) {

  int64_t num_groups = group_a.size();

  std::vector<at::Tensor> a_vec, b_vec, out_vec;
  a_vec.reserve(num_groups);
  b_vec.reserve(num_groups);
  out_vec.reserve(num_groups);

  std::vector<int64_t> m_vec, n_vec, k_vec, batch_vec;
  m_vec.reserve(num_groups);
  n_vec.reserve(num_groups);
  k_vec.reserve(num_groups);
  batch_vec.reserve(num_groups);

  std::vector<std::optional<at::Tensor>> bias_vec;
  bias_vec.reserve(num_groups);

  for (int64_t i = 0; i < num_groups; ++i) {
    a_vec.push_back(group_a.get(i));
    b_vec.push_back(group_b.get(i));
    out_vec.push_back(group_out.get(i));
    m_vec.push_back(m.get(i));
    n_vec.push_back(n.get(i));
    k_vec.push_back(k.get(i));
    batch_vec.push_back(batch.get(i));

    if (use_bias) {
      TORCH_CHECK(bias.has_value(),
                  "hipblaslt::groupedgemm: use_bias=True but bias is None");
      bias_vec.push_back(bias->get(i));
    } else {
      bias_vec.emplace_back();
    }
  }

  if (!use_bias) {
    static at::Tensor zero_bias_;
    if (zero_bias_.numel() == 0) {
      zero_bias_ = at::zeros(
          {1024}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    }
    for (int64_t i = 0; i < num_groups; ++i) {
      bias_vec[i] = zero_bias_;
    }
  }

  hipblasOperation_t opA = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t opB = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  hipStream_t stream = c10::hip::getCurrentHIPStream();

  at::hip::detail::hipblaslt_groupedgemm(
      a_vec, b_vec, out_vec, bias_vec, use_bias,
      m_vec, n_vec, k_vec, batch_vec,
      opA, opB,
      accumulate, stream);
}

}

TORCH_LIBRARY(hipblaslt, m) {
  m.def("groupedgemm(Tensor[] group_a, Tensor[] group_b, "
        "Tensor(a!)[] group_out, Tensor[]? bias, bool use_bias, "
        "int[] m, int[] n, int[] k, int[] batch, "
        "int transa=0, int transb=0, bool accumulate=False) -> ()");
}

TORCH_LIBRARY_IMPL(hipblaslt, CUDA, m) {
  m.impl("groupedgemm", TORCH_FN(groupedgemm_op_impl));
}

#undef TORCH_HIPBLASLT_CHECK
#undef TORCH_HIP_CHECK
