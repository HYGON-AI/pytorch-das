#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#if defined(USE_ROCM)
#include <amd_smi/amdsmi.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#endif

namespace c10d::intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};
static int intraNodeCommIdx = 0;

#if defined(USE_ROCM)
namespace {

constexpr uint32_t kHcusmiStatusSuccess = 0;
constexpr uint32_t kHcusmiStatusNotInit = 32;
constexpr uint64_t kHcusmiInitGpus = 2;

struct HcusmiApi {
  void* library = nullptr;
  uint32_t (*init)(uint64_t) = nullptr;
  uint32_t (*get_processor_handle_from_bdf)(amdsmi_bdf_t, void**) = nullptr;
  uint32_t (*is_P2P_accessible)(void*, void*, bool*) = nullptr;

  explicit operator bool() const {
    return library != nullptr && init != nullptr &&
        get_processor_handle_from_bdf != nullptr &&
        is_P2P_accessible != nullptr;
  }
};

static_assert(sizeof(amdsmi_bdf_t) == sizeof(uint64_t));

HcusmiApi loadHcusmiApi() {
  const char* path = std::getenv("TORCH_HCUSMI_LIB_PATH");
  if (path == nullptr || path[0] == '\0') {
    return {};
  }

  HcusmiApi api{};
  api.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (api.library == nullptr) {
    LOG(WARNING) << "IntraNodeComm: failed to load HCuSMI library " << path
                 << ": " << dlerror() << "; falling back to AMD SMI";
    return {};
  }

  api.init = reinterpret_cast<decltype(api.init)>(
      dlsym(api.library, "hcusmi_init"));
  api.get_processor_handle_from_bdf =
      reinterpret_cast<decltype(api.get_processor_handle_from_bdf)>(
          dlsym(api.library, "hcusmi_get_processor_handle_from_bdf"));
  api.is_P2P_accessible =
      reinterpret_cast<decltype(api.is_P2P_accessible)>(
          dlsym(api.library, "hcusmi_is_P2P_accessible"));
  if (!api) {
    LOG(WARNING) << "IntraNodeComm: failed to resolve required HCuSMI symbols; "
                    "falling back to AMD SMI";
    dlclose(api.library);
    return {};
  }
  return api;
}

std::optional<amdsmi_bdf_t> getHcusmiBdf(int deviceIdx) {
  char pciBusId[32] = {};
  const auto status =
      hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), deviceIdx);
  if (status != hipSuccess) {
    LOG(WARNING) << "IntraNodeComm: hipDeviceGetPCIBusId failed for device "
                 << deviceIdx << ": " << hipGetErrorString(status)
                 << "; falling back to AMD SMI";
    return std::nullopt;
  }

  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  if (std::sscanf(
          pciBusId, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4 ||
      bus > 0xff || device > 0x1f || function > 0x7) {
    LOG(WARNING) << "IntraNodeComm: invalid HIP PCI bus ID " << pciBusId
                 << "; falling back to AMD SMI";
    return std::nullopt;
  }

  amdsmi_bdf_t bdf{};
  bdf.as_uint = (static_cast<uint64_t>(domain) << 16) |
      (static_cast<uint64_t>(bus) << 8) |
      (static_cast<uint64_t>(device) << 3) | function;
  return bdf;
}

std::optional<NvlMesh> tryGetHcusmiMesh(
    const std::vector<int>& rankToDeviceIdx) {
  if (rankToDeviceIdx.size() > kMaxDevices) {
    return std::nullopt;
  }

  static const HcusmiApi hcusmi = loadHcusmiApi();
  if (!hcusmi) {
    return std::nullopt;
  }
  static std::mutex hcusmiMutex;
  const std::lock_guard<std::mutex> lock(hcusmiMutex);

  std::vector<void*> processorHandles;
  processorHandles.reserve(rankToDeviceIdx.size());
  for (const auto deviceIdx : rankToDeviceIdx) {
    const auto bdf = getHcusmiBdf(deviceIdx);
    if (!bdf.has_value()) {
      return std::nullopt;
    }

    void* processorHandle = nullptr;
    auto status =
        hcusmi.get_processor_handle_from_bdf(*bdf, &processorHandle);
    if (status == kHcusmiStatusNotInit) {
      status = hcusmi.init(kHcusmiInitGpus);
      if (status == kHcusmiStatusSuccess) {
        status =
            hcusmi.get_processor_handle_from_bdf(*bdf, &processorHandle);
      }
    }
    if (status != kHcusmiStatusSuccess || processorHandle == nullptr) {
      LOG(WARNING) << "IntraNodeComm: HCuSMI failed to resolve device "
                   << deviceIdx << " by BDF, ret=" << status
                   << "; falling back to AMD SMI";
      return std::nullopt;
    }
    processorHandles.push_back(processorHandle);
  }

  NvlMesh nvlMesh{};
  for (size_t srcRank = 0; srcRank < processorHandles.size(); ++srcRank) {
    for (size_t dstRank = 0; dstRank < processorHandles.size(); ++dstRank) {
      if (srcRank == dstRank) {
        continue;
      }
      bool accessible = false;
      const auto status = hcusmi.is_P2P_accessible(
          processorHandles[srcRank], processorHandles[dstRank], &accessible);
      if (status != kHcusmiStatusSuccess) {
        LOG(WARNING) << "IntraNodeComm: HCuSMI P2P query failed, ret="
                     << status << "; falling back to AMD SMI";
        return std::nullopt;
      }
      nvlMesh[srcRank][dstRank] = accessible ? 1 : 0;
    }
  }
  LOG(INFO) << "IntraNodeComm: using HCuSMI for ROCm topology";
  return nvlMesh;
}

} // namespace
#endif

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<int>& rankToDeviceIdx) {
#if !defined(USE_ROCM)
  auto connectivity = detect_dma_connectivity(c10::DeviceType::CUDA, "nvlink");
  NvlMesh nvlMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() &&
          dstRank < rankToDeviceIdx.size()) {
        nvlMesh[srcRank][dstRank] =
            connectivity
                ->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return nvlMesh;
#else
  if (auto hcusmiMesh = tryGetHcusmiMesh(rankToDeviceIdx)) {
    return *hcusmiMesh;
  }

  // Load libamd_smi at runtime to avoid linking it into torch_hip (double-load
  // with Python amdsmi causes bus errors). Types/constants from amdsmi.h only.
  struct AmdsmiApi {
    amdsmi_status_t (*init)(uint64_t);
    amdsmi_status_t (*get_socket_handles)(uint32_t*, amdsmi_socket_handle*);
    amdsmi_status_t (*get_processor_handles)(
        amdsmi_socket_handle,
        uint32_t*,
        amdsmi_processor_handle*);
    amdsmi_status_t (*is_P2P_accessible)(
        amdsmi_processor_handle,
        amdsmi_processor_handle,
        bool*);
  };
  static void* amdsmi_handle = nullptr;
  static AmdsmiApi amdsmi = {};
  static bool amdsmi_resolved = false;

  if (!amdsmi_resolved) {
    amdsmi_resolved = true;
    const char* rocm = std::getenv("ROCM_PATH");
    std::string path =
        rocm ? std::string(rocm) + "/lib/libamd_smi.so" : "libamd_smi.so";
    amdsmi_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!amdsmi_handle) {
      amdsmi_handle = dlopen("libamd_smi.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!amdsmi_handle) {
      LOG(ERROR) << "IntraNodeComm:: getNvlMesh: dlopen libamd_smi.so failed: "
                 << dlerror();
      return {};
    }
    amdsmi.init = reinterpret_cast<decltype(amdsmi.init)>(
        dlsym(amdsmi_handle, "amdsmi_init"));
    amdsmi.get_socket_handles =
        reinterpret_cast<decltype(amdsmi.get_socket_handles)>(
            dlsym(amdsmi_handle, "amdsmi_get_socket_handles"));
    amdsmi.get_processor_handles =
        reinterpret_cast<decltype(amdsmi.get_processor_handles)>(
            dlsym(amdsmi_handle, "amdsmi_get_processor_handles"));
    amdsmi.is_P2P_accessible =
        reinterpret_cast<decltype(amdsmi.is_P2P_accessible)>(
            dlsym(amdsmi_handle, "amdsmi_is_P2P_accessible"));
    if (!amdsmi.init || !amdsmi.get_socket_handles ||
        !amdsmi.get_processor_handles || !amdsmi.is_P2P_accessible) {
      LOG(ERROR) << "IntraNodeComm:: getNvlMesh: dlsym amdsmi failed";
      return {};
    }
  }

  NvlMesh nvlMesh = {};
  const auto worldSize = rankToDeviceIdx.size();

  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi.get_socket_handles(&socket_count, nullptr);
  if (ret == AMDSMI_STATUS_NOT_INIT) {
    ret = amdsmi.init(AMDSMI_INIT_AMD_GPUS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      LOG(ERROR) << "IntraNodeComm:: getNvlMesh: amdsmi_init failed, ret="
                 << static_cast<int>(ret);
      return {};
    }
    socket_count = 0;
    ret = amdsmi.get_socket_handles(&socket_count, nullptr);
  }
  if (ret != AMDSMI_STATUS_SUCCESS) {
    LOG(ERROR)
        << "IntraNodeComm:: getNvlMesh: amdsmi_get_socket_handles failed, ret="
        << static_cast<int>(ret);
    return {};
  }

  std::vector<amdsmi_socket_handle> socket_handles(socket_count);
  ret = amdsmi.get_socket_handles(&socket_count, &socket_handles[0]);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    LOG(ERROR)
        << "IntraNodeComm:: getNvlMesh: amdsmi_get_socket_handles (buffer) failed, ret="
        << static_cast<int>(ret);
    return {};
  }

  std::vector<amdsmi_processor_handle> processor_handles;
  for (size_t i = 0; i < socket_count; ++i) {
    uint32_t device_count = 0;
    ret =
        amdsmi.get_processor_handles(socket_handles[i], &device_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      LOG(ERROR)
          << "IntraNodeComm:: getNvlMesh: amdsmi_get_processor_handles (count) failed, ret="
          << static_cast<int>(ret);
      return {};
    }
    std::vector<amdsmi_processor_handle> _processor_handles(device_count);
    ret = amdsmi.get_processor_handles(
        socket_handles[i], &device_count, &_processor_handles[0]);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      LOG(ERROR)
          << "IntraNodeComm:: getNvlMesh: amdsmi_get_processor_handles (buffer) failed, ret="
          << static_cast<int>(ret);
      return {};
    }
    processor_handles.insert(
        processor_handles.end(),
        _processor_handles.begin(),
        _processor_handles.end());
  }

  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxDevices; ++link) {
      if (idx == link)
        continue;
      bool conn = false;
      ret = amdsmi.is_P2P_accessible(
          processor_handles[idx], processor_handles[link], &conn);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        LOG(ERROR)
            << "IntraNodeComm: getNvlMesh: amdsmi_is_P2P_accessible failed, ret="
            << static_cast<int>(ret);
        return {};
      }
      if (conn) {
        nvlMesh[idx][link] += 1;
      }
    }
  }
  return nvlMesh;
#endif
}

/**
 * Detect topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
}

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize,
    std::string groupName)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize),
      groupName_(std::move(groupName)) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(symmetricMemoryPtr_);
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << '-' << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  deviceIdx_ = at::cuda::current_device();

  // Exchange hostname and device bus ID
  struct DevInfo {
    // NOLINTNEXTLINE
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ')';
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  {
    std::unordered_set uniqueDeviceIdxs(
        rankToDeviceIdx.begin(), rankToDeviceIdx.end());
    if (uniqueDeviceIdxs.size() != worldSize_) {
      LOG(WARNING)
          << "Skipping IntraNodeComm::rendezvous() because participants have "
             "overlapping devices. To resolve this, call torch.cuda.set_device() "
             "before init_process_group().";
      return false;
    }
  }

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(rankToDeviceIdx);

  // Detect topology
  topology_ = detectTopology(nvlMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }

  const std::string name = groupName_.empty()
      ? "IntraNodeComm" + std::to_string(intraNodeCommIdx++)
      : groupName_;
  set_group_info(
      name, static_cast<int>(rank_), static_cast<int>(worldSize_), store_);
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  symmetricMemoryPtr_ = allocator->alloc(bufferSize_, deviceIdx_, name);
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_, std::nullopt);
  isInitialized_ = true;
  return true;
}

} // namespace c10d::intra_node_comm
