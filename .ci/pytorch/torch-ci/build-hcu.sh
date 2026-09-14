#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RESOURCE_SERVER="${RESOURCE_SERVER:-http://42.228.13.241:18000}"
INSTALL_DIR_URL="${INSTALL_DIR_URL:-${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.4.1/install_dir.zip}"
THIRD_PARTY_URL="${THIRD_PARTY_URL:-${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip}"
FBJNI_URL="${FBJNI_URL:-${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.12/fbjni.zip}"
MAGMA_URL="${MAGMA_URL:-${RESOURCE_SERVER}/thirdparty/pytorch/magma-install.tar.gz}"
INSTALL_DIR_SHA256="bb71aa69c003b2f9e3ee6143e9a2d6176c58bc352fa7d3e1fdc18f0cdd227589"
THIRD_PARTY_SHA256="61e5c3eacb5cc02e5f97336591f2f243296df1f59c1f043d9e90d87b22c084ce"
FBJNI_SHA256="142a7609f9df056a33528ee085d2b98d709a61e23f4e3a9e53e6948a85d64408"
MAGMA_SHA256="facbada3e2e8d366a1931627521e16484328abbec91c997500506dc51d59d2d8"
BUILD_PIP_INDEX_URL="${BUILD_PIP_INDEX_URL:-http://10.16.1.201:9929/nightly/dtk2604/+simple/}"
BUILD_PIP_TRUSTED_HOST="${BUILD_PIP_TRUSTED_HOST:-10.16.1.201}"
DTK_PKG_URL="${DTK_PKG_URL:?DTK_PKG_URL is required}"
DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
ROCM_PATH="${DTK_ROOT}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx906;gfx926;gfx928;gfx936;gfx938}"
CONFIGURED_AMDGPU_TARGETS="${AMDGPU_TARGETS}"
RESOURCE_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-hcu.XXXXXX")"
trap 'rm -rf -- "${RESOURCE_TMP_DIR}"' EXIT

MAGMA_ENABLED=1
for arg in "$@"; do
  case "${arg}" in
    --no-magma) MAGMA_ENABLED=0 ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

download_file() {
  local source="$1"
  local destination="$2"
  local expected_sha256="${3:-}"
  local actual_sha256

  if [[ "${source}" =~ ^https?:// ]]; then
    wget --quiet --show-progress --tries=3 --timeout=120 \
      --output-document="${destination}" -- "${source}"
  else
    cp -- "${source}" "${destination}"
  fi
  [[ -s "${destination}" ]] || {
    echo "ERROR: downloaded resource is empty: ${source}" >&2
    return 1
  }
  if [[ -n "${expected_sha256}" ]]; then
    actual_sha256="$(sha256sum -- "${destination}" | awk '{print $1}')"
    [[ "${actual_sha256}" == "${expected_sha256}" ]] || {
      echo "ERROR: SHA256 mismatch for ${source}" >&2
      echo "expected=${expected_sha256}" >&2
      echo "actual=${actual_sha256}" >&2
      return 1
    }
  fi
}

extract_zip() {
  local source="$1"
  local destination="$2"
  local archive="${RESOURCE_TMP_DIR}/$3"
  local expected_sha256="$4"

  download_file "${source}" "${archive}" "${expected_sha256}"
  unzip -q -o "${archive}" -d "${destination}"
}

pip_install_build() {
  python3 -m pip install \
    --index-url "${BUILD_PIP_INDEX_URL}" \
    --trusted-host "${BUILD_PIP_TRUSTED_HOST}" \
    --default-timeout 1000 \
    "$@"
}

install_dtk() {
  local archive="${RESOURCE_TMP_DIR}/dtk.tar.gz"
  local extract_dir="${RESOURCE_TMP_DIR}/dtk"
  local version_file
  local extracted_root
  local version

  if [[ -f "${DTK_ROOT}/env.sh" && -f "${DTK_ROOT}/.info/rocm_version" ]]; then
    version="$(tr -d '[:space:]' < "${DTK_ROOT}/.info/rocm_version")"
    if [[ "${version}" == 26.04* ]]; then
      echo "========== Using existing DTK ${version} =========="
      return
    fi
  fi

  echo "========== Installing DTK =========="
  mkdir -p "${extract_dir}"
  download_file "${DTK_PKG_URL}" "${archive}"
  tar -xzf "${archive}" -C "${extract_dir}"
  version_file="$(find "${extract_dir}" -type f -path '*/.info/rocm_version' -print -quit)"
  [[ -n "${version_file}" ]] || {
    echo "ERROR: DTK package does not contain .info/rocm_version" >&2
    return 1
  }
  extracted_root="${version_file%/.info/rocm_version}"
  version="$(tr -d '[:space:]' < "${version_file}")"
  [[ "${version}" == 26.04* ]] || {
    echo "ERROR: expected DTK 26.04, found ${version}" >&2
    return 1
  }
  rm -rf -- "${DTK_ROOT}"
  mkdir -p "${DTK_ROOT}"
  cp -a "${extracted_root}/." "${DTK_ROOT}/"
}

find_mkl_library_dir() {
  local candidate
  for candidate in \
    "${MKL_LIBRARY_DIR:-}" \
    /opt/intel/mkl/lib/intel64_lin \
    /opt/intel/mkl/lib/intel64 \
    /opt/intel/oneapi/mkl/latest/lib \
    /opt/intel/oneapi/mkl/latest/lib/intel64 \
    /opt/intel/oneapi/mkl/*/lib \
    /opt/intel/oneapi/mkl/*/lib/intel64
  do
    if [[ -n "${candidate}" && -d "${candidate}" && \
          ( -e "${candidate}/libmkl_core.so" || -e "${candidate}/libmkl_core.so.2" ) ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

copy_mkl_component() {
  local base_name="$1"
  local destination="$2"
  local unversioned="${base_name}.so"
  local versioned="${base_name}.so.2"

  mkdir -p "${destination}"
  if [[ -e "${MKL_LIBRARY_DIR}/${versioned}" ]]; then
    cp -Lv -- "${MKL_LIBRARY_DIR}/${versioned}" "${destination}/${versioned}"
    ln -sfn "${versioned}" "${destination}/${unversioned}"
  elif [[ -e "${MKL_LIBRARY_DIR}/${unversioned}" ]]; then
    cp -Lv -- "${MKL_LIBRARY_DIR}/${unversioned}" "${destination}/${versioned}"
    ln -sfn "${versioned}" "${destination}/${unversioned}"
  else
    echo "ERROR: missing MKL component: ${unversioned}[.2]" >&2
    return 1
  fi
}

cd "${PYTORCH_ROOT}"
[[ "$(tr -d '[:space:]' < version.txt)" == 2.12.0 ]] || {
  echo "ERROR: this build script requires PyTorch 2.12.0" >&2
  exit 1
}

install_dtk
set +u
source "${DTK_ROOT}/env.sh"
if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
  source /opt/rh/gcc-toolset-13/enable
fi
set -u

ROCM_PATH="${DTK_ROOT}"
AMDGPU_TARGETS="${CONFIGURED_AMDGPU_TARGETS}"
export DTK_ROOT ROCM_PATH AMDGPU_TARGETS
export HIP_PATH="${ROCM_PATH}/hip"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export PYTORCH_CUSTOM_SUFFIX=''
export PATH="/opt/rh/gcc-toolset-13/root/usr/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${CPLUS_INCLUDE_PATH:-}"
export LDFLAGS="${LDFLAGS:-} -Wl,--copy-dt-needed-entries"

export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
[[ -x "${CC}" && -x "${CXX}" ]] || {
  echo "ERROR: GCC toolset 13 is missing" >&2
  exit 1
}

HIP_COMPILER=""
for candidate in "${ROCM_PATH}/llvm/bin/clang++" "${ROCM_PATH}/bin/clang++"; do
  if [[ -x "${candidate}" ]]; then
    HIP_COMPILER="${candidate}"
    break
  fi
done
[[ -n "${HIP_COMPILER}" ]] || {
  echo "ERROR: DTK clang++ was not found" >&2
  exit 1
}
export HIPCXX="${HIP_COMPILER}"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_HIP_COMPILER=${HIP_COMPILER} -DCMAKE_HIP_COMPILER_ROCM_ROOT=${ROCM_PATH} -DCMAKE_PREFIX_PATH=${ROCM_PATH}"

echo "========== Installing build dependencies =========="
pip_install_build --upgrade pip
pip_install_build -r .ci/docker/requirements-ci.txt
pip_install_build -r requirements.txt
pip_install_build \
  xdoctest==1.1.3 \
  pytest==8.1.1 \
  pytest-timeout \
  Pillow==10.0.1 \
  mypy==1.8.0 \
  jmespath==1.0.1 \
  Jinja2==3.1.3 \
  expecttest==0.2.1 \
  tlparse==0.3.5 \
  numba==0.61.2 \
  sympy==1.13.3 \
  MonkeyType \
  auditwheel \
  patchelf

echo "========== Preparing third-party sources =========="
extract_zip "${INSTALL_DIR_URL}" "${PYTORCH_ROOT}" install_dir.zip "${INSTALL_DIR_SHA256}"
extract_zip "${THIRD_PARTY_URL}" "${PYTORCH_ROOT}" third_party.zip "${THIRD_PARTY_SHA256}"
extract_zip "${FBJNI_URL}" "${PYTORCH_ROOT}/android/libs" fbjni.zip "${FBJNI_SHA256}"

echo "========== Preparing MAGMA =========="
if [[ "${MAGMA_ENABLED}" == 0 ]]; then
  echo "MAGMA disabled by --no-magma"
elif [[ ! -d "${ROCM_PATH}/magma" ]]; then
  magma_stage="${RESOURCE_TMP_DIR}/magma"
  mkdir -p "${magma_stage}"
  download_file "${MAGMA_URL}" "${magma_stage}/magma.tar.gz" "${MAGMA_SHA256}"
  tar -xzf "${magma_stage}/magma.tar.gz" -C "${magma_stage}"
  [[ -d "${magma_stage}/magma-install" ]] || {
    echo "ERROR: MAGMA archive does not contain magma-install" >&2
    exit 1
  }
  mv -- "${magma_stage}/magma-install" "${ROCM_PATH}/magma"
fi

echo "========== Preparing MKL =========="
MKL_LIBRARY_DIR="$(find_mkl_library_dir)" || {
  echo "ERROR: no usable MKL runtime was found" >&2
  exit 1
}
export MKL_LIBRARY_DIR
if [[ "${MKL_LIBRARY_DIR}" == /opt/intel/oneapi/* ]]; then
  export MKLROOT="${MKL_LIBRARY_DIR%/lib*}"
else
  export MKLROOT=/opt/intel/mkl
fi
export LD_LIBRARY_PATH="${MKL_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
if [[ "${MAGMA_ENABLED}" == 1 ]]; then
  export LD_LIBRARY_PATH="${ROCM_PATH}/magma/lib:${LD_LIBRARY_PATH}"
fi

mkl_components=(
  libmkl_core
  libmkl_gf_lp64
  libmkl_gnu_thread
  libmkl_def
  libmkl_intel_lp64
  libmkl_vml_def
)
for component in "${mkl_components[@]}"; do
  copy_mkl_component "${component}" "${PYTORCH_ROOT}/torch/lib"
done
if [[ "${MAGMA_ENABLED}" == 1 ]]; then
  for component in "${mkl_components[@]}"; do
    copy_mkl_component "${component}" "${ROCM_PATH}/magma/lib"
  done
  [[ -e "${ROCM_PATH}/magma/lib/libmagma.so" ]] || {
    echo "ERROR: MAGMA runtime is missing" >&2
    exit 1
  }
  cp -Lv -- "${ROCM_PATH}/magma/lib/libmagma.so" "${PYTORCH_ROOT}/torch/lib/libmagma.so"
fi

if [[ -f third_party/gloo/tools/amd_build/build_amd.py ]]; then
  chmod 755 third_party/gloo/tools/amd_build/build_amd.py
fi
python3 tools/hcu_build/build_hcu.py

echo "========== Building PyTorch wheel =========="
rm -rf -- build dist ./*.egg-info torch.egg-info
MAGMA_BUILD_OPTION=OFF
if [[ "${MAGMA_ENABLED}" == 1 ]]; then
  MAGMA_BUILD_OPTION=ON
fi
build_env=(
  "CMAKE_FRESH=1"
  "ROCM_PATH=${ROCM_PATH}"
  "USE_CUDA=0"
  "USE_KINETO=1"
  "BUILD_TEST=ON"
  "USE_MAGMA=${MAGMA_BUILD_OPTION}"
  "NO_CAFFE2_OPS=1"
  "USE_NCCL=1"
  "USE_RCCL=1"
  "USE_DISTRIBUTED=1"
  "USE_GLOO_IBVERBS=1"
  "USE_GLOO=1"
  "USE_IBVERBS=1"
  "USE_C10D_MPI=1"
  "USE_MPI=1"
  "USE_OPENMP=1"
  "USE_MIOPEN=1"
  "USE_ROCM=1"
  "USE_LMDB=1"
  "USE_OPENCV=1"
  "USE_NUMPY=1"
  "USE_GLOG=1"
  "USE_GFLAGS=1"
  "USE_STATIC_MKL=0"
  "PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS}"
  "USE_ROCM_CK_GEMM=0"
  "USE_ROCM_CK_SDPA=0"
  "USE_CUFILE=0"
  "USE_AOTRITON_FA=0"
  "USE_MEM_EFF_ATTENTION=0"
  "USE_FLASH_ATTENTION=1"
  "USE_FBGEMM=ON"
  "USE_MKLDNN=1"
  "USE_MKL=ON"
  "USE_NINJA=ON"
)
env "${build_env[@]}" python3 -m build --wheel --no-isolation

mapfile -t built_wheels < <(find dist -maxdepth 1 -type f -name '*-linux_x86_64.whl' -print)
[[ "${#built_wheels[@]}" -eq 1 ]] || {
  echo "ERROR: expected one built Linux wheel, found ${#built_wheels[@]}" >&2
  exit 1
}
python3 -m pip install --no-deps --force-reinstall "${built_wheels[0]}"

echo "========== Generating manylinux wheel =========="
mapfile -t linux_wheels < <(find dist -maxdepth 1 -type f -name '*-linux_x86_64.whl' -print)
[[ "${#linux_wheels[@]}" -eq 1 ]] || {
  echo "ERROR: expected one Linux wheel, found ${#linux_wheels[@]}" >&2
  exit 1
}
glibc_version="$(ldd --version | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)"
[[ -n "${glibc_version}" ]] || {
  echo "ERROR: unable to determine glibc version" >&2
  exit 1
}
platform="manylinux_${glibc_version/./_}_x86_64"
TORCH_LOCATION="$(python3 -m pip show torch | awk -F': ' '/^Location:/ {print $2}')"
[[ -n "${TORCH_LOCATION}" ]] || {
  echo "ERROR: the newly built torch package is not installed" >&2
  exit 1
}
export LD_LIBRARY_PATH="${TORCH_LOCATION}/torch/lib:${MKL_LIBRARY_DIR}:${LD_LIBRARY_PATH:-}"
if [[ "${MAGMA_ENABLED}" == 1 ]]; then
  export LD_LIBRARY_PATH="${ROCM_PATH}/magma/lib:${LD_LIBRARY_PATH}"
fi
auditwheel_excludes=(
  libgalaxyhip.so.5
  libMIOpen.so.1
  librccl.so.1
  libhipblas.so.0
  libhipfft.so
  libhiprand.so.1
  libhipsolver.so.0
  libhipsparse.so.0
  libhipnn.so
  librocblas.so.0
  librocsolver.so.0
  librocfft.so.0
  librocrand.so.1
  librocsparse.so.0
  librocm_smi64.so.2
  librocfft-device-0.so.0
  librocfft-device-1.so.0
  librocfft-device-2.so.0
  librocfft-device-3.so.0
  libc10.so
  libc10_hip.so
  libtorch_cpu.so
  libtorch_hip.so
  libtorch_python.so
  libtorch.so
  libtorchaudio_ffmpeg4.so
  libtorchaudio_ffmpeg5.so
  libtorchaudio_ffmpeg6.so
  libtorchaudio.so
  libtorchaudio_sox.so
  _torchaudio_ffmpeg4.so
  _torchaudio_ffmpeg5.so
  _torchaudio_ffmpeg6.so
  _torchaudio.so
  _torchaudio_sox.so
  libavcodec.so.58
  libavcodec.so.59
  libavcodec.so.60
  libavdevice.so.58
  libavdevice.so.59
  libavdevice.so.60
  libavfilter.so.7
  libavfilter.so.8
  libavfilter.so.9
  libavformat.so.58
  libavformat.so.59
  libavformat.so.60
  libavutil.so.56
  libavutil.so.57
  libavutil.so.58
  libsox.so
  libomp.so
  libhipblaslt.so.0
  libhipblas.so.2
  libhipfft.so.0
  libhipsparse.so.1
  librocblas.so.4
  librocsparse.so.1
  libnuma.so.1
)
auditwheel_args=()
for library in "${auditwheel_excludes[@]}"; do
  auditwheel_args+=(--exclude "${library}")
done
auditwheel repair \
  --plat "${platform}" \
  --strip \
  "${auditwheel_args[@]}" \
  --wheel-dir dist \
  "${linux_wheels[0]}"

mapfile -t final_wheels < <(find dist -maxdepth 1 -type f -name '*.whl' -print)
[[ "${#final_wheels[@]}" -eq 2 ]] || {
  echo "ERROR: expected Linux and manylinux wheels, found ${#final_wheels[@]}" >&2
  exit 1
}
printf 'wheel: %s\n' "${final_wheels[@]}"
echo "========== Build complete =========="
