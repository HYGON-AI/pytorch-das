#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# HCU PyTorch 2.14 build.
# 手动编译前需先设置 RESOURCE_SERVER_URL 环境变量，再运行 bash scripts/build_hcu.sh。
# CI 可直接传入 MAGMA_URL。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_PIP_INDEX_URL="${BUILD_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
BUILD_PIP_TIMEOUT="${BUILD_PIP_TIMEOUT:-1000}"
DTK_ROOT="${DTK_ROOT:-${ROCM_PATH:-/opt/dtk}}"
ROCM_PATH="${DTK_ROOT}"
cd "${PYTORCH_ROOT}"

# Parse arguments.
case "${USE_MAGMA:-ON}" in
  0|OFF|off|FALSE|false) MAGMA_ENABLED=0 ;;
  *) MAGMA_ENABLED=1 ;;
esac
for arg in "$@"; do
  case "${arg}" in
    --no-magma) MAGMA_ENABLED=0 ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

# Check DTK.
if [[ ! -f "${DTK_ROOT}/env.sh" ]]; then
  echo "ERROR: DTK is not installed at ${DTK_ROOT}." >&2
  exit 1
fi
echo "========== Using DTK: ${DTK_ROOT} =========="

# Locate MKL before loading DTK so its library directory can be preserved.
MKL_LIBRARY_DIR="${MKL_LIBRARY_DIR:-}"
mkl_candidates=(
  "${MKL_LIBRARY_DIR}"
  /opt/intel/mkl/lib/intel64_lin
  /opt/intel/mkl/lib/intel64
  /opt/intel/oneapi/mkl/latest/lib
  /opt/intel/oneapi/mkl/latest/lib/intel64
  /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin
)
MKL_LIBRARY_DIR=""
for candidate in "${mkl_candidates[@]}"; do
  if [[ -n "${candidate}" \
      && ( -e "${candidate}/libmkl_core.so" || -e "${candidate}/libmkl_core.so.2" ) \
      && ( -e "${candidate}/libmkl_def.so" || -e "${candidate}/libmkl_def.so.2" ) ]]; then
    MKL_LIBRARY_DIR="${candidate}"
    break
  fi
done
if [[ -z "${MKL_LIBRARY_DIR}" ]]; then
  echo "ERROR: no usable MKL runtime was found in the build image." >&2
  exit 1
fi
export MKL_LIBRARY_DIR

PYTHON_BIN="$(command -v python3)"
PYTHON_BIN="$(readlink -f "${PYTHON_BIN}")"
IMAGE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Load the DTK environment.
echo "========== Loading DTK environment =========="
set +u
# DTK_ROOT is configurable, so ShellCheck cannot resolve this path statically.
# shellcheck disable=SC1091
source "${DTK_ROOT}/env.sh"
set -u
ROCM_PATH="${DTK_ROOT}"
: "${AMDGPU_TARGETS:?ERROR: DTK env.sh did not set AMDGPU_TARGETS}"
export DTK_ROOT ROCM_PATH AMDGPU_TARGETS
export HIP_PATH="${ROCM_PATH}/hip"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/miopen/bin:${PATH}"
HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} --gcc-toolchain=$(dirname "$(dirname "$(readlink -f "$(command -v g++)")")")"
export HIPCC_COMPILE_FLAGS_APPEND
DTK_LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${ROCM_PATH}/magma/lib:${MKL_LIBRARY_DIR}"
export LD_LIBRARY_PATH="${DTK_LD_LIBRARY_PATH}${IMAGE_LD_LIBRARY_PATH:+:${IMAGE_LD_LIBRARY_PATH}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export C_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="/opt/rh/gcc-toolset-13/root/usr/include/c++/13:/opt/rh/gcc-toolset-13/root/usr/include/c++/13/x86_64-redhat-linux:${ROCM_PATH}/rocrand/include:${CPLUS_INCLUDE_PATH:-}"

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
export PYTHON_VERSION
echo "========== Build Python: ${PYTHON_BIN} =========="
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version

# PyTorch 2.14.1 CI supports DTK 26.10 only.
DTK_BUILD_VERSION="$(tr -d '.[:space:]' < "${DTK_ROOT}/.info/rocm_version" 2>/dev/null || true)"
if [[ "${DTK_BUILD_VERSION}" != 2610* ]]; then
  echo "ERROR: PyTorch 2.14.1 CI requires DTK 26.10, found ${DTK_BUILD_VERSION:-unknown}." >&2
  exit 1
fi
pip_install_build() {
  "${PYTHON_BIN}" -m pip install \
    --index-url "${BUILD_PIP_INDEX_URL}" \
    --default-timeout "${BUILD_PIP_TIMEOUT}" \
    "$@"
}

# Install build dependencies.
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
  sympy==1.13.3 \
  MonkeyType

# Select dependencies based on the Torch version.
IFS='.' read -r MAJOR_ MINOR_ _ < version.txt
if (( MAJOR_ < 2 || (MAJOR_ == 2 && MINOR_ < 4) )); then
  pip_install_build sympy==1.12.1
fi


# Prepare third-party sources.
echo "========== Preparing third-party sources =========="
git submodule update --init --recursive
bash tools/third_party/apply_patches.sh

# Prepare MAGMA.
if [[ "${MAGMA_ENABLED}" == "0" ]]; then
  echo "========== Skipping MAGMA =========="
else
  if [[ -z "${MAGMA_URL:-}" ]]; then
    : "${RESOURCE_SERVER_URL:?ERROR: set RESOURCE_SERVER_URL or MAGMA_URL before installing MAGMA}"
    MAGMA_URL="${RESOURCE_SERVER_URL%/}/thirdparty/pytorch/magma_v2.8.0-hip_DTK26.10/MAGMA-v2.8.0_hip.dtk26.10-16Oct2026.tar.gz"
  fi
  MAGMA_SHA256="${MAGMA_SHA256:-ebd3f355981ee9f7b5405ab53a826c85741ff076821372e3eba6324e09dfd9f5}"
  echo "========== Loading MAGMA =========="
  cd "${ROCM_PATH}"
  MAGMA_ARCHIVE=magma-package.tar.gz
  wget -O "${MAGMA_ARCHIVE}" "${MAGMA_URL}"
  echo "${MAGMA_SHA256}  ${MAGMA_ARCHIVE}" | sha256sum -c -
  rm -rf magma
  tar -zxf "${MAGMA_ARCHIVE}"
  rm -f "${MAGMA_ARCHIVE}"
  if [[ ! -d magma ]]; then
    MAGMA_EXTRACT_DIR="$(find . -maxdepth 1 -type d -iname 'magma*' ! -name magma | sort | tail -n 1)"
    if [[ -z "${MAGMA_EXTRACT_DIR}" ]]; then
      echo "ERROR: cannot find extracted MAGMA directory under ${ROCM_PATH}" >&2
      exit 1
    fi
    mv "${MAGMA_EXTRACT_DIR}" magma
  fi
  cd magma/lib
  for magma_library in libmagma.so libmagma_sparse.so; do
    [[ -e "${magma_library}" ]] || continue
    if readelf -d "${magma_library}" 2>/dev/null |
       grep -Fq 'Shared library: [libhipblas.so.2]'; then
      command -v patchelf >/dev/null 2>&1 || {
        echo "ERROR: ${magma_library} needs patchelf" >&2
        exit 1
      }
      if [[ ! -e "${ROCM_PATH}/lib/libhipblas.so.3" ]]; then
        echo "ERROR: DTK 26.10 has no ${ROCM_PATH}/lib/libhipblas.so.3" >&2
        exit 1
      fi
      patchelf --replace-needed libhipblas.so.2 libhipblas.so.3 "${magma_library}"
      if readelf -d "${magma_library}" 2>/dev/null |
         grep -Fq 'Shared library: [libhipblas.so.2]'; then
        echo "ERROR: failed to update ${magma_library} hipBLAS dependency" >&2
        exit 1
      fi
    fi
  done
  cd "${PYTORCH_ROOT}"
fi

# Prepare MKL.
echo "========== Preparing MKL =========="
if [[ -z "${MKLROOT:-}" ]]; then
  case "${MKL_LIBRARY_DIR}" in
    */lib) MKLROOT="${MKL_LIBRARY_DIR%/lib}" ;;
    */lib/*) MKLROOT="${MKL_LIBRARY_DIR%%/lib/*}" ;;
  esac
fi
export MKLROOT

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
    cp -Lv -- "${MKL_LIBRARY_DIR}/${unversioned}" "${destination}/${unversioned}"
  else
    echo "ERROR: required MKL component is missing: ${unversioned}[.2]" >&2
    return 1
  fi
}

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
if [[ "${MAGMA_ENABLED}" == "1" ]]; then
  for component in "${mkl_components[@]}"; do
    copy_mkl_component "${component}" "${ROCM_PATH}/magma/lib"
  done
  if [[ ! -e "${ROCM_PATH}/magma/lib/libmagma.so" ]]; then
    echo "ERROR: MAGMA runtime is missing: ${ROCM_PATH}/magma/lib/libmagma.so" >&2
    exit 1
  fi
  cp -Lv -- "${ROCM_PATH}/magma/lib/libmagma.so" "${PYTORCH_ROOT}/torch/lib/libmagma.so"
fi

# Run hipify.
if [ -f third_party/gloo/tools/amd_build/build_amd.py ]; then
    chmod 755 third_party/gloo/tools/amd_build/build_amd.py
fi
"${PYTHON_BIN}" tools/hcu_build/build_hcu.py


# Build the wheel.
echo "========== Building PyTorch wheel =========="
rm -rf -- build dist ./*.egg-info torch.egg-info
MAGMA_BUILD_OPTION=OFF
if [[ "${MAGMA_ENABLED}" == "1" ]]; then
  MAGMA_BUILD_OPTION=ON
fi
STALE_BUILD_OPTIONS_OBJECT="${PYTORCH_ROOT}/build/caffe2/CMakeFiles/torch_cpu.dir/core/common.cc.o"
if [[ -e "${STALE_BUILD_OPTIONS_OBJECT}" ]]; then
  rm -f -- "${STALE_BUILD_OPTIONS_OBJECT}"
fi

build_env=(
  "VERBOSE=0"
  "CMAKE_FRESH=1"
  "ROCM_PATH=${ROCM_PATH}"
  "USE_CUDA=0"
  "USE_KINETO=1"
  "BUILD_TEST=ON"
  "USE_MAGMA=${MAGMA_BUILD_OPTION}"
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
  "USE_MKLDNN=${USE_MKLDNN:-1}"
  "NO_CAFFE2_OPS=1"
  "USE_MKL=ON"
  "USE_NINJA=ON"
)
env "${build_env[@]}" "${PYTHON_BIN}" -m build --wheel --no-isolation

