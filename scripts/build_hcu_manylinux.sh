#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

# HCU PyTorch 2.14 manylinux packaging.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
ROCM_PATH="${DTK_ROOT}"
test -f "${DTK_ROOT}/env.sh"
set +u
source "${DTK_ROOT}/env.sh"
set -u
cd "${PYTORCH_ROOT}"
python3 -m pip install auditwheel patchelf
DTK_BUILD_VERSION="$(tr -d '.[:space:]' < "${DTK_ROOT}/.info/rocm_version")"
MKL_LIBRARY_DIR="${MKL_LIBRARY_DIR:-/opt/intel/oneapi/mkl/latest/lib}"
if [[ ! -d "${MKL_LIBRARY_DIR}" ]]; then
  MKL_LIBRARY_DIR=/opt/intel/mkl/lib/intel64_lin
fi
# Generate the manylinux wheel.
echo "========== Generating manylinux wheel =========="
mapfile -t wheel_candidates < <(find dist -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#wheel_candidates[@]}" -ne 1 ]]; then
  echo "ERROR: expected exactly one input wheel in dist/, found ${#wheel_candidates[@]}." >&2
  exit 1
fi

glibc_version_info="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
if [[ ! "${glibc_version_info}" =~ ([0-9]+\.[0-9]+) ]]; then
  glibc_version_info="$(ldd --version 2>&1 || true)"
fi
if [[ "${glibc_version_info}" =~ ([0-9]+\.[0-9]+) ]]; then
  glibc_version="${BASH_REMATCH[1]}"
else
  echo "ERROR: unable to determine glibc version for the manylinux tag." >&2
  exit 1
fi
PLATFORM="manylinux_${glibc_version/./_}_x86_64"
export LD_LIBRARY_PATH="${PYTORCH_ROOT}/torch/lib:${MKL_LIBRARY_DIR}:${ROCM_PATH}/magma/lib:${LD_LIBRARY_PATH:-}"

# DTK 26.10 updates SONAME suffixes between packages, so its approved
# exclusion set uses wildcard suffixes.
selected_auditwheel_excludes=(
      'libgalaxyhip.so*'
      'libMIOpen.so*'
      'librccl.so*'
      'libhipblas.so*'
      'libhipblaslt.so*'
      'libhipfft.so*'
      'libhiprand.so*'
      'libhiprtc.so*'
      'libhipsolver.so*'
      'libhipsparse.so*'
      'libhipnn.so*'
      'libhipprof*'
      'librocblas.so*'
      'librocsolver.so*'
      'librocfft.so*'
      'librocrand.so*'
      'librocsparse.so*'
      'librocm_smi64.so*'
      'libroctracer64.so*'
      'libroctx64.so*'
      'librocfft-device-*.so*'
      'libhsa-runtime64.so*'
      'libamdhip64.so*'
      'libamd_comgr.so*'
      'libhsakmt.so*'
      'libc10.so*'
      'libc10_hip.so*'
      'libtorch_cpu.so*'
      'libtorch_hip.so*'
      'libtorch_python.so*'
      'libtorch.so*'
      'libtorchaudio*'
      '_torchaudio*'
      'libavcodec*'
      'libavdevice*'
      'libavfilter*'
      'libavformat*'
      'libavutil*'
      'libsox*'
      'libomp.so*'
      'libnuma.so*'
      'libibverbs*'
      'libnl*'
      'libmpi*'
      'libpmix*'
)

auditwheel_excludes=()
for library in "${selected_auditwheel_excludes[@]}"; do
  auditwheel_excludes+=(--exclude "${library}")
done
echo "========== auditwheel exclusions selected for DTK ${DTK_BUILD_VERSION:-unknown} =========="
auditwheel repair \
  --plat "${PLATFORM}" \
  "${auditwheel_excludes[@]}" \
  --wheel-dir dist/ \
  "${wheel_candidates[0]}"

mapfile -t repaired_wheels < <(find dist -maxdepth 1 -type f -name '*manylinux*.whl' -print)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
  echo "ERROR: expected exactly one repaired manylinux wheel, found ${#repaired_wheels[@]}." >&2
  exit 1
fi

echo "========== Build complete =========="
