#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause


### 将 scripts/build_hcu.sh 生成的 wheel 修复为 manylinux wheel。
### 运行前须安装 DTK，并确保当前 Python 环境已安装对应的 torch。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTORCH_ROOT="${PYTORCH_ROOT:-${REPO_ROOT}}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export MANYLINUX_OUTPUT_DIR="${MANYLINUX_OUTPUT_DIR:-${PYTORCH_ROOT}/dist}"

if [ ! -f "${DTK_ROOT}/env.sh" ]; then
    echo "ERROR: DTK is not installed at ${DTK_ROOT}" >&2
    echo "Install DTK first, or set DTK_ROOT to the DTK installation directory." >&2
    exit 1
fi
if [ ! -d "${PYTORCH_ROOT}" ]; then
    echo "ERROR: PYTORCH_ROOT does not exist: ${PYTORCH_ROOT}" >&2
    exit 1
fi

set +u
source "${DTK_ROOT}/env.sh"
set -u
cd "${PYTORCH_ROOT}"

if [ -d /opt/intel/mkl/lib/intel64_lin ]; then
    export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
elif [ -d /opt/intel/oneapi/mkl/latest/lib ]; then
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
else
    echo "ERROR: cannot find MKL lib directory" >&2
    exit 1
fi

TORCH_LOCATION=$(python3 -m pip show torch | awk -F': ' '/^Location:/ {print $2}')
if [ -z "${TORCH_LOCATION}" ]; then
    echo "ERROR: torch is not installed in the current Python environment" >&2
    exit 1
fi
export TORCH_LOCATION
export LD_LIBRARY_PATH="${TORCH_LOCATION}/torch/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

python3 -m pip install \
    --default-timeout=1000 \
    patchelf auditwheel
GLIBC_VERSION="$(ldd --version 2>&1 | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | tail -n 1 || true)"
if [ -z "${GLIBC_VERSION}" ]; then
    echo "ERROR: unable to determine glibc version" >&2
    exit 1
fi
PLATFORM="manylinux_${GLIBC_VERSION/./_}_x86_64"
export PLATFORM

DTK_BUILD_VERSION="$(tr -d '.[:space:]' < "${DTK_ROOT}/.info/rocm_version" 2>/dev/null || true)"

shopt -s nullglob
WHEELS=("${PYTORCH_ROOT}"/dist/*-linux_x86_64.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one Linux wheel under ${PYTORCH_ROOT}/dist, found ${#WHEELS[@]}" >&2
    echo "Run scripts/build_hcu.sh first." >&2
    exit 1
fi
mkdir -p "${MANYLINUX_OUTPUT_DIR}"

selected_auditwheel_excludes=(
    'libgalaxyhip.so*'
    'libMIOpen.so*'
    'librccl.so*'
    'libhipblas.so*'
    'libhipfft.so*'
    'libhiprand.so*'
    'libhipsolver.so*'
    'libhipsparse.so*'
    'libhipnn.so*'
    'librocblas.so*'
    'librocsolver.so*'
    'librocfft.so*'
    'librocrand.so*'
    'librocsparse.so*'
    'librocm_smi64.so*'
    'librocfft-device-*.so*'
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
    'libhipblaslt.so*'
    'libnuma.so*'
    'libibverbs*'
    'libnl*'
    'libmpi*'
    'libpmix*'
    'libhipprof*'
    'libomp.so*'
)

auditwheel_excludes=()
for library in "${selected_auditwheel_excludes[@]}"; do
    auditwheel_excludes+=(--exclude "${library}")
done
echo "========== auditwheel exclusions selected for DTK ${DTK_BUILD_VERSION:-unknown} =========="
auditwheel repair --plat "${PLATFORM}" \
    "${auditwheel_excludes[@]}" \
    -w "${MANYLINUX_OUTPUT_DIR}" \
    "${WHEELS[@]}"

shopt -s nullglob
MANYLINUX_WHEELS=("${MANYLINUX_OUTPUT_DIR}"/*manylinux*.whl)
shopt -u nullglob
if [ "${#MANYLINUX_WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one manylinux wheel, found ${#MANYLINUX_WHEELS[@]}" >&2
    exit 1
fi
