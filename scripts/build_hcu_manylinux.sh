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
GLIBC_VERSION="$(ldd --version | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)"
if [ -z "${GLIBC_VERSION}" ]; then
    echo "ERROR: unable to determine glibc version" >&2
    exit 1
fi
PLATFORM="manylinux_${GLIBC_VERSION/./_}_x86_64"
export PLATFORM

shopt -s nullglob
WHEELS=("${PYTORCH_ROOT}"/dist/*-linux_x86_64.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one Linux wheel under ${PYTORCH_ROOT}/dist, found ${#WHEELS[@]}" >&2
    echo "Run scripts/build_hcu.sh first." >&2
    exit 1
fi
mkdir -p "${MANYLINUX_OUTPUT_DIR}"

auditwheel repair --plat "${PLATFORM}" --strip \
    --exclude libgalaxyhip.so.5 \
    --exclude libMIOpen.so.1 \
    --exclude librccl.so.1 \
    --exclude libhipblas.so.0 \
    --exclude libhipfft.so \
    --exclude libhiprand.so.1 \
    --exclude libhipsolver.so.0 \
    --exclude libhipsparse.so.0 \
    --exclude libhipnn.so \
    --exclude librocblas.so.0 \
    --exclude librocsolver.so.0 \
    --exclude librocfft.so.0 \
    --exclude librocrand.so.1 \
    --exclude librocsparse.so.0 \
    --exclude librocm_smi64.so.2 \
    --exclude librocfft-device-0.so.0 \
    --exclude librocfft-device-1.so.0 \
    --exclude librocfft-device-2.so.0 \
    --exclude librocfft-device-3.so.0 \
    --exclude libc10.so \
    --exclude libc10_hip.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_hip.so \
    --exclude libtorch_python.so \
    --exclude libtorch.so \
    --exclude libtorchaudio_ffmpeg4.so \
    --exclude libtorchaudio_ffmpeg5.so \
    --exclude libtorchaudio_ffmpeg6.so \
    --exclude libtorchaudio.so \
    --exclude libtorchaudio_sox.so \
    --exclude _torchaudio_ffmpeg4.so \
    --exclude _torchaudio_ffmpeg5.so \
    --exclude _torchaudio_ffmpeg6.so \
    --exclude _torchaudio.so \
    --exclude _torchaudio_sox.so \
    --exclude libavcodec.so.58 \
    --exclude libavcodec.so.59 \
    --exclude libavcodec.so.60 \
    --exclude libavdevice.so.58 \
    --exclude libavdevice.so.59 \
    --exclude libavdevice.so.60 \
    --exclude libavfilter.so.7 \
    --exclude libavfilter.so.8 \
    --exclude libavfilter.so.9 \
    --exclude libavformat.so.58 \
    --exclude libavformat.so.59 \
    --exclude libavformat.so.60 \
    --exclude libavutil.so.56 \
    --exclude libavutil.so.57 \
    --exclude libavutil.so.58 \
    --exclude libsox.so \
    --exclude libomp.so \
    --exclude libhipblaslt.so.0 \
    --exclude libhipblas.so.2 \
    --exclude libhipfft.so.0 \
    --exclude libhipsparse.so.1 \
    --exclude librocblas.so.4 \
    --exclude librocsparse.so.1 \
    --exclude libnuma.so.1 \
    -w "${MANYLINUX_OUTPUT_DIR}" \
    "${WHEELS[@]}"

shopt -s nullglob
MANYLINUX_WHEELS=("${MANYLINUX_OUTPUT_DIR}"/*manylinux*.whl)
shopt -u nullglob
if [ "${#MANYLINUX_WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one manylinux wheel, found ${#MANYLINUX_WHEELS[@]}" >&2
    exit 1
fi
