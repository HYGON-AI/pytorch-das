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

case "${DTK_BUILD_VERSION}" in
    2610*)
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
            'libgflags*'
            'libglog*'
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
        ;;
    *)
        selected_auditwheel_excludes=(
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
            libhipblaslt.so.0
            libhipblas.so.2
            libhipfft.so.0
            libhipsparse.so.1
            librocblas.so.4
            librocsparse.so.1
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
            libnuma.so.1
        )
        ;;
esac

auditwheel_excludes=()
for library in "${selected_auditwheel_excludes[@]}"; do
    auditwheel_excludes+=(--exclude "${library}")
done
echo "========== auditwheel exclusions selected for DTK ${DTK_BUILD_VERSION:-unknown} =========="
auditwheel repair --plat "${PLATFORM}" --strip \
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
