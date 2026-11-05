#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from scripts/build_hcu.sh (same HCU/DTK flow) with MKL replaced by HML:
#   - no MKL_LIB_DIR requirement, no libmkl_* copying into cold_workspace/torch/lib
#   - USE_HML=ON (v4 master switch) instead of USE_MKL=ON: it selects BLAS=HBLAS and
#     enables HLAPACK/HSPARSE/HFFT/HVML (legacy BLAS=HBLAS still works too)
#   - USE_HDNN=ON (hygonDNN replaces bundled oneDNN; implies USE_MKLDNN=ON)
#   - build toolchain overrides (system cmake 3.22 is too old for PyTorch 2.12):
#     PATH gets ~/.local/bin first, CMAKE_PREFIX_PATH=/opt/hyhal:/tmp/dtkshim:/opt/dtk:/opt/dtk/lib64/cmake,
#     CPLUS_INCLUDE_PATH=/tmp/hyhalstub (amdsmi smi.h stub)
#
# Usage:
#   bash scripts/build_hcu_hml.sh [options]
#
#   --hml-home DIR        HML install root                        [required*]
#   --hml-root DIR        HML parent dir (alternative to --hml-home)
#   --hml-variant NAME    shortcut for --hml-root/NAME
#                         NAME = gcc | hgcc | hcc | gcc_intel
#   --hdnn-home DIR       HygonDNN install root                   [required*]
#
#   * No built-in default path on purpose: run without --hml-home/--hdnn-home
#     (and without $HML_HOME/$HDNN_HOME) and the script prints this usage
#     instead of guessing a directory.
#   --check               resolve + validate paths, then exit without building
#   --with-submodules     re-init git submodules before building
#   --help                show usage
#
# Env knobs:
#   INSTALL_WHEEL=0   build the wheel but do NOT pip install it. Default is 1, which runs
#                     `pip install --force-reinstall` and therefore REPLACES the system torch.
#   MANYLINUX=1       after the wheel is built, run scripts/build_hcu_hml_manylinux.sh to produce
#                     a manylinux_<glibc>_x86_64 wheel as well (default 0: only linux_x86_64).
#                     Requires the wheel to be installed (INSTALL_WHEEL=1) plus patchelf/auditwheel.
#   EXTRA_EXCLUDES    extra auditwheel --exclude libs; when MANYLINUX=1 this script defaults it
#                     to the HML/HDNN runtime libs (see below), override to change.
#
# Precedence: command line > environment ($HML_HOME/$HDNN_HOME/$HML_ROOT).
# There is no built-in default path: both roots must be given explicitly.
# Each component (HBLAS/HLAPACK/HSPARSE/HFFT/HVML_HOME) may also be overridden individually.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/build_hcu_hml.sh [options]

  --hml-home DIR      HML install root                        [required*]
  --hml-root DIR      HML parent directory (alternative to --hml-home)
  --hml-variant NAME  shortcut for --hml-root/NAME (requires --hml-root or $HML_ROOT)
                      NAME = gcc | hgcc | hcc | gcc_intel
  --hdnn-home DIR     HygonDNN install root                   [required*]
  --check             resolve + validate paths, then exit without building
  --with-submodules   re-init git submodules before building
  -h, --help          show this help

Environment: $HML_HOME, $HDNN_HOME and $HML_ROOT are honoured.
Precedence: command line > environment.

  * --hml-home and --hdnn-home have NO built-in default. Without them (and
    without the matching environment variables) this usage is printed and the
    script exits with an error instead of silently picking a directory.

Examples:
  bash scripts/build_hcu_hml.sh --hml-home /path/to/hml/gcc --hdnn-home /path/to/hdnn
  bash scripts/build_hcu_hml.sh --hml-root /path/to/hml --hml-variant gcc \
                                --hdnn-home /path/to/hdnn
  HML_HOME=/path/to/hml/gcc HDNN_HOME=/path/to/hdnn bash scripts/build_hcu_hml.sh
EOF
}

usage_error() {
    usage >&2
    echo "" >&2
    echo "ERROR: $1" >&2
    echo "       Pass --hml-home DIR and --hdnn-home DIR (or set HML_HOME / HDNN_HOME)." >&2
    echo "       There is intentionally no built-in default path." >&2
    exit 1
}

WITH_SUBMODULES=0
CHECK_ONLY=0
HML_HOME_ARG=""
HDNN_HOME_ARG=""
HML_ROOT_ARG=""
HML_VARIANT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-submodules) WITH_SUBMODULES=1 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        --hml-home)
            [ "$#" -ge 2 ] || { echo "ERROR: --hml-home needs a directory" >&2; exit 1; }
            HML_HOME_ARG="$2"; shift ;;
        --hml-home=*) HML_HOME_ARG="${1#*=}" ;;
        --hml-variant)
            [ "$#" -ge 2 ] || { echo "ERROR: --hml-variant needs a name" >&2; exit 1; }
            HML_VARIANT="$2"; shift ;;
        --hml-variant=*) HML_VARIANT="${1#*=}" ;;
        --hdnn-home)
            [ "$#" -ge 2 ] || { echo "ERROR: --hdnn-home needs a directory" >&2; exit 1; }
            HDNN_HOME_ARG="$2"; shift ;;
        --hdnn-home=*) HDNN_HOME_ARG="${1#*=}" ;;
        --hml-root)
            [ "$#" -ge 2 ] || { echo "ERROR: --hml-root needs a directory" >&2; exit 1; }
            HML_ROOT_ARG="$2"; shift ;;
        --hml-root=*) HML_ROOT_ARG="${1#*=}" ;;
        *) echo "ERROR: unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
    shift
done

export PYTORCH_ROOT="${PYTORCH_ROOT:-${REPO_ROOT}}"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export WORK_SPACE="${PYTORCH_ROOT}"
export DISTDIR="${WORK_SPACE}/dist"

if [ ! -f "${WORK_SPACE}/cmake/hml/HML.cmake" ]; then
    echo "ERROR: HML patch is not applied (missing ${WORK_SPACE}/cmake/hml/HML.cmake)." >&2
    echo "       Apply hygon-hml-hdnn-pytorch-v4.patch first; otherwise USE_HML=ON is silently" >&2
    echo "       ignored by CMake and you get a plain (non-HML) build." >&2
    exit 1
fi

if [ -n "${HML_HOME_ARG}" ] && [ -n "${HML_VARIANT}" ]; then
    echo "ERROR: --hml-home and --hml-variant are mutually exclusive" >&2
    exit 1
fi
if [ -n "${HML_HOME_ARG}" ] && [ -n "${HML_ROOT_ARG}" ]; then
    echo "ERROR: --hml-home and --hml-root are mutually exclusive" >&2
    exit 1
fi

if [ -n "${HML_HOME_ARG}" ]; then
    HML_HOME="${HML_HOME_ARG}"
elif [ -n "${HML_VARIANT}" ]; then
    HML_ROOT="${HML_ROOT_ARG:-${HML_ROOT:-}}"
    if [ -z "${HML_ROOT}" ]; then
        usage_error "--hml-variant needs a parent directory: pass --hml-root DIR (or set HML_ROOT)."
    fi
    HML_HOME="${HML_ROOT%/}/${HML_VARIANT}"
elif [ -n "${HML_HOME:-}" ]; then
    : # from the environment
else
    usage_error "--hml-home is required (no built-in default path)."
fi
export HML_HOME
export HBLAS_HOME="${HBLAS_HOME:-${HML_HOME}}"
export HLAPACK_HOME="${HLAPACK_HOME:-${HML_HOME}}"
export HSPARSE_HOME="${HSPARSE_HOME:-${HML_HOME}}"
export HFFT_HOME="${HFFT_HOME:-${HML_HOME}}"
export HVML_HOME="${HVML_HOME:-${HML_HOME}}"

if [ -n "${HDNN_HOME_ARG}" ]; then
    HDNN_HOME="${HDNN_HOME_ARG}"
elif [ -n "${HDNN_HOME:-}" ]; then
    : # from the environment
else
    usage_error "--hdnn-home is required (no built-in default path)."
fi
export HDNN_HOME
export USE_HDNN=ON

if [ ! -f "${DTK_ROOT}/env.sh" ]; then
    echo "ERROR: DTK is not installed at ${DTK_ROOT}" >&2
    echo "Install DTK first, or set DTK_ROOT to the DTK installation directory." >&2
    exit 1
fi

if [ -d "${HML_HOME}/lib" ]; then
    HML_LIB_DIR="${HML_HOME}/lib"
elif [ -d "${HML_HOME}/lib64" ]; then
    HML_LIB_DIR="${HML_HOME}/lib64"
else
    echo "ERROR: no library dir under ${HML_HOME} (expected lib/ or lib64/)" >&2
    exit 1
fi
if [ -d "${HDNN_HOME}/lib64" ]; then
    HDNN_LIB_DIR="${HDNN_HOME}/lib64"
elif [ -d "${HDNN_HOME}/lib" ]; then
    HDNN_LIB_DIR="${HDNN_HOME}/lib"
else
    echo "ERROR: no library dir under ${HDNN_HOME} (expected lib64/ or lib/)" >&2
    exit 1
fi

for d in "${HML_HOME}/include" "${HML_LIB_DIR}" "${HDNN_HOME}/include" "${HDNN_LIB_DIR}"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: path missing: $d" >&2
        echo "       Use --hml-home DIR / --hml-root DIR --hml-variant NAME / --hdnn-home DIR (see --help)." >&2
        exit 1
    fi
done

echo ">>>>>>>>>>HML : ${HML_HOME}<<<<<<<<<<"
echo ">>>>>>>>>>HDNN: ${HDNN_HOME}<<<<<<<<<<"
echo ">>>>>>>>>>LOAD INSTALLED DTK<<<<<<<<<<"
set +u
source "${DTK_ROOT}/env.sh"
set -u

export ROCM_PATH="${DTK_ROOT}"
export HIP_PATH="${ROCM_PATH}/hip"
export PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/miopen/bin:${PATH}"
export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} --gcc-toolchain=$(dirname "$(dirname "$(readlink -f "$(command -v g++)")")")"
# libhblas.so.1 needs libgfortran.so.5/libquadmath.so.0 from the Fortran runtime.
# It must be on the link-time search path as well, otherwise linking any binary
# against libtorch_cpu.so fails with "undefined reference to `_gfortran_*@GFORTRAN_8'".
export HML_GFORTRAN_DIR="${HML_GFORTRAN_DIR:-/opt/hygon/hgcc/lib64}"
# libhblas.so.1 has a DT_NEEDED on libgfortran.so.5, so the Fortran runtime dir must
# be on LD_LIBRARY_PATH both while linking and at run time (`import torch` on the
# target machine). LD_RUN_PATH cannot express this: PyTorch adds its own -rpath to
# every link line and that wins over the environment variable.
export LD_LIBRARY_PATH="${HML_LIB_DIR}:${HDNN_LIB_DIR}:${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${HML_GFORTRAN_DIR}:${LD_LIBRARY_PATH:-}"

# DTK ships an NCCL compatibility shim under cuda/cuda-12/...: `nccl.h` plus
# `libnccl.so -> librccl.so.1.0`. PyTorch's ROCm first pass sets USE_CUDA=ON, so
# cmake picks cmake/External/nccl.cmake (not rccl.cmake) and requires
# find_package(NCCL) to succeed. Without these two variables a clean build fails with
#   Could NOT find NCCL (missing: NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
NCCL_COMPAT_DIR="${ROCM_PATH}/cuda/cuda-12/targets/x86_64-linux"
if [ -z "${NCCL_INCLUDE_DIR:-}" ] && [ -f "${NCCL_COMPAT_DIR}/include/nccl.h" ]; then
    export NCCL_INCLUDE_DIR="${NCCL_COMPAT_DIR}/include"
fi
if [ -z "${NCCL_LIB_DIR:-}" ] && [ -e "${NCCL_COMPAT_DIR}/lib/libnccl.so" ]; then
    export NCCL_LIB_DIR="${NCCL_COMPAT_DIR}/lib"
fi
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export PYTORCH_CUSTOM_SUFFIX=''
export LDFLAGS="-Wl,--copy-dt-needed-entries"

# ---- Build toolchain / DTK shims ----
export PATH="${HOME}/.local/bin:${PATH}"
# Append (never replace): DTK's env.sh already sets CMAKE_PREFIX_PATH=/opt/hyhal:..., so a
# "${VAR:-default}" form would silently drop the entries below. Also probe the two amd_comgr
# layouts shipped by DTK (lib64/cmake) and older packaging (dcc/comgr/lib64/cmake); missing
# amd_comgr aborts the HIP language enablement in cmake/public/LoadHIP.cmake.
COMRG_CMAKE_DIR=""
for _d in "${ROCM_PATH}/lib64/cmake" "${ROCM_PATH}/dcc/comgr/lib64/cmake"; do
    if [ -d "${_d}/amd_comgr" ]; then
        COMRG_CMAKE_DIR="${_d}"
        break
    fi
done
export CMAKE_PREFIX_PATH="/opt/hyhal:/tmp/dtkshim:${ROCM_PATH}${COMRG_CMAKE_DIR:+:${COMRG_CMAKE_DIR}}:${CMAKE_PREFIX_PATH:-}"
export CPLUS_INCLUDE_PATH="/tmp/hyhalstub:${CPLUS_INCLUDE_PATH:-}"

CMAKE_BIN="$(command -v cmake || true)"
if [ -z "${CMAKE_BIN}" ]; then
    echo "ERROR: cmake not found in PATH" >&2
    exit 1
fi
CMAKE_VER="$("${CMAKE_BIN}" --version 2>/dev/null | head -n 1 | awk '{print $NF}')"
CMAKE_MAJOR="${CMAKE_VER%%.*}"
CMAKE_MINOR="${CMAKE_VER#*.}"
CMAKE_MINOR="${CMAKE_MINOR%%.*}"
if [ -z "${CMAKE_MAJOR}" ] || [ -z "${CMAKE_MINOR}" ] \
   || [ "${CMAKE_MAJOR}" -lt 3 ] \
   || { [ "${CMAKE_MAJOR}" -eq 3 ] && [ "${CMAKE_MINOR}" -lt 27 ]; }; then
    echo "ERROR: cmake >= 3.27 required, found '${CMAKE_VER}' at ${CMAKE_BIN}" >&2
    echo "       Hint: ${HOME}/.local/bin/cmake is 3.31.6" >&2
    exit 1
fi
echo ">>>>>>>>>>CMAKE: ${CMAKE_BIN} (${CMAKE_VER})<<<<<<<<<<"

if [ ! -d /tmp/dtkshim ]; then
    echo "WARNING: /tmp/dtkshim missing - it carries the CMake config shims (amd_comgr/, miopen/)." >&2
    echo "         DTK's own amd_comgr config lives under ${ROCM_PATH}/lib64/cmake, which is now on CMAKE_PREFIX_PATH," >&2
    echo "         so the build may still work; if not, recreate the shim or pass -Damd_comgr_DIR=..." >&2
fi
if [ ! -d /tmp/hyhalstub ]; then
    echo "WARNING: /tmp/hyhalstub missing - it carries the amdsmi smi.h stub used via CPLUS_INCLUDE_PATH." >&2
fi

if [ "${CHECK_ONLY}" = "1" ]; then
    echo ">>>>>>>>>>CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}<<<<<<<<<<"
    echo ">>>>>>>>>>amd_comgr cmake dir: ${COMRG_CMAKE_DIR:-NOT FOUND}<<<<<<<<<<"
    echo ">>>>>>>>>>CHECK OK (paths resolved, nothing built)<<<<<<<<<<"
    exit 0
fi

if [ "${WITH_SUBMODULES}" = "1" ]; then
    echo ">>>>>>>>>>third party submodules<<<<<<<<<<"
    cd "${WORK_SPACE}"
    git submodule deinit --force --all || true
    rm -rf -- "$(git rev-parse --git-dir)/modules"
    submodules_ok=0
    for ((attempt = 1; attempt <= 3; attempt++)); do
        echo "Updating git submodules (attempt ${attempt}/3)"
        git submodule sync --recursive
        if git -c http.version=HTTP/1.1 submodule update --init --recursive; then
            submodules_ok=1
            break
        fi
        sleep "$((attempt * 10))"
    done
    if [ "${submodules_ok}" != "1" ]; then
        echo "ERROR: git submodules still incomplete after 3 attempts" >&2
        echo "Check network/mirror access, then rerun with --with-submodules" >&2
        exit 1
    fi
fi

cd "${WORK_SPACE}"
if [ ! -f tools/third_party/apply_patches.sh ]; then
    echo "ERROR: tools/third_party/apply_patches.sh does not exist" >&2
    exit 1
fi
echo ">>>>>>>>>>third party patches<<<<<<<<<<"
bash tools/third_party/apply_patches.sh

cd "${WORK_SPACE}"
PYTHON_BIN="$(readlink -f "$(command -v python3)")"
echo "Build Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
case "${PYTHON_BIN}" in
    /usr/bin/python3*) ;;
    *) echo "WARNING: build python is ${PYTHON_BIN}; expected /usr/bin/python3 (cp310 ABI)" >&2 ;;
esac

# The CI requirements step below is inherited from build_hcu.sh and needs network access to
# a PyPI mirror. Its pins can also conflict with an already provisioned environment
# (e.g. .ci/docker/requirements-ci.txt pins typing-extensions==4.12.2 while requirements.txt
# asks for 4.16.0), which makes an offline/air-gapped build hang for hours. Set
# SKIP_PIP_BUILD=1 to skip it when the interpreter already has the build requirements.
pip_install_build() {
    if [ "${SKIP_PIP_BUILD:-0}" = "1" ]; then
        echo "SKIP_PIP_BUILD=1, skipping: pip install $*"
        return 0
    fi
    "${PYTHON_BIN}" -m pip install \
        --index-url https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
        --default-timeout=1000 \
        "$@"
}

CI_REQUIREMENTS_NO_PIP="$(mktemp /tmp/pytorch-requirements-ci.XXXXXX.txt)"
awk '!/^[[:space:]]*pip([[:space:]]|[<=>!~])/ { print }' .ci/docker/requirements-ci.txt > "${CI_REQUIREMENTS_NO_PIP}"
pip_install_build -r "${CI_REQUIREMENTS_NO_PIP}"
rm -f -- "${CI_REQUIREMENTS_NO_PIP}"
pip_install_build -r requirements.txt
pip_install_build ninja build

IFS='.' read -r MAJOR_ MINOR_ _ < version.txt
if (( MAJOR_ < 2 || (MAJOR_ == 2 && MINOR_ < 4) )); then
    pip_install_build sympy==1.12.1
fi

echo ">>>>>>>>>>hipify<<<<<<<<<<"
if [ -f third_party/gloo/tools/amd_build/build_amd.py ]; then
    chmod 755 third_party/gloo/tools/amd_build/build_amd.py
fi
python3 tools/hcu_build/build_hcu.py

cd "${WORK_SPACE}"
echo ">>>>>>>>>>compile start (MKL replaced by HML/HDNN)>>>>>>>>>>"
rm -rf -- build dist ./*.egg-info torch.egg-info

ROCM_PATH=${ROCM_PATH} \
USE_AOTRITON_FA=0 \
USE_MEM_EFF_ATTENTION=0 \
USE_FLASH_ATTENTION=1 \
USE_FBGEMM=ON \
USE_MKLDNN="${USE_MKLDNN:-1}" \
USE_CUDA=0 \
USE_KINETO=1 \
USE_HML=ON \
BUILD_TEST=ON \
NO_CAFFE2_OPS=1 \
USE_NCCL=1 \
USE_RCCL=1 \
USE_DISTRIBUTED=1 \
USE_GLOO_IBVERBS=0 \
USE_GLOO=1 \
USE_IBVERBS=0 \
USE_C10D_MPI=1 \
USE_MPI=1 \
USE_OPENMP=1 \
USE_MIOPEN=1 \
USE_ROCM=1 \
USE_LMDB=1 \
USE_OPENCV=1 \
USE_NUMPY=1 \
USE_GLOG=1 \
USE_GFLAGS=1 \
USE_ROCM_CK_GEMM=0 \
USE_ROCM_CK_SDPA=0 \
USE_CUFILE=0 \
USE_NINJA=ON \
python3 -m build --wheel --no-isolation

shopt -s nullglob
WHEELS=("${DISTDIR}"/*-linux_x86_64.whl)
shopt -u nullglob
if [ "${#WHEELS[@]}" -ne 1 ]; then
    echo "ERROR: expected one Linux wheel under ${DISTDIR}, found ${#WHEELS[@]}" >&2
    exit 1
fi
if [ "${INSTALL_WHEEL:-1}" = "1" ]; then
    echo ">>>>>>>>>>installing wheel (replaces system torch; INSTALL_WHEEL=0 to skip)<<<<<<<<<<"
    python3 -m pip install --no-deps --force-reinstall "${WHEELS[0]}"
else
    echo ">>>>>>>>>>SKIP pip install (INSTALL_WHEEL=0)<<<<<<<<<<"
fi
echo ">>>>>>>>>>WHEEL OK: ${WHEELS[0]}<<<<<<<<<<"

if [ "${MANYLINUX:-0}" = "1" ]; then
    if [ "${INSTALL_WHEEL:-1}" != "1" ]; then
        echo "ERROR: MANYLINUX=1 needs the wheel installed for auditwheel; run with INSTALL_WHEEL=1" >&2
        exit 1
    fi
    # HML/HDNN 运行时库由目标机的 DTK/HML 安装提供，不随 wheel 打包。
    export EXTRA_EXCLUDES="${EXTRA_EXCLUDES:-libhblas.so.1 libhlapack_lp64.so libhsparse.so.1 libhvml.so.1 libfftw3.so libdnnl.so.3 libgfortran.so.5 libquadmath.so.0}"
    echo ">>>>>>>>>>manylinux repair (EXTRA_EXCLUDES=${EXTRA_EXCLUDES})<<<<<<<<<<"
    PYTORCH_ROOT="${PYTORCH_ROOT}" MANYLINUX_OUTPUT_DIR="${DISTDIR}" \
        bash "${SCRIPT_DIR}/build_hcu_hml_manylinux.sh"
fi
