#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause


### PyTorch HCU/DTK 编译流程
### 说明：
### 1. 运行本脚本前须安装 DTK，默认路径为 /opt/dtk。
### 2. 默认使用脚本所在仓库的当前 checkout；代码仓库地址和分支仅在需要自行拉取时配置。
### 3. 本脚本依赖编译镜像自带 GCC 工具链，无需额外激活。
### 4. 三方库通过 git submodule 拉取，并使用 tools/third_party/apply_patches.sh 应用仓库补丁。
### 5. MAGMA 默认启用；如需跳过（编译参数 USE_MAGMA=0），请追加参数 --no-magma。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

### 使用方式
### 默认编译（启用 MAGMA）：
###   bash scripts/build_hcu.sh
### 使用资源服务器自动选择 MAGMA：
###   RESOURCE_SERVER_URL=http://10.16.1.201:8000 bash scripts/build_hcu.sh
### 关闭 MAGMA（跳过下载，USE_MAGMA=0）：
###   bash scripts/build_hcu.sh --no-magma
### manylinux 出包请在编译完成后单独执行：
###   bash scripts/build_hcu_manylinux.sh

MAGMA_ENABLED=1
for arg in "$@"; do
    case "$arg" in
        --no-magma)
            MAGMA_ENABLED=0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

### 从代码仓库拉取 PyTorch 代码分支（按需配置）
export PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-}"
export PYTORCH_BRANCH="${PYTORCH_BRANCH:-}"
export PYTORCH_ROOT="${PYTORCH_ROOT:-${REPO_ROOT}}"

### 设置已安装 DTK 路径和 MAGMA 下载地址（按需配置）
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export RESOURCE_SERVER_URL="${RESOURCE_SERVER_URL:-}"
export MAGMA_URL="${MAGMA_URL:-}"

if [ ! -f "${DTK_ROOT}/env.sh" ]; then
    echo "ERROR: DTK is not installed at ${DTK_ROOT}" >&2
    echo "Install DTK first, or set DTK_ROOT to the DTK installation directory." >&2
    exit 1
fi

### 设置 workspace。CI 默认直接使用已经 checkout/合并完成的源码。
if [ -n "${PYTORCH_REPO_URL}" ] || [ -n "${PYTORCH_BRANCH}" ]; then
    if [ -z "${PYTORCH_REPO_URL}" ] || [ -z "${PYTORCH_BRANCH}" ]; then
        echo "ERROR: PYTORCH_REPO_URL and PYTORCH_BRANCH must be set together" >&2
        exit 1
    fi
elif [ ! -d "${PYTORCH_ROOT}/.git" ]; then
    echo "ERROR: existing PyTorch checkout not found at ${PYTORCH_ROOT}" >&2
    exit 1
fi

if [ -n "${PYTORCH_REPO_URL}" ] && [ -d "${PYTORCH_ROOT}/.git" ]; then
    echo ">>>>>>>>>>USE EXISTING PYTORCH SOURCE<<<<<<<<<<"
    cd "${PYTORCH_ROOT}"
    git remote set-url origin "${PYTORCH_REPO_URL}"
    git fetch origin "${PYTORCH_BRANCH}"
    if git show-ref --verify --quiet "refs/heads/${PYTORCH_BRANCH}"; then
        git checkout "${PYTORCH_BRANCH}"
        git merge --ff-only "origin/${PYTORCH_BRANCH}"
    else
        git checkout -b "${PYTORCH_BRANCH}" "origin/${PYTORCH_BRANCH}"
    fi
elif [ -n "${PYTORCH_REPO_URL}" ]; then
    echo ">>>>>>>>>>CLONE PYTORCH SOURCE<<<<<<<<<<"
    mkdir -p "$(dirname "${PYTORCH_ROOT}")"
    git clone --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO_URL}" "${PYTORCH_ROOT}"
    cd "${PYTORCH_ROOT}"
else
    echo ">>>>>>>>>>USE CURRENT PYTORCH CHECKOUT<<<<<<<<<<"
    cd "${PYTORCH_ROOT}"
fi
WORK_SPACE=$(pwd)
export WORK_SPACE
export DISTDIR=${WORK_SPACE}/dist

### 设置 MKL 路径
if [ -f /opt/intel/oneapi/mkl/latest/env/vars.sh ]; then
    set +u
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
    set -u
fi
if [ -n "${MKLROOT:-}" ] && [ -d "${MKLROOT}/lib" ]; then
    export MKL_LIB_DIR="${MKLROOT}/lib"
elif [ -d /opt/intel/mkl/lib/intel64_lin ]; then
    export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
elif [ -d /opt/intel/oneapi/mkl/latest/lib ]; then
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
else
    echo "ERROR: cannot find MKL lib directory" >&2
    exit 1
fi

copy_mkl_runtime_lib() {
    local base_name="$1"
    local dst_dir="$2"
    local source_file real_file real_name

    if [ -e "${MKL_LIB_DIR}/${base_name}.so" ]; then
        source_file="${MKL_LIB_DIR}/${base_name}.so"
    elif [ -e "${MKL_LIB_DIR}/${base_name}.so.2" ]; then
        source_file="${MKL_LIB_DIR}/${base_name}.so.2"
    else
        echo "ERROR: cannot find ${base_name}.so or .so.2 in ${MKL_LIB_DIR}" >&2
        exit 1
    fi
    real_file="$(readlink -f "${source_file}")"
    real_name="$(basename "${real_file}")"
    mkdir -p "${dst_dir}"
    cp -a "${real_file}" "${dst_dir}/${real_name}"
    if [ "${real_name}" != "${base_name}.so" ]; then
        ln -sfn "${real_name}" "${dst_dir}/${base_name}.so"
    fi
}

### 导入 DTK 内置环境变量
echo ">>>>>>>>>>LOAD INSTALLED DTK<<<<<<<<<<"
set +u
source "${DTK_ROOT}/env.sh"
set -u

### 设置 PyTorch 编译所需的 DTK/ROCm 环境
export ROCM_PATH="${DTK_ROOT}"
export HIP_PATH="${ROCM_PATH}/hip"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx906;gfx926;gfx928;gfx936;gfx938}"
export PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/miopen/bin:${PATH}"
export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} --gcc-toolchain=$(dirname "$(dirname "$(readlink -f "$(command -v g++)")")")"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${ROCM_PATH}/magma/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export PYTORCH_CUSTOM_SUFFIX=''

# Match the version source used by cmake/public/LoadHIP.cmake.
if [ ! -f "${DTK_ROOT}/.dtk_version" ]; then
    echo "ERROR: missing ${DTK_ROOT}/.dtk_version" >&2
    exit 1
fi
DTK_VERSION_LINE="$(head -n 1 "${DTK_ROOT}/.dtk_version")"
if [[ "${DTK_VERSION_LINE}" =~ ^DTK[-_]([0-9]+)\.([0-9][0-9])([._-]|$) ]]; then
    DTK_RELEASE="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
else
    echo "ERROR: invalid DTK version: ${DTK_VERSION_LINE}" >&2
    exit 1
fi
echo "Detected DTK release: ${DTK_RELEASE}"

if [ "${DTK_RELEASE}" = "2604" ]; then
    : "${DTK2604_DEPENDENCY_1:?ERROR: DTK2604_DEPENDENCY_1 is required for DTK 26.04}"
    : "${DTK2604_DEPENDENCY_2:?ERROR: DTK2604_DEPENDENCY_2 is required for DTK 26.04}"
    HIP_LANG_CONFIG="${DTK_ROOT}/hip/lib/cmake/hip-lang/hip-lang-config.cmake"
    python3 - "${HIP_LANG_CONFIG}" <<'PY'
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

path = Path(sys.argv[1])
dependency_1 = os.environ["DTK2604_DEPENDENCY_1"]
dependency_2 = os.environ["DTK2604_DEPENDENCY_2"]
for dependency in (dependency_1, dependency_2):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dependency):
        raise ValueError(f"invalid CMake dependency name: {dependency!r}")
raw = path.read_bytes()
bom = raw.startswith(b"\xef\xbb\xbf")
text = raw.decode("utf-8-sig")
if "\ufffd" in text:
    raise ValueError(f"replacement character in {path}")
if b"\r\n" in raw and raw.count(b"\r\n") != raw.count(b"\n"):
    raise ValueError(f"mixed line endings in {path}")
newline = "\r\n" if b"\r\n" in raw else "\n"
text = text.replace("\r\n", "\n")
old = re.compile(
    rf"(?m)^(?P<indent>[ \t]*)find_dependency\({re.escape(dependency_1)}\)\n"
    rf"(?P=indent)find_dependency\({re.escape(dependency_2)}\)[ \t]*$"
)
anchor = 'include( "${CMAKE_CURRENT_LIST_DIR}/hip-lang-targets.cmake" )'
replacement = (
    f"find_dependency({dependency_1})\n"
    f"find_dependency({dependency_2} PATHS ${{ROCM_PATH}}/lib64/cmake/{dependency_2})\n"
    + anchor
)
if replacement in text:
    if old.search(text):
        raise ValueError(f"duplicate active dependencies in {path}")
    print(f"DTK 26.04 hip-lang dependencies already adjusted: {path}")
    sys.exit(0)
if len(old.findall(text)) != 1 or text.count(anchor) != 1:
    raise ValueError(f"unexpected hip-lang dependency layout in {path}")
updated = old.sub(
    lambda match: (
        f"{match['indent']}#find_dependency({dependency_1})\n"
        f"{match['indent']}#find_dependency({dependency_2})"
    ),
    text,
    count=1,
)
updated = updated.replace(anchor, replacement, 1)
encoded = (b"\xef\xbb\xbf" if bom else b"") + updated.replace("\n", newline).encode("utf-8")
fd, temp_name = tempfile.mkstemp(prefix=".hip-lang-config.", dir=path.parent)
try:
    with os.fdopen(fd, "wb") as temp:
        temp.write(encoded)
    os.chmod(temp_name, stat.S_IMODE(path.stat().st_mode))
    os.replace(temp_name, path)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
if path.read_bytes() != encoded or "\ufffd" in path.read_bytes().decode("utf-8-sig"):
    raise ValueError(f"write verification failed: {path}")
print(f"Adjusted DTK 26.04 hip-lang dependencies: {path}")
PY
elif [ "${DTK_RELEASE}" = "2610" ]; then
    # The 26.10 package keeps both temporary hip-lang edits disabled.
    echo "DTK 26.10: skip temporary hip-lang dependency edits"
fi

if [ -z "${MAGMA_URL}" ] && [ -n "${RESOURCE_SERVER_URL}" ]; then
    RESOURCE_SERVER_URL="${RESOURCE_SERVER_URL%/}"
    if [ "${DTK_RELEASE}" = "2604" ]; then
        MAGMA_URL="${RESOURCE_SERVER_URL}/thirdparty/pytorch/magma-install.tar.gz"
    elif [ "${DTK_RELEASE}" = "2610" ]; then
        MAGMA_URL="${RESOURCE_SERVER_URL}/thirdparty/pytorch/magma_v2.8.0-hip_DTK26.10/MAGMA-v2.8.0_hip.dtk26.10-16Oct2026.tar.gz"
    fi
    export MAGMA_URL
fi

### 加载 magma 库
if [ "${MAGMA_ENABLED}" = "1" ]; then
    if [ -z "${MAGMA_URL}" ]; then
        echo "ERROR: MAGMA_URL is not set" >&2
        exit 1
    fi
    echo ">>>>>>>>>>LOAD MAGMA<<<<<<<<<<"
    cd "${ROCM_PATH}"
    MAGMA_ARCHIVE=magma-package.tar.gz
    wget -O "${MAGMA_ARCHIVE}" "${MAGMA_URL}"
    rm -rf magma
    tar -zxf "${MAGMA_ARCHIVE}"
    rm -f "${MAGMA_ARCHIVE}"
    if [ ! -d magma ]; then
        MAGMA_EXTRACT_DIR=$(find . -maxdepth 1 -type d -iname 'magma*' ! -name magma | sort | tail -n 1)
        if [ -z "${MAGMA_EXTRACT_DIR}" ]; then
            echo "ERROR: cannot find extracted MAGMA directory under /opt/dtk" >&2
            exit 1
        fi
        mv "${MAGMA_EXTRACT_DIR}" magma
    fi

    ### 将 MKL 的库文件复制到 magma 库路径下
    cd magma/lib/
    if [ "${DTK_RELEASE}" = "2610" ]; then
        for magma_library in libmagma.so libmagma_sparse.so; do
            [ -e "${magma_library}" ] || continue
            if readelf -d "${magma_library}" 2>/dev/null |
                    grep -Fq 'Shared library: [libhipblas.so.2]'; then
                if ! command -v patchelf >/dev/null 2>&1; then
                    echo "ERROR: ${magma_library} needs libhipblas.so.2 but patchelf is unavailable" >&2
                    exit 1
                fi
                if [ ! -e "${ROCM_PATH}/lib/libhipblas.so.3" ]; then
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
    fi
    for name in \
        libmkl_core \
        libmkl_gf_lp64 \
        libmkl_gnu_thread \
        libmkl_def \
        libmkl_vml_def
    do
        copy_mkl_runtime_lib "${name}" .
    done
else
    echo ">>>>>>>>>>SKIP MAGMA (USE_MAGMA=0)<<<<<<<<<<"
fi

### 更新第三方库并应用仓库补丁
echo ">>>>>>>>>>third party<<<<<<<<<<"
cd "${WORK_SPACE}"
git submodule update --init --recursive

cd "${WORK_SPACE}"
if [ ! -f tools/third_party/apply_patches.sh ]; then
    echo "ERROR: tools/third_party/apply_patches.sh does not exist" >&2
    exit 1
fi
bash tools/third_party/apply_patches.sh

### 返回 PyTorch 目录下
cd "${WORK_SPACE}"
export PYTORCH_VERSION_DTK="$(cat version.txt)"

### 升级 pip
python3 -m pip install --upgrade pip \
    -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
    --default-timeout=1000

### 安装 .ci/docker/requirements-ci.txt 文件中的依赖
pip3 install -r .ci/docker/requirements-ci.txt \
    -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
    --default-timeout=1000

### 安装 requirements.txt 文件中的依赖
pip3 install -r requirements.txt \
    -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
    --default-timeout=1000

### 安装其余依赖
pip3 install xdoctest==1.1.3 \
    pytest==8.1.1 \
    pytest-timeout \
    Pillow==10.0.1 \
    mypy==1.8.0 \
    jmespath==1.0.1 \
    Jinja2==3.1.3 \
    expecttest==0.2.1 \
    tlparse==0.3.5 \
    sympy==1.13.3 \
    MonkeyType \
    -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
    --default-timeout=1000

### 根据 torch 版本判断
IFS='.' read -r MAJOR_ MINOR_ _ < version.txt
if (( MAJOR_ < 2 || (MAJOR_ == 2 && MINOR_ < 4) )); then
    pip3 install sympy==1.12.1 \
        -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
        --default-timeout=1000
fi

### 导入环境变量
echo ">>>>>>>export envs >>>>>>>"
export LD_LIBRARY_PATH="${ROCM_PATH}/rocblas/lib:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${ROCM_PATH}/rocrand/include:${CPLUS_INCLUDE_PATH:-}"
if [ "${MAGMA_ENABLED}" = "1" ]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/magma/lib:${LD_LIBRARY_PATH}"
fi
# Do not force --copy-dt-needed-entries; keep caller-provided LDFLAGS.

### 给 hipify 脚本加上权限并转码
if [ -f third_party/gloo/tools/amd_build/build_amd.py ]; then
    chmod 755 third_party/gloo/tools/amd_build/build_amd.py
fi
python3 tools/hcu_build/build_hcu.py

### 复制 MKL / MAGMA 到 torch/lib
mkdir -p "${WORK_SPACE}/torch/lib"
for name in \
    libmkl_core \
    libmkl_gf_lp64 \
    libmkl_gnu_thread \
    libmkl_def \
    libmkl_intel_lp64 \
    libmkl_vml_def
do
    copy_mkl_runtime_lib "${name}" "${WORK_SPACE}/torch/lib"
done
if [ "${MAGMA_ENABLED}" = "1" ]; then
    cp "${ROCM_PATH}/magma/lib/libmagma.so" "${WORK_SPACE}/torch/lib/"
    if command -v patchelf >/dev/null 2>&1; then
        MAGMA_RUNPATH='$ORIGIN:'"${ROCM_PATH}/lib:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:/opt/hyhal/lib"
        patchelf --set-rpath "${MAGMA_RUNPATH}" "${WORK_SPACE}/torch/lib/libmagma.so"
        echo "libmagma.so RUNPATH: $(patchelf --print-rpath "${WORK_SPACE}/torch/lib/libmagma.so")"
    else
        echo "WARNING: patchelf unavailable; libmagma.so may not locate bundled MKL libraries" >&2
    fi
fi

### 开始编译
cd "${WORK_SPACE}"
echo ">>>>>>>compile start>>>>>>>"
if [ "${MAGMA_ENABLED}" = "1" ]; then
    MAGMA_OPT=ON
else
    MAGMA_OPT=OFF
fi
ROCM_PATH=${ROCM_PATH} \
USE_AOTRITON_FA=0 \
USE_MEM_EFF_ATTENTION=0 \
USE_FLASH_ATTENTION=1 \
USE_FBGEMM=ON \
USE_MKLDNN=1 \
USE_CUDA=0 \
USE_KINETO=1 \
USE_MKL=ON \
BUILD_TEST=ON \
USE_MAGMA=${MAGMA_OPT} \
NO_CAFFE2_OPS=1 \
USE_NCCL=1 \
USE_RCCL=1 \
USE_DISTRIBUTED=1 \
USE_GLOO_IBVERBS=1 \
USE_GLOO=1 \
USE_IBVERBS=1 \
USE_C10D_MPI=1 \
USE_MPI=1 \
USE_OPENMP=1 \
USE_MIOPEN=1 \
USE_ROCM=1 \
USE_LMDB=1 \
USE_OPENCV=1 \
USE_NUMPY=1 \
USE_GLOG="${USE_GLOG:-1}" \
USE_GFLAGS=1 \
USE_ROCM_CK_GEMM=0 \
USE_ROCM_CK_SDPA=0 \
USE_CUFILE=0 \
USE_NINJA=ON \
PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS} \
python3 setup.py install --user bdist_wheel

### 编译后出包到 pytorch/dist 文件夹下
cd "${WORK_SPACE}"
echo ">>>>>>>>>>wheel<<<<<<<<<<"
USE_NCCL=0 python3 -m pip wheel . -w ./dist --no-deps --no-build-isolation
