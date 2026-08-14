#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause


### PyTorch HCU/DTK 编译流程
### 说明：
### 1. 代码仓库地址、分支以及 DTK/MAGMA 下载地址在下方变量区按需配置。
### 2. 本脚本依赖编译镜像自带 GCC 工具链，无需额外激活。
### 3. 三方库通过 git submodule 拉取，并使用 tools/third_party/apply_patches.sh 应用仓库补丁。
### 4. MAGMA 默认启用；如需跳过（编译参数 USE_MAGMA=0），请追加参数 --no-magma。

set -e

### 使用方式
### 默认编译（启用 MAGMA）：
###   bash scripts/build_hcu.sh
### 关闭 MAGMA（跳过下载，USE_MAGMA=0）：
###   bash scripts/build_hcu.sh --no-magma

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
export PYTORCH_ROOT="${PYTORCH_ROOT:-/workspace/pytorch}"

### 设置 DTK / MAGMA 下载地址（按需配置）
export DTK_URL="${DTK_URL:-}"
export MAGMA_URL="${MAGMA_URL:-}"

if [ -z "${PYTORCH_REPO_URL}" ] || [ -z "${PYTORCH_BRANCH}" ]; then
    echo "ERROR: PYTORCH_REPO_URL and PYTORCH_BRANCH must be set" >&2
    exit 1
fi
if [ -z "${DTK_URL}" ]; then
    echo "ERROR: DTK_URL is not set" >&2
    exit 1
fi

### 设置 workspace
if [ -d "${PYTORCH_ROOT}/.git" ]; then
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
else
    echo ">>>>>>>>>>CLONE PYTORCH SOURCE<<<<<<<<<<"
    mkdir -p "$(dirname "${PYTORCH_ROOT}")"
    git clone --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO_URL}" "${PYTORCH_ROOT}"
    cd "${PYTORCH_ROOT}"
fi
export WORK_SPACE=$(pwd)
export DISTDIR=${WORK_SPACE}/dist

### 设置 MKL 路径
### 老镜像通常是 /opt/intel/mkl/lib/intel64_lin；
### 当前部分容器是 oneAPI MKL，例如 /opt/intel/oneapi/mkl/latest/lib。
if [ -d /opt/intel/mkl/lib/intel64_lin ]; then
    export MKL_LIB_DIR=/opt/intel/mkl/lib/intel64_lin
elif [ -d /opt/intel/oneapi/mkl/latest/lib ]; then
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib
else
    echo "ERROR: cannot find MKL lib directory" >&2
    exit 1
fi

copy_mkl_lib() {
    local name="$1"
    local dst_dir="$2"

    mkdir -p "${dst_dir}"
    if [ -f "${MKL_LIB_DIR}/${name}" ]; then
        cp "${MKL_LIB_DIR}/${name}" "${dst_dir}/${name}"
    elif [ -f "${MKL_LIB_DIR}/${name}.2" ]; then
        cp "${MKL_LIB_DIR}/${name}.2" "${dst_dir}/${name}"
    else
        echo "ERROR: cannot find ${name} or ${name}.2 in ${MKL_LIB_DIR}" >&2
        exit 1
    fi
}

### 同时保留 .so.2 实体文件和 .so 软链接，兼容 PyTorch 与 MAGMA 的依赖名
copy_mkl_lib_pair() {
    local base_name="$1"
    local dst_dir="$2"

    copy_mkl_lib "${base_name}.so.2" "${dst_dir}"
    ln -sfn "${base_name}.so.2" "${dst_dir}/${base_name}.so"
}

### 开始下载 DTK
echo ">>>>>>>>>>LOAD DTK<<<<<<<<<<"
cd /opt/
wget -O DTK.tar.gz "${DTK_URL}"
tar -xf DTK.tar.gz -C /opt/
DTK_EXTRACT_DIR=$(find /opt -maxdepth 1 -type d -name 'dtk*' ! -name dtk | sort | tail -n 1)
if [ -z "${DTK_EXTRACT_DIR}" ]; then
    echo "ERROR: cannot find extracted DTK directory under /opt" >&2
    exit 1
fi
ln -sfn "${DTK_EXTRACT_DIR}" /opt/dtk
rm -rf DTK.tar.gz

### 导入 DTK 内置环境变量
source /opt/dtk/env.sh

### 设置 PyTorch 编译所需的 DTK/ROCm 环境
export ROCM_PATH="/opt/dtk"
export HIP_PATH="${ROCM_PATH}/hip"
export TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"
export AMDGPU_TARGETS="${TARGETS}"
export PYTHON_VERSION=3.10.12
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/miopen/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:${ROCM_PATH}/opencl/lib/x86_64:${ROCM_PATH}/miopen/lib:${ROCM_PATH}/rocblas/lib:${ROCM_PATH}/magma/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export MIOPEN_SYSTEM_DB_PATH="${ROCM_PATH}/miopen/share/miopen/db/"
export PYTORCH_CUSTOM_SUFFIX=''

### 加载 magma 库
if [ "${MAGMA_ENABLED}" = "1" ]; then
    if [ -z "${MAGMA_URL}" ]; then
        echo "ERROR: MAGMA_URL is not set" >&2
        exit 1
    fi
    echo ">>>>>>>>>>LOAD MAGMA<<<<<<<<<<"
    cd /opt/dtk
    wget "${MAGMA_URL}"
    rm -rf magma
    tar -zxf magma*.tar.gz
    rm -f magma*.tar.gz
    if [ ! -d magma ]; then
        MAGMA_EXTRACT_DIR=$(find . -maxdepth 1 -type d -name 'magma*' ! -name magma | sort | tail -n 1)
        if [ -z "${MAGMA_EXTRACT_DIR}" ]; then
            echo "ERROR: cannot find extracted MAGMA directory under /opt/dtk" >&2
            exit 1
        fi
        mv "${MAGMA_EXTRACT_DIR}" magma
    fi

    ### 将 MKL 的库文件复制到 magma 库路径下
    cd magma/lib/
    for name in \
        libmkl_core \
        libmkl_gf_lp64 \
        libmkl_gnu_thread \
        libmkl_def \
        libmkl_vml_def
    do
        copy_mkl_lib_pair "${name}" .
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
export PYTORCH_VERSION_DTK=$(cat version.txt)

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
    numpy==1.24.3 \
    mypy==1.8.0 \
    jmespath==1.0.1 \
    Jinja2==3.1.3 \
    expecttest==0.2.1 \
    tlparse==0.3.5 \
    numba==0.60.0 \
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
export LD_LIBRARY_PATH=${ROCM_PATH}/rocblas/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${ROCM_PATH}/rocrand/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${ROCM_PATH}/rocrand/include:$CPLUS_INCLUDE_PATH
if [ "${MAGMA_ENABLED}" = "1" ]; then
    export LD_LIBRARY_PATH=${ROCM_PATH}/magma/lib:$LD_LIBRARY_PATH
fi
export LDFLAGS="-Wl,--copy-dt-needed-entries"

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
    copy_mkl_lib_pair "${name}" "${WORK_SPACE}/torch/lib"
done
if [ "${MAGMA_ENABLED}" = "1" ]; then
    cp /opt/dtk/magma/lib/libmagma.so "${WORK_SPACE}/torch/lib/"
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
USE_MKLDNN=0 \
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
USE_GLOG=1 \
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

### manylinux 出包
echo ">>>>>>>>>>manylinux<<<<<<<<<<"
cd "${WORK_SPACE}"
export PYTHON_VERSION=3.10.12
export TORCH_LOCATION=$(pip show torch | awk -F': ' '/^Location:/ {print $2}')
export LD_LIBRARY_PATH="${TORCH_LOCATION}/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${MKL_LIB_DIR}:$LD_LIBRARY_PATH"

pip install patchelf auditwheel
export PLATFORM=$(ldd --version | awk '{print $NF}' | head -n1 | awk -F "." '{print "manylinux_"$1"_"$2"_x86_64"}')

auditwheel repair --plat ${PLATFORM} --strip \
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
    -w ./ \
    dist/torch*.whl
