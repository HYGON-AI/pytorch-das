#!/bin/bash

set -e

cd /pytorch

export ROCM_PATH=/opt/dtk
# 加载DTK完整环境
if [[ -f /opt/dtk/env.sh ]]; then
    set +u
    source /opt/dtk/env.sh
    set -u
fi

# 防止env.sh恢复成其他路径
export ROCM_PATH=/opt/dtk
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip

export PATH="/opt/dtk/bin:/opt/dtk/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/dtk/lib:/opt/dtk/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="/opt/dtk:${CMAKE_PREFIX_PATH:-}"

# 查找DTK中的HIP clang编译器
HIP_COMPILER=""

for compiler in \
    /opt/dtk/llvm/bin/clang++ \
    /opt/dtk/bin/clang++
do
    if [[ -x "${compiler}" ]]; then
        HIP_COMPILER="${compiler}"
        break
    fi
done

if [[ -z "${HIP_COMPILER}" ]]; then
    echo "ERROR: 没有找到DTK clang++"
    find /opt/dtk -maxdepth 4 -name 'clang++' -o -name 'hipcc'
    exit 1
fi

if [[ ! -f /opt/dtk/lib/cmake/hip-lang/hip-lang-config.cmake ]]; then
    echo "ERROR: 缺少 hip-lang-config.cmake"
    find /opt/dtk -path '*hip-lang*' -print
    exit 1
fi

echo "HIP compiler: ${HIP_COMPILER}"
echo "ROCM_PATH: ${ROCM_PATH}"

"${HIP_COMPILER}" --version

export HIPCXX="${HIP_COMPILER}"

export CMAKE_ARGS="${CMAKE_ARGS:-} \
-DCMAKE_HIP_COMPILER=${HIP_COMPILER} \
-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/dtk \
-DCMAKE_PREFIX_PATH=/opt/dtk"

# 清除上一次失败留下的CMake缓存
rm -rf /pytorch/build

USE_ROCM_CK_GEMM=0 \
USE_ROCM_CK_SDPA=0 \
USE_CUFILE=0 \
CXXFLAGS="${CXXFLAGS:-} -ffunction-sections -fdata-sections" \
LDFLAGS="${LDFLAGS:-} -Wl,--gc-sections" \
USE_AOTRITON_FA=0 \
USE_MEM_EFF_ATTENTION=0 \
USE_FLASH_ATTENTION=1 \
USE_FBGEMM=ON \
USE_MKLDNN=0 \
USE_MKL=ON \
USE_NINJA=ON \
USE_CUDA=0 \
USE_KINETO=1 \
BUILD_TEST=OFF \
USE_MAGMA=ON \
NO_CAFFE2_OPS=1 \
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
USE_NUMPY=1 \
USE_GLOG=1 \
USE_GFLAGS=1 \
PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS} \
python3 setup.py bdist_wheel

mkdir -p /dist
cp -v /pytorch/dist/*.whl /dist/
echo "PyTorch wheel编译完成："
ls -lh /dist/
