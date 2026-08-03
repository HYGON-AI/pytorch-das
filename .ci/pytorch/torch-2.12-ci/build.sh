#!/bin/bash
set -Eeuo pipefail

cd /pytorch

export ROCM_PATH=/opt/dtk
# 加载DTK完整环境
if [[ -f /opt/dtk/env.sh ]]; then
    set +u
    source /opt/dtk/env.sh
    set -u
fi
# GCC13重新生效
if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
    set +u
    source /opt/rh/gcc-toolset-13/enable
    set -u
fi
# 设置 wheel 版本
# 命名规则
PKG_TYPE="${PKG_TYPE:-nightly}"
BASE_VERSION="$(tr -d '[:space:]' < /pytorch/version.txt)"
DTK_VERSION_NUMBER="$(
    tr -d '.[:space:]' < /opt/dtk/.info/rocm_version
)"
COMMIT_ID="${GIT_COMMIT_SHORT:-}"
if [[ -z "${COMMIT_ID}" ]]; then
    if git -C /pytorch rev-parse HEAD >/dev/null 2>&1; then
        COMMIT_ID="$(git -C /pytorch rev-parse --short=6 HEAD)"
    else
        COMMIT_ID="unknown"
    fi
fi
if [[ "${PKG_TYPE}" == "nightly" ]]; then
    BUILD_TIME="$(date +%y%m%d%H%M)"

    PYTORCH_BUILD_VERSION="${BASE_VERSION}+das.opt1.dtk${DTK_VERSION_NUMBER}.${BUILD_TIME}.g${COMMIT_ID}"
else
    PYTORCH_BUILD_VERSION="${BASE_VERSION}+das.opt1.dtk${DTK_VERSION_NUMBER}"
fi
export PYTORCH_BUILD_VERSION
export PYTORCH_BUILD_NUMBER=1
echo "PKG_TYPE=${PKG_TYPE}"
echo "BASE_VERSION=${BASE_VERSION}"
echo "DTK_VERSION_NUMBER=${DTK_VERSION_NUMBER}"
echo "COMMIT_ID=${COMMIT_ID}"
echo "PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION}"
# 固定编译器
export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
export CMAKE_C_COMPILER=${CC}
export CMAKE_CXX_COMPILER=${CXX}
# DTK路径
export ROCM_PATH=/opt/dtk
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export PATH="/opt/rh/gcc-toolset-13/root/usr/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:${PATH}"
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
-DCMAKE_C_COMPILER=${CC} \
-DCMAKE_CXX_COMPILER=${CXX} \
-DCMAKE_HIP_COMPILER=${HIP_COMPILER} \
-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/dtk \
-DCMAKE_PREFIX_PATH=/opt/dtk"
# 清除上一次失败留下的CMake缓存
echo "===== 清理旧构建产物 ====="
rm -rf /pytorch/build
rm -rf /pytorch/dist
rm -rf /pytorch/*.egg-info
rm -rf /pytorch/torch.egg-info
mkdir -p /pytorch/dist
echo "===== Compiler Check ====="
echo "CC=${CC}"
echo "CXX=${CXX}"
${CC} --version | head -1
${CXX} --version | head -1
which gcc
which g++

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
PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION}" \
PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER}" \
python3 setup.py bdist_wheel
# 生成 manylinux wheel
echo "===== PyTorch 原始 wheel 编译完成 ====="
shopt -s nullglob
raw_wheels=(/pytorch/dist/*.whl)
shopt -u nullglob
if (( ${#raw_wheels[@]} == 0 )); then
    echo "ERROR: /pytorch/dist 中没有找到 wheel"
    exit 1
fi
if (( ${#raw_wheels[@]} > 1 )); then
    echo "ERROR: 找到多个原始 wheel："
    printf '%s\n' "${raw_wheels[@]}"
    exit 1
fi
rm -rf /dist
mkdir -p /dist/raw
# 保存原始 linux wheel
cp -v "${raw_wheels[@]}" /dist/raw/

echo "===== 修正 wheel 版本 ====="
python3 /usr/local/bin/repair_wheel.py
echo "===== 修正后的 linux wheel ====="
ls -lh /dist/raw/*.whl
echo "===== 安装修正后的 wheel ====="
python3 -m pip install \
    --no-deps \
    --force-reinstall \
    /dist/raw/*.whl
# 确保 auditwheel 和 patchelf 版本满足要求
python3 -m pip install -U auditwheel patchelf
# 优先使用 pip 安装的版本
export PATH="/usr/local/bin:${PATH}"
echo "auditwheel版本:"
auditwheel --version
echo "patchelf版本:"
patchelf --version
# Rocky Linux 8 通常得到 2.28，即 manylinux_2_28_x86_64
GLIBC_VERSION="$(
    ldd --version |
        awk 'NR == 1 {print $NF}'
)"

PLATFORM="manylinux_${GLIBC_VERSION//./_}_x86_64"

echo "GLIBC_VERSION=${GLIBC_VERSION}"
echo "PLATFORM=${PLATFORM}"
# 必须从安装后的 torch 中获取 lib 目录，不能误用源码目录
TORCH_PATH="$(
    cd / &&
        python3 -c '
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("没有找到已安装的 torch")
print(spec.submodule_search_locations[0])
'
)"

echo "TORCH_PATH=${TORCH_PATH}"
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/2024.2/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:${LD_LIBRARY_PATH}"
EXCLUDE_LIBS=(
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

AUDITWHEEL_ARGS=()

for library in "${EXCLUDE_LIBS[@]}"; do
    AUDITWHEEL_ARGS+=(--exclude "${library}")
done
echo "===== 开始生成 ${PLATFORM} wheel ====="
auditwheel repair \
    --plat "${PLATFORM}" \
    "${AUDITWHEEL_ARGS[@]}" \
    -w /dist \
    /dist/raw/*.whl
echo "===== linux wheel 到 /dist ====="
rm -rf /dist/raw
echo "===== 全部 wheel ====="
find /dist \
    -maxdepth 1 \
    -type f \
    -name '*.whl' \
    -exec ls -lh {} \;