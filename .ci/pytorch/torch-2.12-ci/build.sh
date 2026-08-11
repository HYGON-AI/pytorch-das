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
# CIUpload 0.3.0 根据这些构建信息生成 wheel 本地版本和包名。
BASE_VERSION="${TORCH_VERSION:-$(tr -d '[:space:]' < /pytorch/version.txt)}"
DTK_VERSION_NUMBER="${DTK_VERSION:-$(
    tr -d '.[:space:]' < /opt/dtk/.info/rocm_version
)}"
if ! [[ "${BASE_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: invalid TORCH_VERSION=${BASE_VERSION}" >&2
    exit 1
fi
if ! [[ "${DTK_VERSION_NUMBER}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid DTK_VERSION=${DTK_VERSION_NUMBER}" >&2
    exit 1
fi
COMMIT_ID="${GIT_COMMIT_SHORT:-}"
if [[ -z "${COMMIT_ID}" ]]; then
    if git -C /pytorch rev-parse HEAD >/dev/null 2>&1; then
        COMMIT_ID="$(git -C /pytorch rev-parse --short=6 HEAD)"
    else
        COMMIT_ID="unknown"
    fi
fi
echo "BASE_VERSION=${BASE_VERSION}"
echo "DTK_VERSION_NUMBER=${DTK_VERSION_NUMBER}"
echo "COMMIT_ID=${COMMIT_ID}"
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
BUILD_TEST=ON \
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
mkdir -p /dist/source /dist/raw
# 保存原始 linux wheel
cp -v "${raw_wheels[@]}" /dist/source/

echo "===== 修正 wheel 版本 ====="
CIUPLOAD_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("CIUpload"))')"
if [[ "${CIUPLOAD_VERSION}" != "0.3.0" ]]; then
    echo "ERROR: expected CIUpload 0.3.0, found ${CIUPLOAD_VERSION}" >&2
    exit 1
fi
CIUpload REPAIR \
    --dtk_version "${DTK_VERSION_NUMBER}" \
    --torch_version "${BASE_VERSION}" \
    --git_commit "${COMMIT_ID}" \
    --outputdir /dist/raw \
    -f /dist/source/*.whl
rm -rf /dist/source
echo "===== 修正后的 linux wheel ====="
shopt -s nullglob
repaired_linux_wheels=(/dist/raw/*.whl)
shopt -u nullglob
if (( ${#repaired_linux_wheels[@]} != 1 )); then
    echo "ERROR: expected one CIUpload-repaired wheel, found ${#repaired_linux_wheels[@]}" >&2
    find /dist/raw -maxdepth 1 -type f -ls || true
    exit 1
fi
ls -lh "${repaired_linux_wheels[0]}"
echo "===== 安装修正后的 wheel ====="
python3 -m pip install \
    --no-deps \
    --force-reinstall \
    "${repaired_linux_wheels[0]}"
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
shopt -s nullglob
linux_wheels=(/dist/raw/*.whl)
shopt -u nullglob

if (( ${#linux_wheels[@]} != 1 )); then
    echo "ERROR: /dist/raw 中 Linux wheel 数量异常：${#linux_wheels[@]}"
    find /dist/raw -maxdepth 1 -type f -ls || true
    exit 1
fi

linux_wheel="${linux_wheels[0]}"
auditwheel repair \
    --plat "${PLATFORM}" \
    "${AUDITWHEEL_ARGS[@]}" \
    -w /dist \
    "${linux_wheel}"
echo "===== 保留 Linux wheel 到 /dist ====="
mv -v "${linux_wheel}" /dist/
rmdir /dist/raw
echo "===== 全部 wheel ====="
shopt -s nullglob
final_wheels=(/dist/*.whl)
shopt -u nullglob
if (( ${#final_wheels[@]} != 2 )); then
    echo "ERROR: 预期生成 2 个 wheel，实际生成 ${#final_wheels[@]} 个"
    find /dist -maxdepth 1 -type f -ls || true
    exit 1
fi
printf '最终 wheel:\n'
printf '  %s\n' "${final_wheels[@]}"
ls -lh "${final_wheels[@]}"

echo "===== Build custom test artifacts ====="
CUSTOM_TEST_ARTIFACT_BUILD_DIR="${TOOLKIT_DIR}/build/custom_test_artifacts"
CUSTOM_TEST_MODULE_PATH="${TOOLKIT_DIR}/cmake/public"
SITE_PACKAGES="$({
    cd /
    python3 -c 'import site; print(";".join([x for x in site.getsitepackages()] + [x + "/torch" for x in site.getsitepackages()]))'
})"

rm -rf "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"
mkdir -p "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

build_custom_test() {
    local source_dir="$1"
    local build_dir="$2"

    cmake \
        -S "${source_dir}" \
        -B "${build_dir}" \
        -DCMAKE_PREFIX_PATH="${SITE_PACKAGES}" \
        -DPython_EXECUTABLE="$(command -v python3)" \
        -DCMAKE_MODULE_PATH="${CUSTOM_TEST_MODULE_PATH}" \
        -DUSE_ROCM=ON
    cmake --build "${build_dir}" --parallel "${MAX_JOBS:-4}"
}

build_custom_test \
    "${TOOLKIT_DIR}/test/custom_operator" \
    "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-op-build"
build_custom_test \
    "${TOOLKIT_DIR}/test/jit_hooks" \
    "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/jit-hook-build"
build_custom_test \
    "${TOOLKIT_DIR}/test/custom_backend" \
    "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-backend-build"

echo "===== Export test timing data ====="
cd "${TOOLKIT_DIR}"
PYTHONPATH=. python3 tools/stats/export_test_times.py
