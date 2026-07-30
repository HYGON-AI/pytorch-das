#!/bin/bash

set -euo pipefail

TOOLKIT_DIR="${TOOLKIT_DIR:-/pytorch}"
ROCM_PATH="${ROCM_PATH:-/opt/dtk}"
RESOURCE_SERVER="${RESOURCE_SERVER:-http://42.228.13.241:18000}"

PYTORCH_VERSION_FILE="${TOOLKIT_DIR}/version.txt"
DTK_VERSION_FILE="${ROCM_PATH}/.info/rocm_version"

MKL_LIB_DIR="/opt/intel/mkl/lib/intel64_lin"

log() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}


fail() {
    echo "ERROR: $1" >&2
    exit 1
}

# ----------------------------
# 检查基础目录
# ----------------------------

log "检查基础环境"

[[ -d "${TOOLKIT_DIR}" ]] || fail "PyTorch 源码目录不存在：${TOOLKIT_DIR}"
[[ -d "${ROCM_PATH}" ]] || fail "DTK目录不存在：${ROCM_PATH}"
[[ -f "${PYTORCH_VERSION_FILE}" ]] || fail "版本文件不存在：${PYTORCH_VERSION_FILE}"
[[ -f "${DTK_VERSION_FILE}" ]] || fail "DTK版本文件不存在：${DTK_VERSION_FILE}"

pytorch_version="$(tr -d '[:space:]' < "${PYTORCH_VERSION_FILE}")"
dtk_version="$(head -n 1 "${DTK_VERSION_FILE}" | tr -d '[:space:]')"
os_name="$(
    grep '^NAME=' /etc/os-release |
        cut -d '"' -f 2
)"

echo "PyTorch source : ${TOOLKIT_DIR}"
echo "PyTorch version: ${pytorch_version}"
echo "DTK path        : ${ROCM_PATH}"
echo "DTK version     : ${dtk_version}"
echo "OS              : ${os_name}"

# ----------------------------
# 限制版本
# ----------------------------

if [[ "${pytorch_version}" != "2.12.0" ]]; then
    fail "当前 GitHub 编译流程只支持 PyTorch 2.12.0，实际版本为 ${pytorch_version}"
fi

if [[ "${dtk_version}" != *"26.04"* ]]; then
    fail "当前脚本按照 DTK 26.04 编写，实际版本为 ${dtk_version}"
fi


# ----------------------------
# 安装 MKL
# ----------------------------

log "检查 MKL"

if [[ -d "${MKL_LIB_DIR}" ]]; then
    echo "MKL目录已经存在：${MKL_LIB_DIR}"
else
    echo "MKL目录不存在，开始安装 Intel MKL 2020"

    yum install -y ca-certificates yum-utils
    update-ca-trust

    yum-config-manager \
        --add-repo \
        https://yum.repos.intel.com/mkl/setup/intel-mkl.repo

    yum install -y \
        intel-mkl-2020.0-088 \
        --nogpgcheck
fi

[[ -d "${MKL_LIB_DIR}" ]] || fail "MKL安装完成后仍未找到：${MKL_LIB_DIR}"


# ----------------------------
# 安装 Python 依赖
# ----------------------------

log "安装 PyTorch 编译依赖"

python3 -m pip install \
    -r "${TOOLKIT_DIR}/.ci/docker/requirements-ci.txt"

python3 -m pip install \
    -r "${TOOLKIT_DIR}/requirements.txt"

python3 -m pip install \
    pyyaml \
    MonkeyType \
    packaging \
    einops \
    hypothesis==5.35.1

python3 -m pip install --no-cache-dir \
    ninja==1.11.1 \
    cmake==3.29.2 \
    xdoctest==1.1.3 \
    urllib3 \
    pytest==8.1.1 \
    Pillow==10.0.1 \
    numpy==1.24.3 \
    networkx \
    mypy==1.8.0 \
    jmespath==1.0.1 \
    Jinja2==3.1.3 \
    hypothesis \
    expecttest==0.2.1 \
    scipy \
    sympy==1.12.1 \
    tlparse==0.3.5 \
    botocore \
    numba \
    amdsmi==1.0.0+630c16a6.dirty \
    cloudpickle

# ----------------------------
# 下载工具函数
# ----------------------------

download_file() {
    local url="$1"
    local output="$2"
    echo "下载：${url}"
    wget \
        --quiet \
        --show-progress \
        --tries=3 \
        --timeout=60 \
        -O "${output}" \
        "${url}"
}

download_and_unzip() {
    local url="$1"
    local target_dir="$2"
    local archive_name

    archive_name="/tmp/$(basename "${url}")"
    mkdir -p "${target_dir}"
    download_file "${url}" "${archive_name}"
    echo "解压 ${archive_name} 到 ${target_dir}"
    unzip -q \
        -o \
        "${archive_name}" \
        -d "${target_dir}"

    rm -f "${archive_name}"
}
# 准备 MAGMA

log "准备 DTK 26.04 MAGMA"

MAGMA_ARCHIVE="/tmp/magma-install.tar.gz"
MAGMA_URL="${RESOURCE_SERVER}/thirdparty/pytorch/magma-install.tar.gz"

rm -rf "${ROCM_PATH}/magma"

download_file \
    "${MAGMA_URL}" \
    "${MAGMA_ARCHIVE}"

tar -zxf \
    "${MAGMA_ARCHIVE}" \
    -C "${ROCM_PATH}"

rm -f "${MAGMA_ARCHIVE}"

if [[ -d "${ROCM_PATH}/magma-install" ]]; then
    mv \
        "${ROCM_PATH}/magma-install" \
        "${ROCM_PATH}/magma"
fi

[[ -d "${ROCM_PATH}/magma/lib" ]] ||
    fail "MAGMA目录不存在：${ROCM_PATH}/magma/lib"

# 将 MKL 动态库放入 MAGMA
log "拷贝 MKL 动态库到 MAGMA"
mkl_libraries=(
    "libmkl_core.so"
    "libmkl_gf_lp64.so"
    "libmkl_gnu_thread.so"
    "libmkl_def.so"
    "libmkl_vml_def.so"
)

for library in "${mkl_libraries[@]}"; do
    source_file="${MKL_LIB_DIR}/${library}"

    [[ -f "${source_file}" ]] ||
        fail "缺少 MKL 动态库：${source_file}"

    cp -vf \
        "${source_file}" \
        "${ROCM_PATH}/magma/lib/"
done
# 下载 Flash Attention 文件
log "准备 Flash Attention 文件"

FLASH_ATTN_URL="${RESOURCE_SERVER}/thirdparty/flash_attn/cutlass_fa_so/rocky/python310/20260107_nmz_tmp/flash_attn_2_cuda.so"
FLASH_ATTN_FILE="/opt/flash_attn_2_cuda.so"

download_file \
    "${FLASH_ATTN_URL}" \
    "${FLASH_ATTN_FILE}"

[[ -f "${FLASH_ATTN_FILE}" ]] ||
    fail "Flash Attention 文件下载失败：${FLASH_ATTN_FILE}"

# 准备 PyTorch third_party
INSTALL_DIR_URL="${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.4.1/install_dir.zip"
THIRD_PARTY_URL="${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip"
FBJNI_URL="${RESOURCE_SERVER}/thirdparty/pytorch/pytorch2.12/fbjni.zip"

cd "${TOOLKIT_DIR}"
rm -rf \
    "${TOOLKIT_DIR}/third_party" \
    "${TOOLKIT_DIR}/android/libs/fbjni"
download_and_unzip \
    "${INSTALL_DIR_URL}" \
    "${TOOLKIT_DIR}"
download_and_unzip \
    "${THIRD_PARTY_URL}" \
    "${TOOLKIT_DIR}"
download_and_unzip \
    "${FBJNI_URL}" \
    "${TOOLKIT_DIR}/android/libs"

# ----------------------------
# 检查依赖目录
# ----------------------------

log "检查 PyTorch 依赖目录"

required_paths=(
    "${TOOLKIT_DIR}/third_party"
    "${TOOLKIT_DIR}/third_party/miniz-3.0.2"
    "${TOOLKIT_DIR}/third_party/gloo/tools/amd_build/build_amd.py"
    "${TOOLKIT_DIR}/android/libs/fbjni"
)

for required_path in "${required_paths[@]}"; do
    [[ -e "${required_path}" ]] ||
        fail "缺少 PyTorch 依赖：${required_path}"

    echo "已找到：${required_path}"
done

chmod 755 \
    "${TOOLKIT_DIR}/third_party/gloo/tools/amd_build/build_amd.py"

# ----------------------------
# 显示 Git 信息
# ----------------------------
log "检查 Git 和源码状态"
cd "${TOOLKIT_DIR}"
echo "源码目录："
ls -la | head -n 20
echo "Git目录："
ls -la .git 2>/dev/null || true

echo "Submodule状态："
git submodule status 2>/dev/null || true

# ----------------------------
# 修正 DTK 版本文件
# ----------------------------
log "修正 DTK version-dev 和 version-utils"

mkdir -p "${ROCM_PATH}/.info"

printf '%s\n' "5.4.3-59" \
    > "${ROCM_PATH}/.info/version-dev"

printf '%s\n' "5.4.3-59" \
    > "${ROCM_PATH}/.info/version-utils"

# ----------------------------
# 替换 ROCm 默认路径
# ----------------------------

log "替换源码中的 /opt/rocm 路径"

path_files=(
    "${TOOLKIT_DIR}/cmake/Dependencies.cmake"
    "${TOOLKIT_DIR}/third_party/kineto/libkineto/CMakeLists.txt"
    "${TOOLKIT_DIR}/third_party/gloo/cmake/Modules/Findrccl.cmake"
)

for path_file in "${path_files[@]}"; do
    [[ -f "${path_file}" ]] ||
        fail "需要修改的文件不存在：${path_file}"

    sed -i \
        's#/opt/rocm#${ROCM_PATH}#g' \
        "${path_file}"
done

# ----------------------------
# 拷贝 MKL 和 MAGMA 到 torch/lib
# ----------------------------
log "拷贝编译所需动态库到 torch/lib"

TORCH_LIB_DIR="${TOOLKIT_DIR}/torch/lib"

mkdir -p "${TORCH_LIB_DIR}"

torch_mkl_libraries=(
    "libmkl_core.so"
    "libmkl_gf_lp64.so"
    "libmkl_gnu_thread.so"
    "libmkl_def.so"
    "libmkl_intel_lp64.so"
    "libmkl_vml_def.so"
)

for library in "${torch_mkl_libraries[@]}"; do
    source_file="${MKL_LIB_DIR}/${library}"

    [[ -f "${source_file}" ]] ||
        fail "缺少 MKL 动态库：${source_file}"

    cp -vf \
        "${source_file}" \
        "${TORCH_LIB_DIR}/"
done

[[ -f "${ROCM_PATH}/magma/lib/libmagma.so" ]] ||
    fail "缺少 MAGMA 动态库：${ROCM_PATH}/magma/lib/libmagma.so"

cp -vf \
    "${ROCM_PATH}/magma/lib/libmagma.so" \
    "${TORCH_LIB_DIR}/"

# ----------------------------
# 执行 AMD 源码生成脚本
# ----------------------------

log "执行 PyTorch AMD build 脚本"

cd "${TOOLKIT_DIR}"

python3 \
    tools/amd_build/build_amd.py
# ----------------------------
# 检查
# ----------------------------
log "PyTorch 2.12 编译环境准备完成"
echo "Python版本："
python3 --version

echo "GCC版本："
gcc --version | head -n 1

echo "CMake版本："
cmake --version | head -n 1

echo "Ninja版本："
ninja --version

echo "ROCM_PATH=${ROCM_PATH}"
echo "TOOLKIT_DIR=${TOOLKIT_DIR}"
echo "PyTorch version=${pytorch_version}"
echo "DTK version=${dtk_version}"