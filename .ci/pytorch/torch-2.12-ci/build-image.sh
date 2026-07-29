#!/usr/bin/env bash
set -Eeuo pipefail

# 用法：
# bash .ci/pytorch/torch-2.12-ci/build-image.sh \
#   <DTK_PKG_URL> <DOCKER_IMAGE_NAME> [PYTHON_VERSION]

DTK_PKG_URL="${1:?缺少 DTK_PKG_URL}"
DOCKER_IMAGE_NAME="${2:?缺少 DOCKER_IMAGE_NAME}"
PYTHON_VERSION="${3:-3.10}"
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_ROOT="$(cd "${CI_DIR}/../../.." && pwd)"
TEMP_BASE="${RUNNER_TEMP:-/tmp}"
BUILD_CONTEXT="$(mktemp -d "${TEMP_BASE}/pytorch-2.12-image.XXXXXX")"

cleanup() {
    rc=$?
    if [[ "${KEEP_BUILD_CONTEXT:-0}" == "1" ]]; then
        echo "保留构建目录：${BUILD_CONTEXT}"
    else
        rm -rf "${BUILD_CONTEXT}"
    fi
    exit "${rc}"
}
trap cleanup EXIT

for file in Dockerfile prepare.sh build.sh; do
    [[ -f "${CI_DIR}/${file}" ]] || {
        echo "ERROR: 缺少 ${CI_DIR}/${file}" >&2
        exit 1
    }
done

[[ -f "${PYTORCH_ROOT}/version.txt" ]] || {
    echo "ERROR: 没有找到 ${PYTORCH_ROOT}/version.txt" >&2
    exit 1
}
TORCH_SOURCE_VERSION="$(tr -d '[:space:]' < "${PYTORCH_ROOT}/version.txt")"
[[ "${TORCH_SOURCE_VERSION}" == 2.12* ]] || {
    echo "ERROR: 当前脚本只支持 PyTorch 2.12，实际为 ${TORCH_SOURCE_VERSION}" >&2
    exit 1
}

echo "===== Build information ====="
echo "PyTorch root : ${PYTORCH_ROOT}"
echo "PyTorch      : ${TORCH_SOURCE_VERSION}"
echo "DTK package  : ${DTK_PKG_URL}"
echo "Python       : ${PYTHON_VERSION}"
echo "Image        : ${DOCKER_IMAGE_NAME}"
echo "Context      : ${BUILD_CONTEXT}"

# 1. 复制 Dockerfile 和脚本
cp "${CI_DIR}/Dockerfile" "${BUILD_CONTEXT}/Dockerfile"
cp "${CI_DIR}/prepare.sh" "${BUILD_CONTEXT}/prepare.sh"
cp "${CI_DIR}/build.sh" "${BUILD_CONTEXT}/build.sh"
sed -i 's/\r$//' "${BUILD_CONTEXT}/prepare.sh" "${BUILD_CONTEXT}/build.sh"
chmod 755 "${BUILD_CONTEXT}/prepare.sh" "${BUILD_CONTEXT}/build.sh"

# 2. 复制当前 GitHub 仓库中的 PyTorch 源码
mkdir -p "${BUILD_CONTEXT}/pytorch"
tar \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./dist' \
    --exclude='./artifacts' \
    --exclude='./build_logs' \
    -C "${PYTORCH_ROOT}" \
    -cf - . | tar -C "${BUILD_CONTEXT}/pytorch" -xf -

# 3. 下载并解压 DTK 26.04
DTK_ARCHIVE="${BUILD_CONTEXT}/dtk.tar.gz"
DTK_EXTRACT_DIR="${BUILD_CONTEXT}/dtk-extract"
mkdir -p "${DTK_EXTRACT_DIR}"

if [[ "${DTK_PKG_URL}" =~ ^https?:// ]]; then
    wget --tries=3 --timeout=120 -O "${DTK_ARCHIVE}" "${DTK_PKG_URL}"
else
    [[ -f "${DTK_PKG_URL}" ]] || {
        echo "ERROR: 本地 DTK 包不存在：${DTK_PKG_URL}" >&2
        exit 1
    }
    cp "${DTK_PKG_URL}" "${DTK_ARCHIVE}"
fi

tar -xzf "${DTK_ARCHIVE}" -C "${DTK_EXTRACT_DIR}"

DTK_VERSION_FILE="$(find "${DTK_EXTRACT_DIR}" -type f -path '*/.info/rocm_version' -print -quit)"
[[ -n "${DTK_VERSION_FILE}" ]] || {
    echo "ERROR: DTK 包中没有找到 .info/rocm_version" >&2
    find "${DTK_EXTRACT_DIR}" -maxdepth 3 -print
    exit 1
}

DTK_ROOT="${DTK_VERSION_FILE%/.info/rocm_version}"
DTK_VERSION="$(tr -d '[:space:]' < "${DTK_VERSION_FILE}")"
[[ "${DTK_VERSION}" == 26.04* ]] || {
    echo "ERROR: 要求 DTK 26.04，实际下载到 ${DTK_VERSION}" >&2
    exit 1
}
mkdir -p "${BUILD_CONTEXT}/dtk"
cp -a "${DTK_ROOT}/." "${BUILD_CONTEXT}/dtk/"
rm -rf "${DTK_EXTRACT_DIR}" "${DTK_ARCHIVE}"

echo "===== DTK in build context ====="
cat "${BUILD_CONTEXT}/dtk/.info/rocm_version"

# 4. 修改临时 Dockerfile：先删除基础镜像可能已有的 /opt/dtk，避免与 25.04 合并
python3 - "${BUILD_CONTEXT}/Dockerfile" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
replacement = '''RUN rm -rf "${ROCM_PATH}"
COPY dtk/ ${ROCM_PATH}/
RUN actual_dtk="$(tr -d '[:space:]' < "${ROCM_PATH}/.info/rocm_version")" && \\
    echo "DTK version installed in image: ${actual_dtk}" && \\
    [[ "${actual_dtk}" == 26.04* ]]
'''
text, count = re.subn(
    r'(?m)^\s*COPY\s+dtk/?\s+[^\n]+\n',
    replacement,
    text,
    count=1,
)
if count != 1:
    raise SystemExit('ERROR: Dockerfile 中没有找到 COPY dtk 这一行')
path.write_text(text)
PY
echo "===== Patched Dockerfile DTK section ====="
grep -A7 -B2 'DTK version installed' "${BUILD_CONTEXT}/Dockerfile"

# 5. 构建镜像
docker build \
    --network=host \
    --progress=plain \
    --no-cache \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "DTK_VERSION=2604" \
    --build-arg "INSTALL_TORCH_VERSION=2.12.0" \
    -f "${BUILD_CONTEXT}/Dockerfile" \
    -t "${DOCKER_IMAGE_NAME}" \
    "${BUILD_CONTEXT}"
# 6. 构建后再次确认镜像内的 DTK 版本
echo "===== DTK in built image ====="
docker run --rm "${DOCKER_IMAGE_NAME}" \
    bash -lc 'cat /opt/dtk/.info/rocm_version'

echo "镜像构建完成：${DOCKER_IMAGE_NAME}"