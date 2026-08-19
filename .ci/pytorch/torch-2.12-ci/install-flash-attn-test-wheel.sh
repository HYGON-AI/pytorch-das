#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_TEST_INDEX_URL="${FLASH_ATTN_TEST_INDEX_URL:?FLASH_ATTN_TEST_INDEX_URL is required}"
FLASH_ATTN_TEST_VERSION="${FLASH_ATTN_TEST_VERSION:?FLASH_ATTN_TEST_VERSION is required}"
FLASH_ATTN_TORCH_TAG="${FLASH_ATTN_TORCH_TAG:?FLASH_ATTN_TORCH_TAG is required}"
FLASH_ATTN_LINK_PATH="${FLASH_ATTN_LINK_PATH:-/opt/flash_attn_2_cuda.so}"
FLASH_ATTN_RUNTIME_PATH_FILE="${FLASH_ATTN_RUNTIME_PATH_FILE:-/opt/flash_attn_runtime_library_path}"

work_dir="$(mktemp -d /tmp/pytorch-flash-attn.XXXXXX)"
cleanup() {
    rm -rf -- "${work_dir}"
}
trap cleanup EXIT

wheel_path="$(python3 "${SCRIPT_DIR}/resolve_flash_attn_wheel.py" \
    --index-url "${FLASH_ATTN_TEST_INDEX_URL}" \
    --version "${FLASH_ATTN_TEST_VERSION}" \
    --torch-tag "${FLASH_ATTN_TORCH_TAG}" \
    --dest "${work_dir}")"

[[ -s "${wheel_path}" ]] || {
    echo "ERROR: resolved Flash Attention wheel is missing: ${wheel_path}" >&2
    exit 1
}

echo "Installing selected Flash Attention wheel: ${wheel_path}"
sha256sum "${wheel_path}"
python3 -m pip install \
    --no-cache-dir \
    --no-deps \
    --force-reinstall \
    "${wheel_path}"

mkdir -p "$(dirname "${FLASH_ATTN_RUNTIME_PATH_FILE}")"
flash_so_path="$(FLASH_ATTN_EXPECTED_TORCH_TAG="${FLASH_ATTN_TORCH_TAG}" \
    FLASH_ATTN_RUNTIME_PATH_FILE="${FLASH_ATTN_RUNTIME_PATH_FILE}" \
    python3 - <<'PY'
import importlib.metadata
import os
from pathlib import Path


distribution = importlib.metadata.distribution("flash-attn")
expected_torch_tag = os.environ["FLASH_ATTN_EXPECTED_TORCH_TAG"].lower()
installed_version = distribution.version.lower()
if expected_torch_tag not in installed_version:
    raise RuntimeError(
        f"installed flash-attn {distribution.version!r} does not contain "
        f"the required {expected_torch_tag!r} tag"
    )

installed_files = {
    Path(distribution.locate_file(entry)).resolve(): Path(entry).name
    for entry in distribution.files or ()
}
libraries = sorted(
    {
        path
        for path, name in installed_files.items()
        if name.startswith("flash_attn_2_cuda") and path.suffix == ".so"
    }
)
if len(libraries) != 1:
    formatted = ", ".join(str(path) for path in libraries) or "none"
    raise RuntimeError(
        "expected exactly one flash_attn_2_cuda*.so in the selected wheel, "
        f"found {len(libraries)}: {formatted}"
    )

flash_runtime_libraries = sorted(
    path
    for path, name in installed_files.items()
    if name == "libflash_attention.so"
)
if len(flash_runtime_libraries) != 1:
    formatted = ", ".join(str(path) for path in flash_runtime_libraries) or "none"
    raise RuntimeError(
        "expected exactly one libflash_attention.so in the selected wheel, "
        f"found {len(flash_runtime_libraries)}: {formatted}"
    )

shared_library_directories = sorted(
    {
        path.parent
        for path, name in installed_files.items()
        if ".so" in name
    }
)
runtime_path_file = Path(os.environ["FLASH_ATTN_RUNTIME_PATH_FILE"])
runtime_path_file.write_text(
    ":".join(str(path) for path in shared_library_directories) + "\n",
    encoding="utf-8",
)

print(libraries[0])
PY
)"

[[ -s "${flash_so_path}" ]] || {
    echo "ERROR: installed Flash Attention library is missing: ${flash_so_path}" >&2
    exit 1
}

mkdir -p "$(dirname "${FLASH_ATTN_LINK_PATH}")"
ln -sfn "${flash_so_path}" "${FLASH_ATTN_LINK_PATH}"

torch_lib_path="$(cd / && python3 - <<'PY'
import importlib.util
from pathlib import Path


spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("cannot locate the installed torch package")
print(Path(next(iter(spec.submodule_search_locations))).resolve() / "lib")
PY
)"
[[ -d "${torch_lib_path}" ]] || {
    echo "ERROR: installed torch library directory is missing: ${torch_lib_path}" >&2
    exit 1
}

flash_runtime_library_path="$(< "${FLASH_ATTN_RUNTIME_PATH_FILE}")"
[[ -n "${flash_runtime_library_path}" ]] || {
    echo "ERROR: Flash Attention runtime library path is empty" >&2
    exit 1
}

libomp_path="$(find /opt/dtk \
    \( -type f -o -type l \) \
    -name 'libomp.so' -print -quit)"
[[ -n "${libomp_path}" ]] || {
    echo "ERROR: unable to locate libomp.so below /opt/dtk" >&2
    exit 1
}
flash_runtime_library_path="${flash_runtime_library_path}:$(dirname "${libomp_path}")"
printf '%s\n' \
    "${flash_runtime_library_path}" \
    > "${FLASH_ATTN_RUNTIME_PATH_FILE}"

export LD_LIBRARY_PATH="${flash_runtime_library_path}:${torch_lib_path}:/opt/intel/oneapi/mkl/2024.2/lib:/opt/intel/oneapi/mkl/2024.2/lib/intel64:/opt/dtk/llvm/lib:/opt/dtk/aillvm/lib:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/usr/local/hyhal/lib:${LD_LIBRARY_PATH:-}"
echo "Flash Attention runtime library path: ${flash_runtime_library_path}"

fa_ldd_output="$(ldd "${FLASH_ATTN_LINK_PATH}" 2>&1)"
printf '%s\n' "${fa_ldd_output}"
if grep -Fq 'not found' <<< "${fa_ldd_output}"; then
    echo "ERROR: selected Flash Attention library has unresolved dependencies" >&2
    exit 1
fi

cd /
python3 - <<'PY'
import importlib.metadata

print("Installed flash-attn:", importlib.metadata.version("flash-attn"))
PY

echo "Flash Attention extension prepared for HCU validation: ${flash_so_path}"
echo "Flash Attention compatibility link: ${FLASH_ATTN_LINK_PATH} -> ${flash_so_path}"
