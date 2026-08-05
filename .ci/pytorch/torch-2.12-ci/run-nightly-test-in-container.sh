#!/usr/bin/env bash

set -Eeuo pipefail

source_optional_environment() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${path}"
        set -u
    fi
}

source_optional_environment /opt/dtk/env.sh
source_optional_environment /opt/rh/gcc-toolset-13/enable

export PATH="/opt/dtk/bin:/opt/dtk/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/2024.2/lib:/opt/intel/oneapi/mkl/2024.2/lib/intel64:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/usr/local/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /pytorch
shopt -s nullglob
torch_wheels=(dist/*.whl)
shopt -u nullglob
if (( ${#torch_wheels[@]} != 1 )); then
    echo "ERROR: expected exactly one test wheel in /pytorch/dist, found ${#torch_wheels[@]}" >&2
    find /pytorch/dist -maxdepth 2 -type f -name '*.whl' -ls || true
    exit 1
fi
torch_wheel="/pytorch/${torch_wheels[0]}"

# All dependencies are prepared in the image. Avoid resolving an incompatible
# public ROCm/CUDA dependency set while installing the wheel under test.
cd /
python3 -m pip install --no-deps --force-reinstall "${torch_wheel}"

torch_path="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
export LD_LIBRARY_PATH="${torch_path}/lib:${LD_LIBRARY_PATH}"

python3 - <<'PY'
import torch

print("Installed torch:", torch.__version__)
print("Torch path:", torch.__file__)
print("HIP version:", torch.version.hip)
print("DCU available:", torch.cuda.is_available())
print("DCU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("The installed wheel cannot access a DCU device")
if torch.version.hip is None:
    raise RuntimeError("The installed wheel does not report a HIP runtime")
PY

if [[ "${TEST_CONFIG}" == "smoke" ]]; then
    # Do not run Python from /pytorch: the source-tree torch package would
    # shadow the wheel installed in site-packages.
    cd /
    touch /tmp/pytorch-nightly-test-started
    python3 - <<'PY'
import torch

x = torch.randn(32, 32, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Device:", torch.cuda.get_device_name(0))
print("Smoke result:", y.norm().item())
PY
else
    # Same test entry point used by release/2.12 _rocm-test.yml.
    cd /pytorch
    test -x .ci/pytorch/test.sh
    touch /tmp/pytorch-nightly-test-started
    .ci/pytorch/test.sh
fi
