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

print_pr_timeout_diagnostics() {
    local cache_root="/pytorch/.pytest_cache/v/cache/stepcurrent"
    local cache_path
    local timestamp

    echo "===== PR shard timeout diagnostics =====" >&2
    echo "Most recently started pytest cases (newest first):" >&2
    if [[ -d "${cache_root}" ]]; then
        while read -r timestamp cache_path; do
            [[ -n "${cache_path:-}" ]] || continue
            printf '%s: ' "${cache_path}" >&2
            cat "${cache_path}" >&2 || true
            printf '\n' >&2
        done < <(
            find "${cache_root}" -type f -name lastrun -printf '%T@ %p\n' |
                sort -nr | sed -n '1,20p'
        )
    else
        echo "No pytest stepcurrent cache was created." >&2
    fi
    echo "Remaining test-related processes:" >&2
    ps -eo pid,ppid,stat,etime,args | \
        grep -E 'python|pytest|run_test.py' | grep -v grep >&2 || true
    echo "===== End PR shard timeout diagnostics =====" >&2
}

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

FA_SO_PATH="${FA_SO_PATH:?FA_SO_PATH is required}"
export FA_SO_PATH
if [[ ! -s "${FA_SO_PATH}" ]]; then
    echo "ERROR: Flash Attention library is missing or empty: ${FA_SO_PATH}" >&2
    exit 1
fi
echo "Flash Attention library: ${FA_SO_PATH}"
fa_ldd_output="$(ldd "${FA_SO_PATH}" 2>&1)"
printf '%s\n' "${fa_ldd_output}"
if grep -Fq 'not found' <<< "${fa_ldd_output}"; then
    echo "ERROR: Flash Attention library has unresolved dependencies" >&2
    exit 1
fi

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
    set +e
    python3 - <<'PY'
import torch

x = torch.randn(32, 32, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Device:", torch.cuda.get_device_name(0))
print("Smoke result:", y.norm().item())
PY
    test_rc=$?
    set -e
elif [[ "${TEST_CONFIG}" == "pr" ]]; then
    cd /pytorch
    pr_test_list=".ci/pytorch/torch-2.12-ci/pr-tests.txt"
    pr_skip_list=".ci/pytorch/torch-2.12-ci/pr-skipped-tests.txt"
    test -s "${pr_test_list}"
    test -s "${pr_skip_list}"
    mapfile -t pr_tests < "${pr_test_list}"
    if (( ${#pr_tests[@]} == 0 )); then
        echo "ERROR: PR test list is empty" >&2
        exit 1
    fi
    for test_module in "${pr_tests[@]}"; do
        if [[ -z "${test_module}" || "${test_module}" == \#* ]]; then
            echo "ERROR: PR test list must contain only canonical test module names" >&2
            exit 1
        fi
        test -f "test/${test_module}.py"
    done

    shopt -s nullglob
    torchvision_wheels=(dist/vision/torchvision-*.whl)
    shopt -u nullglob
    if (( ${#torchvision_wheels[@]} != 1 )); then
        echo "ERROR: expected exactly one torchvision wheel, found ${#torchvision_wheels[@]}" >&2
        exit 1
    fi
    python3 -m pip install --no-deps --force-reinstall \
        "/pytorch/${torchvision_wheels[0]}"

    echo "===== PR test selection (${#pr_tests[@]} modules), shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS} ====="
    printf '  %s\n' "${pr_tests[@]}"
    export PYTORCH_DCU_PR_SKIP_FILE="/pytorch/${pr_skip_list}"
    export PYTHONPATH="/pytorch/.ci/pytorch/torch-2.12-ci:${PYTHONPATH:-}"
    # Keep two in-process pytest reruns so transient DCU failures can recover.
    # The outer run_test.py retry count stays at one, which caps each failing
    # case at three executions instead of multiplying it to as many as nine.
    export PYTORCH_NUM_PYTEST_RERUNS=2
    # Preserve keep-going/resume behavior, but avoid starting two additional
    # pytest processes merely to classify the same failure as consistent.
    export PYTORCH_TEST_FAILURE_ATTEMPTS=1
    # Official timeout selection returns None when no historical timing exists.
    # Bound that module subprocess so a hung DCU call can be retried/skipped by
    # run_test.py and the remaining PR modules can still run.
    export PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS=1800
    PR_TEST_TIMEOUT_SECONDS="${PR_TEST_TIMEOUT_SECONDS:-13500}"
    if ! [[ "${PR_TEST_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid PR_TEST_TIMEOUT_SECONDS=${PR_TEST_TIMEOUT_SECONDS}" >&2
        exit 1
    fi
    command -v timeout >/dev/null 2>&1 || {
        echo "ERROR: the timeout command is required for PR tests" >&2
        exit 1
    }
    echo "PR case-level skip policy: ${PYTORCH_DCU_PR_SKIP_FILE}"
    echo "PR pytest reruns per process: ${PYTORCH_NUM_PYTEST_RERUNS}"
    echo "PR attempts per failed case: ${PYTORCH_TEST_FAILURE_ATTEMPTS}"
    echo "PR fallback module timeout: ${PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS}s"
    echo "PR shard hard timeout: ${PR_TEST_TIMEOUT_SECONDS}s"
    touch /tmp/pytorch-nightly-test-started
    pr_test_command=(
        python3 test/run_test.py
        --include "${pr_tests[@]}"
        --exclude-jit-executor
        --exclude-distributed-tests
        --exclude-quantization-tests
        --shard "${SHARD_NUMBER}" "${NUM_TEST_SHARDS}"
        --verbose
        --keep-going
        --
        -p dcu_pr_skip_plugin
    )
    set +e
    timeout --signal=TERM --kill-after=60s \
        "${PR_TEST_TIMEOUT_SECONDS}s" \
        "${pr_test_command[@]}"
    test_rc=$?
    set -e
    if (( test_rc == 124 || test_rc == 137 )); then
        echo \
            "ERROR: PR shard ${SHARD_NUMBER}/${NUM_TEST_SHARDS} exceeded " \
            "${PR_TEST_TIMEOUT_SECONDS}s" >&2
        print_pr_timeout_diagnostics
    fi
else
    # Same test entry point used by release/2.12 _rocm-test.yml.
    cd /pytorch
    test -x .ci/pytorch/test.sh
    touch /tmp/pytorch-nightly-test-started
    set +e
    .ci/pytorch/test.sh
    test_rc=$?
    set -e
fi

printf '%s\n' "${test_rc}" > /tmp/pytorch-nightly-test-exit-code
touch /tmp/pytorch-nightly-test-completed
exit "${test_rc}"
