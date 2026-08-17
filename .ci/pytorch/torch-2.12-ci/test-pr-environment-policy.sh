#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/../../.." && pwd)"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_ROOT}"' EXIT

skip_file="${CI_DIR}/pr-skipped-tests.txt"
active_skip_count="$(grep -Ev '^[[:space:]]*(#|$)' "${skip_file}" | wc -l)"
test "${active_skip_count}" = "7"

required_clauses=(
    "TestNN and test_batchnorm_2D_train_NCHW_vs_cpu_mixed_bfloat16"
    "TestNN and test_batchnorm_2D_train_NHWC_vs_NCHW_mixed_bfloat16"
    "TestNN and test_batchnorm_3D_train_NCHW_vs_cpu_mixed_bfloat16"
    "TestNN and test_batchnorm_3D_train_NHWC_vs_NCHW_mixed_bfloat16"
    "TestTorchDeviceTypeCUDA and test_cdist_large_cuda"
    "TestGDS and test_gds_read_write_tensors"
    "TestMatmulCudaCUDA and test_grouped_gemm_rocm_ck_flag_cuda"
)
for clause in "${required_clauses[@]}"; do
    grep -Fxq "${clause}" "${skip_file}"
done

# Unmarked implementation/numerical failures must remain visible to R&D.
for retained_test in \
    test_conv_transposed_large_cuda \
    test_depthwise_conv_64bit_indexing_cuda \
    test_batchnorm_2D_train_NCHW_vs_cpu_mixed_float16 \
    test_bincount_cuda \
    test_bincount_ext \
    test_compare_cpu_bincount_cuda_int64; do
    if grep -Fq "${retained_test}" "${skip_file}"; then
        echo "ERROR: unapproved test was added to the PR skip expression: ${retained_test}" >&2
        exit 1
    fi
done

miopen_variable="MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"
if grep -Fq "${miopen_variable}" "${CI_DIR}/run-nightly-tests.sh"; then
    echo "ERROR: ${miopen_variable} must not be set for the whole test container" >&2
    exit 1
fi
if ! grep -B6 -F 'def test_conv_transposed_large' \
    "${REPO_ROOT}/test/nn/test_convolution.py" | \
    grep -Fq "${miopen_variable}"; then
    echo "ERROR: ${miopen_variable} must be scoped to test_conv_transposed_large" >&2
    exit 1
fi

if grep -B5 -F 'def test_multiprocessing_iterdatapipe_with_dill' \
    "${REPO_ROOT}/test/test_dataloader.py" | grep -Fq '@unittest.expectedFailure'; then
    echo "ERROR: DataLoader dill test still has an obsolete expectedFailure decorator" >&2
    exit 1
fi

mkdir -p "${TEMP_ROOT}/bin"
cat > "${TEMP_ROOT}/hy-smi" <<'HY_SMI'
#!/usr/bin/env bash
printf '%s\n' 'header' '0 DCU-a' '1 DCU-b' 'footer'
HY_SMI
chmod 755 "${TEMP_ROOT}/hy-smi"

AMD_SMI_COMPAT_PATH="${TEMP_ROOT}/bin/amd-smi" \
bash "${CI_DIR}/install-test-compat.sh"
test -x "${TEMP_ROOT}/bin/amd-smi"
device_lines="$(HY_SMI_PATH="${TEMP_ROOT}/hy-smi" "${TEMP_ROOT}/bin/amd-smi" list)"
test "$(grep -c '^GPU ' <<< "${device_lines}")" = "2"

echo "PR skip and DCU environment policy passed"
