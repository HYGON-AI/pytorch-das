#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/../../.." && pwd)"
TEST_RUNNER="${CI_DIR}/run-nightly-test-in-container.sh"
WORKFLOW="${REPO_ROOT}/.github/workflows/_pytorch-2.12-hcu-test.yml"

grep -Fq 'export PYTORCH_NUM_PYTEST_RERUNS=2' "${TEST_RUNNER}"
grep -Fq 'export PYTORCH_TEST_FAILURE_ATTEMPTS=1' "${TEST_RUNNER}"
grep -Fq 'export PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS=1800' "${TEST_RUNNER}"
grep -Fq 'PR shard hard timeout: disabled' "${TEST_RUNNER}"
if grep -Fq 'PR_TEST_TIMEOUT_SECONDS' "${TEST_RUNNER}"; then
    echo "ERROR: PR shard hard timeout must remain disabled" >&2
    exit 1
fi
if grep -Fq 'timeout --signal=TERM' "${TEST_RUNNER}"; then
    echo "ERROR: PR test command must not use an outer timeout" >&2
    exit 1
fi
grep -Fq 'PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS' \
    "${REPO_ROOT}/test/run_test.py"
grep -Fq 'PYTORCH_TEST_FAILURE_ATTEMPTS' "${REPO_ROOT}/test/run_test.py"
grep -Fq 'timeout-minutes: 4320' "${WORKFLOW}"
if grep -Fq "inputs.test_suite == 'pr' && 270" "${WORKFLOW}"; then
    echo "ERROR: PR job must not use the old 270-minute timeout" >&2
    exit 1
fi

echo "PR retry and no-shard-deadline policy passed"
