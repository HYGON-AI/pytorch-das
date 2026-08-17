#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/../../.." && pwd)"
TEST_RUNNER="${CI_DIR}/run-nightly-test-in-container.sh"
WORKFLOW="${REPO_ROOT}/.github/workflows/_pytorch-2.12-hcu-test.yml"

grep -Fq 'export PYTORCH_NUM_PYTEST_RERUNS=2' "${TEST_RUNNER}"
grep -Fq 'export PYTORCH_TEST_FAILURE_ATTEMPTS=1' "${TEST_RUNNER}"
grep -Fq 'export PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS=1800' "${TEST_RUNNER}"
grep -Fq 'PR_TEST_TIMEOUT_SECONDS="${PR_TEST_TIMEOUT_SECONDS:-13500}"' \
    "${TEST_RUNNER}"
grep -Fq 'timeout --signal=TERM --kill-after=60s' "${TEST_RUNNER}"
grep -Fq 'PR shard timeout diagnostics' "${TEST_RUNNER}"
grep -Fq 'PYTORCH_TEST_FALLBACK_TIMEOUT_SECONDS' \
    "${REPO_ROOT}/test/run_test.py"
grep -Fq 'PYTORCH_TEST_FAILURE_ATTEMPTS' "${REPO_ROOT}/test/run_test.py"
grep -Fq "inputs.test_suite == 'pr' && 270 || 4320" "${WORKFLOW}"

echo "PR retry and hard-timeout policy passed"
