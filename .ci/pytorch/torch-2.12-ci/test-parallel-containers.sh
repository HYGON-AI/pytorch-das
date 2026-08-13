#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/../../.." && pwd)"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_ROOT}"' EXIT

grep -Fq \
    'test_matrix='\''{"include":[{"config":"pr","shard":0,"num_shards":2}]}'\''' \
    "${REPO_ROOT}/.github/workflows/pytorch-2.12-build.yml"
grep -Fq \
    'PR_DCU_DEVICE_PAIRS: ${{ inputs.pr_dcu_device_pairs }}' \
    "${REPO_ROOT}/.github/workflows/_pytorch-2.12-dcu-test.yml"
grep -Fq \
    'test_matrix='\''{"include":[{"config":"default","shard":0,"num_shards":4}]}'\''' \
    "${REPO_ROOT}/.github/workflows/pytorch-2.12-build.yml"
grep -Fq \
    'NIGHTLY_DCU_DEVICE_PAIRS: ${{ inputs.nightly_dcu_device_pairs }}' \
    "${REPO_ROOT}/.github/workflows/_pytorch-2.12-dcu-test.yml"
grep -Fq \
    'Fail PR after reporting all test failures' \
    "${REPO_ROOT}/.github/workflows/pytorch-2.12-build.yml"
grep -Fq \
    "steps.test_summary.outputs.failed_test_shards != '0'" \
    "${REPO_ROOT}/.github/workflows/pytorch-2.12-build.yml"
grep -Fq \
    'PR tests failed after all shards completed' \
    "${REPO_ROOT}/.github/workflows/pytorch-2.12-build.yml"

MOCK_BIN="${TEMP_ROOT}/bin"
MOCK_LOG="${TEMP_ROOT}/docker.log"
MOCK_STATE_DIR="${TEMP_ROOT}/state"
REPORT_ROOT="${TEMP_ROOT}/reports"
mkdir -p "${MOCK_BIN}" "${MOCK_STATE_DIR}" "${REPORT_ROOT}"

cat > "${MOCK_BIN}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
set -euo pipefail

printf '%q ' "$@" >> "${MOCK_DOCKER_LOG}"
printf '\n' >> "${MOCK_DOCKER_LOG}"

command_name="${1:-}"
shift || true
case "${command_name}" in
    run)
        echo "mock-container-id"
        ;;
    exec)
        container_name="${1:?container name is required}"
        shift
        case "${1:-}" in
            bash)
                touch "${MOCK_STATE_DIR}/${container_name}"
                peer_count=0
                for _ in $(seq 1 50); do
                    peer_count="$(find "${MOCK_STATE_DIR}" -type f | wc -l)"
                    if (( peer_count == MOCK_EXPECTED_CONTAINERS )); then
                        break
                    fi
                    sleep 0.1
                done
                if (( peer_count != MOCK_EXPECTED_CONTAINERS )); then
                    echo "parallel peer container did not start" >&2
                    exit 3
                fi
                if [[ -n "${MOCK_FAIL_CONTAINER:-}" &&
                    "${container_name}" == *"${MOCK_FAIL_CONTAINER}"* ]]; then
                    exit 7
                fi
                ;;
            test)
                ;;
            tar)
                tar -cf - --files-from /dev/null
                ;;
            *)
                echo "unexpected docker exec for ${container_name}: $*" >&2
                exit 2
                ;;
        esac
        ;;
    logs | rm)
        ;;
    *)
        echo "unexpected docker command: ${command_name} $*" >&2
        exit 2
        ;;
esac
MOCK_DOCKER
chmod +x "${MOCK_BIN}/docker"

PATH="${MOCK_BIN}:${PATH}" \
MOCK_DOCKER_LOG="${MOCK_LOG}" \
MOCK_STATE_DIR="${MOCK_STATE_DIR}" \
MOCK_EXPECTED_CONTAINERS=2 \
DOCKER_IMAGE_NAME="pytorch-test:fixture" \
TEST_SUITE="pr" \
TEST_CONFIG="pr" \
SHARD_NUMBER="0" \
NUM_TEST_SHARDS="2" \
DCU_VISIBLE_DEVICES="0,1" \
PR_DCU_DEVICE_PAIRS="0,1;2,3" \
REPORT_ROOT="${REPORT_ROOT}" \
GITHUB_RUN_ID="fixture" \
GITHUB_RUN_ATTEMPT="1" \
bash "${CI_DIR}/run-nightly-tests.sh"

if [[ "$(grep -Fo -- '--env DCU_VISIBLE_DEVICES=0\,1' "${MOCK_LOG}" | wc -l)" -lt 2 ]]; then
    echo "ERROR: both containers must expose their physical pair as logical 0,1" >&2
    exit 1
fi
grep -Fq -- '--env ROCR_VISIBLE_DEVICES=0\,1' "${MOCK_LOG}"
grep -Fq -- '--env ROCR_VISIBLE_DEVICES=2\,3' "${MOCK_LOG}"
grep -Fq 'pytorch-2.12-dcu-test-fixture-1-pr-1' "${MOCK_LOG}"
grep -Fq 'pytorch-2.12-dcu-test-fixture-1-pr-2' "${MOCK_LOG}"

for shard in 1 2; do
    status_file="${REPORT_ROOT}/pr-${shard}-of-2/status.json"
    test -f "${status_file}"
    grep -Fq '"conclusion":"passed"' "${status_file}"
    grep -Fq '"failure_kind":"none"' "${status_file}"
done

echo "PR parallel-container start barrier passed"

rm -rf -- "${MOCK_STATE_DIR}" "${REPORT_ROOT}"
rm -f -- "${MOCK_LOG}"
mkdir -p "${MOCK_STATE_DIR}" "${REPORT_ROOT}"

PATH="${MOCK_BIN}:${PATH}" \
MOCK_DOCKER_LOG="${MOCK_LOG}" \
MOCK_STATE_DIR="${MOCK_STATE_DIR}" \
MOCK_EXPECTED_CONTAINERS=2 \
MOCK_FAIL_CONTAINER="pr-2" \
DOCKER_IMAGE_NAME="pytorch-test:fixture" \
TEST_SUITE="pr" \
TEST_CONFIG="pr" \
SHARD_NUMBER="0" \
NUM_TEST_SHARDS="2" \
DCU_VISIBLE_DEVICES="0,1" \
PR_DCU_DEVICE_PAIRS="0,1;2,3" \
REPORT_ROOT="${REPORT_ROOT}" \
GITHUB_RUN_ID="fixture" \
GITHUB_RUN_ATTEMPT="2" \
bash "${CI_DIR}/run-nightly-tests.sh"

grep -Fq '"failure_kind":"none"' \
    "${REPORT_ROOT}/pr-1-of-2/status.json"
grep -Fq '"exit_code":7' "${REPORT_ROOT}/pr-2-of-2/status.json"
grep -Fq '"failure_kind":"test"' \
    "${REPORT_ROOT}/pr-2-of-2/status.json"
grep -Fq 'test_failure_policy=fail-after-summary' \
    "${REPORT_ROOT}/nightly-summary.txt"
grep -Fq 'failed_test_shards=1' "${REPORT_ROOT}/nightly-summary.txt"

echo "PR collect-all-before-final-failure policy passed"

rm -rf -- "${MOCK_STATE_DIR}" "${REPORT_ROOT}"
rm -f -- "${MOCK_LOG}"
mkdir -p "${MOCK_STATE_DIR}" "${REPORT_ROOT}"

PATH="${MOCK_BIN}:${PATH}" \
MOCK_DOCKER_LOG="${MOCK_LOG}" \
MOCK_STATE_DIR="${MOCK_STATE_DIR}" \
MOCK_EXPECTED_CONTAINERS=4 \
DOCKER_IMAGE_NAME="pytorch-test:fixture" \
TEST_SUITE="nightly" \
TEST_CONFIG="default" \
SHARD_NUMBER="0" \
NUM_TEST_SHARDS="4" \
DCU_VISIBLE_DEVICES="0,1" \
NIGHTLY_DCU_DEVICE_PAIRS="0,1;2,3;4,5;6,7" \
REPORT_ROOT="${REPORT_ROOT}" \
GITHUB_RUN_ID="fixture" \
GITHUB_RUN_ATTEMPT="3" \
bash "${CI_DIR}/run-nightly-tests.sh"

for device_pair in '0\,1' '2\,3' '4\,5' '6\,7'; do
    grep -Fq -- "--env ROCR_VISIBLE_DEVICES=${device_pair}" "${MOCK_LOG}"
done
for shard in 1 2 3 4; do
    status_file="${REPORT_ROOT}/default-${shard}-of-4/status.json"
    test -f "${status_file}"
    grep -Fq '"conclusion":"passed"' "${status_file}"
    grep -Fq "pytorch-2.12-dcu-test-fixture-3-default-${shard}" "${MOCK_LOG}"
done
grep -Fq 'total_shards=4' "${REPORT_ROOT}/nightly-summary.txt"
grep -Fq 'passed_shards=4' "${REPORT_ROOT}/nightly-summary.txt"

echo "Nightly four-container start barrier passed"

rm -rf -- "${MOCK_STATE_DIR}" "${REPORT_ROOT}"
rm -f -- "${MOCK_LOG}"
mkdir -p "${MOCK_STATE_DIR}" "${REPORT_ROOT}"

PATH="${MOCK_BIN}:${PATH}" \
MOCK_DOCKER_LOG="${MOCK_LOG}" \
MOCK_STATE_DIR="${MOCK_STATE_DIR}" \
MOCK_EXPECTED_CONTAINERS=4 \
MOCK_FAIL_CONTAINER="default-3" \
DOCKER_IMAGE_NAME="pytorch-test:fixture" \
TEST_SUITE="nightly" \
TEST_CONFIG="default" \
SHARD_NUMBER="0" \
NUM_TEST_SHARDS="4" \
DCU_VISIBLE_DEVICES="0,1" \
NIGHTLY_DCU_DEVICE_PAIRS="0,1;2,3;4,5;6,7" \
REPORT_ROOT="${REPORT_ROOT}" \
GITHUB_RUN_ID="fixture" \
GITHUB_RUN_ATTEMPT="4" \
bash "${CI_DIR}/run-nightly-tests.sh"

for shard in 1 2 3 4; do
    test -f "${REPORT_ROOT}/default-${shard}-of-4/status.json"
done
grep -Fq '"failure_kind":"test"' \
    "${REPORT_ROOT}/default-3-of-4/status.json"
grep -Fq 'failed_test_shards=1' "${REPORT_ROOT}/nightly-summary.txt"
grep -Fq 'passed_shards=3' "${REPORT_ROOT}/nightly-summary.txt"

echo "Nightly report-only failure policy passed"
