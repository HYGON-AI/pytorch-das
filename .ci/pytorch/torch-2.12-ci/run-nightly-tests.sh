#!/usr/bin/env bash

set -Eeuo pipefail

DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:?DOCKER_IMAGE_NAME is required}"
TEST_SUITE="${TEST_SUITE:-smoke}"
REPORT_ROOT="${REPORT_ROOT:-${RUNNER_TEMP:-/tmp}/pytorch-2.12-dcu-reports}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-linux-rocky8.6-rocm-dcu-py3.10}"

single_entry=0
provided_entry_values=0
for value in "${TEST_CONFIG:-}" "${SHARD_NUMBER:-}" "${NUM_TEST_SHARDS:-}"; do
    if [[ -n "${value}" ]]; then
        provided_entry_values=$((provided_entry_values + 1))
    fi
done

if (( provided_entry_values > 0 )); then
    if (( provided_entry_values != 3 )); then
        echo "ERROR: TEST_CONFIG, SHARD_NUMBER and NUM_TEST_SHARDS must be set together" >&2
        exit 1
    fi
    if ! [[ "${TEST_CONFIG}" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
        echo "ERROR: invalid TEST_CONFIG=${TEST_CONFIG}" >&2
        exit 1
    fi
    if ! [[ "${SHARD_NUMBER}" =~ ^[1-9][0-9]*$ && "${NUM_TEST_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: shard values must be positive integers" >&2
        exit 1
    fi
    if (( SHARD_NUMBER > NUM_TEST_SHARDS )); then
        echo "ERROR: shard ${SHARD_NUMBER} exceeds total ${NUM_TEST_SHARDS}" >&2
        exit 1
    fi
    matrix=("${TEST_CONFIG}:${SHARD_NUMBER}:${NUM_TEST_SHARDS}")
    single_entry=1
else
    case "${TEST_SUITE}" in
        smoke)
            matrix=("smoke:1:1")
            ;;
        nightly)
            # Keep this matrix aligned with release/2.12 rocm-nightly.yml.
            matrix=(
                "default:1:6" "default:2:6" "default:3:6"
                "default:4:6" "default:5:6" "default:6:6"
            )
            ;;
        *)
            echo "ERROR: unsupported TEST_SUITE=${TEST_SUITE}" >&2
            exit 1
            ;;
    esac
fi

mkdir -p "${REPORT_ROOT}"
failures=()

run_entry() {
    local config="$1"
    local shard="$2"
    local num_shards="$3"
    local run_scope="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
    local container_name="pytorch-2.12-dcu-test-${run_scope}-${config}-${shard}"
    local entry_dir="${REPORT_ROOT}/${config}-${shard}-of-${num_shards}"
    local hip_visible_devices="0"
    local test_started=0
    local test_completed=0
    local console_rc=0
    local report_rc=0
    local report_missing=0
    local rc=0
    local effective_rc=0
    local conclusion="passed"
    local failure_kind="none"
    local status_tmp
    local -a pipeline_status=()

    mkdir -p "${entry_dir}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    # shellcheck disable=SC2086
    docker run --detach \
        --name "${container_name}" \
        --label "org.pytorch.dcu.workflow-run=${GITHUB_RUN_ID:-local}" \
        --label "org.pytorch.dcu.workflow-attempt=${GITHUB_RUN_ATTEMPT:-1}" \
        --label "org.pytorch.dcu.nightly=${run_scope}" \
        --label "${HCU_CONTAINER_LABEL:-org.pytorch.hcu.run=local}" \
        ${HCU_DOCKER_GPU_ARGS:-} \
        --network host \
        --ipc host \
        --init \
        --shm-size=64g \
        --ulimit stack=10485760:83886080 \
        --ulimit core=0 \
        --env "BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" \
        --env "TEST_CONFIG=${config}" \
        --env "SHARD_NUMBER=${shard}" \
        --env "NUM_TEST_SHARDS=${num_shards}" \
        --env "HIP_VISIBLE_DEVICES=${hip_visible_devices}" \
        --env "CI=1" \
        --env "PYTHONUNBUFFERED=1" \
        --env "PYTORCH_TEST_WITH_ROCM=1" \
        --env "PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda" \
        --env "CONTINUE_THROUGH_ERROR=1" \
        --env "NO_TD=1" \
        --env "MAX_JOBS=${MAX_JOBS:-4}" \
        "${DOCKER_IMAGE_NAME}" \
        tail -f /dev/null

    set +e
    docker exec "${container_name}" bash \
        /pytorch/.ci/pytorch/torch-2.12-ci/run-nightly-test-in-container.sh \
        2>&1 | tee "${entry_dir}/console.log"
    pipeline_status=("${PIPESTATUS[@]}")
    rc="${pipeline_status[0]}"
    console_rc="${pipeline_status[1]}"
    set -e

    if docker exec "${container_name}" test -f /tmp/pytorch-nightly-test-started; then
        test_started=1
    fi
    if docker exec "${container_name}" test -f /tmp/pytorch-nightly-test-completed; then
        test_completed=1
    fi

    if docker exec "${container_name}" test -d /pytorch/test/test-reports; then
        mkdir -p "${entry_dir}/test-reports"
        # Extract as the runner user. docker cp would create root-owned files
        # and make the next actions/checkout cleanup fail with EACCES.
        set +e
        docker exec "${container_name}" \
            tar -C /pytorch/test/test-reports -cf - . |
            tar --no-same-owner -C "${entry_dir}/test-reports" -xf -
        report_rc=$?
        set -e
    elif [[ "${config}" != "smoke" ]]; then
        report_missing=1
    fi

    docker logs "${container_name}" > "${entry_dir}/container.log" 2>&1 || true
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    effective_rc="${rc}"
    if (( console_rc != 0 )); then
        effective_rc="${console_rc}"
        conclusion="failed"
        failure_kind="infrastructure"
    elif (( test_started == 0 || test_completed == 0 )); then
        if (( effective_rc == 0 )); then
            effective_rc=1
        fi
        conclusion="failed"
        failure_kind="infrastructure"
    elif (( report_missing != 0 )); then
        effective_rc=1
        conclusion="failed"
        failure_kind="infrastructure"
    elif (( report_rc != 0 )); then
        effective_rc="${report_rc}"
        conclusion="failed"
        failure_kind="infrastructure"
    elif (( rc != 0 )); then
        conclusion="failed"
        failure_kind="test"
        failures+=("${config}:${shard}/${num_shards}:rc=${rc}")
    fi

    status_tmp="${entry_dir}/status.json.tmp.$$"
    printf \
        '{"config":"%s","shard":%s,"num_shards":%s,"exit_code":%s,"conclusion":"%s","failure_kind":"%s","tests_started":%s,"tests_completed":%s}\n' \
        "${config}" "${shard}" "${num_shards}" "${effective_rc}" \
        "${conclusion}" "${failure_kind}" "${test_started}" "${test_completed}" \
        > "${status_tmp}"
    mv -f "${status_tmp}" "${entry_dir}/status.json"

    if [[ "${failure_kind}" == "infrastructure" ]]; then
        echo "ERROR: ${config} shard ${shard}/${num_shards} did not complete cleanly" >&2
        if (( console_rc != 0 )); then
            echo "ERROR: console log collection failed with rc=${console_rc}" >&2
        elif (( test_started == 0 )); then
            echo "ERROR: failure occurred before tests started" >&2
        elif (( test_completed == 0 )); then
            echo "ERROR: test process ended without a completion marker" >&2
        elif (( report_missing != 0 )); then
            echo "ERROR: nightly shard did not create /pytorch/test/test-reports" >&2
        else
            echo "ERROR: test report extraction failed with rc=${report_rc}" >&2
        fi
        return "${effective_rc}"
    fi
}

for entry in "${matrix[@]}"; do
    IFS=: read -r config shard num_shards <<< "${entry}"
    echo "===== Running ${config} shard ${shard}/${num_shards} ====="
    run_entry "${config}" "${shard}" "${num_shards}"
done

total_shards="${#matrix[@]}"
failed_shards="${#failures[@]}"
passed_shards="$((total_shards - failed_shards))"
if (( single_entry == 1 )); then
    summary_file="${REPORT_ROOT}/${TEST_CONFIG}-${SHARD_NUMBER}-of-${NUM_TEST_SHARDS}/shard-summary.txt"
else
    summary_file="${REPORT_ROOT}/nightly-summary.txt"
fi

{
    echo "test_failure_policy=report-only"
    echo "total_shards=${total_shards}"
    echo "passed_shards=${passed_shards}"
    echo "failed_test_shards=${failed_shards}"
    if (( failed_shards > 0 )); then
        echo "failed_test_entries:"
        printf '  %s\n' "${failures[@]}"
    fi
} | tee "${summary_file}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "## DCU test shard summary"
        echo
        echo "| Policy | Total shards | Passed | Test failures |"
        echo "| --- | ---: | ---: | ---: |"
        echo "| Report only | ${total_shards} | ${passed_shards} | ${failed_shards} |"
        if (( failed_shards > 0 )); then
            echo
            echo "Failed test entries:"
            printf -- '- `%s`\n' "${failures[@]}"
        fi
    } >> "${GITHUB_STEP_SUMMARY}"
fi

if (( failed_shards > 0 )); then
    for failure in "${failures[@]}"; do
        echo "::warning title=DCU nightly test failure::${failure}"
    done
    echo "Test failures are report-only; this shard job remains green."
fi

exit 0
