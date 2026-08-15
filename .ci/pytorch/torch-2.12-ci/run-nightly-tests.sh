#!/usr/bin/env bash

set -Eeuo pipefail

DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:?DOCKER_IMAGE_NAME is required}"
TEST_SUITE="${TEST_SUITE:-smoke}"
REPORT_ROOT="${REPORT_ROOT:-${RUNNER_TEMP:-/tmp}/pytorch-2.12-dcu-reports}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-linux-rocky8.6-rocm-dcu-py3.10}"
DCU_VISIBLE_DEVICES="${DCU_VISIBLE_DEVICES:-0,1}"
PR_DCU_DEVICE_PAIRS="${PR_DCU_DEVICE_PAIRS:-4,5;6,7}"
NIGHTLY_DCU_DEVICE_PAIRS="${NIGHTLY_DCU_DEVICE_PAIRS:-0,1;2,3;4,5;6,7}"

validate_device_pair() {
    local device_pair="$1"
    if ! [[ "${device_pair}" =~ ^([0-9]+),([0-9]+)$ ]]; then
        echo "ERROR: device pair must contain exactly two indexes: ${device_pair}" >&2
        return 1
    fi
    if [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
        echo "ERROR: device pair must contain distinct indexes: ${device_pair}" >&2
        return 1
    fi
}

validate_device_pair "${DCU_VISIBLE_DEVICES}"

single_entry=0
provided_entry_values=0
for value in "${TEST_CONFIG:-}" "${SHARD_NUMBER:-}" "${NUM_TEST_SHARDS:-}"; do
    if [[ -n "${value}" ]]; then
        provided_entry_values=$((provided_entry_values + 1))
    fi
done

pr_coordinator=0
nightly_coordinator=0
if [[ "${TEST_SUITE}" == "pr" && "${TEST_CONFIG:-}" == "pr" &&
    "${SHARD_NUMBER:-}" == "0" && "${NUM_TEST_SHARDS:-}" == "2" ]]; then
    matrix=("pr:1:2" "pr:2:2")
    pr_coordinator=1
elif [[ "${TEST_SUITE}" == "nightly" && "${TEST_CONFIG:-}" == "default" &&
    "${SHARD_NUMBER:-}" == "0" && "${NUM_TEST_SHARDS:-}" == "4" ]]; then
    matrix=("default:1:4" "default:2:4" "default:3:4" "default:4:4")
    nightly_coordinator=1
elif (( provided_entry_values > 0 )); then
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
        pr)
            matrix=("pr:1:2" "pr:2:2")
            ;;
        nightly)
            # Run the official default test config with four-way sharding.
            matrix=(
                "default:1:4" "default:2:4"
                "default:3:4" "default:4:4"
            )
            ;;
        *)
            echo "ERROR: unsupported TEST_SUITE=${TEST_SUITE}" >&2
            exit 1
            ;;
    esac
fi

mkdir -p "${REPORT_ROOT}"

run_entry() {
    local config="$1"
    local shard="$2"
    local num_shards="$3"
    local physical_device_pair="$4"
    # ROCR selects physical devices. HIP/CUDA then see those two devices as 0,1.
    local logical_device_pair="0,1"
    local run_scope="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
    local container_name="pytorch-2.12-dcu-test-${run_scope}-${config}-${shard}"
    local entry_dir="${REPORT_ROOT}/${config}-${shard}-of-${num_shards}"
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
        --env "DCU_VISIBLE_DEVICES=${logical_device_pair}" \
        --env "CUDA_VISIBLE_DEVICES=${logical_device_pair}" \
        --env "HIP_VISIBLE_DEVICES=${logical_device_pair}" \
        --env "ROCR_VISIBLE_DEVICES=${physical_device_pair}" \
        --env "CI=1" \
        --env "PYTHONUNBUFFERED=1" \
        --env "PYTORCH_TEST_WITH_ROCM=1" \
        --env "PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda" \
        --env "PYTORCH_TEST_VISIBLE_DEVICE_COUNT=2" \
        --env "TORCH_SERIALIZATION_DEBUG=1" \
        --env "FA_SO_PATH=/opt/flash_attn_2_cuda.so" \
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
            echo "ERROR: DCU test shard did not create /pytorch/test/test-reports" >&2
        else
            echo "ERROR: test report extraction failed with rc=${report_rc}" >&2
        fi
        return "${effective_rc}"
    fi
}

infrastructure_failed=0
if (( pr_coordinator == 1 || nightly_coordinator == 1 )); then
    if (( pr_coordinator == 1 )); then
        coordinator_name="PR"
        coordinator_config="pr"
        coordinator_shards=2
        coordinator_pairs="${PR_DCU_DEVICE_PAIRS}"
    else
        coordinator_name="nightly"
        coordinator_config="default"
        coordinator_shards=4
        coordinator_pairs="${NIGHTLY_DCU_DEVICE_PAIRS}"
    fi

    IFS=';' read -r -a device_pairs <<< "${coordinator_pairs}"
    if (( ${#device_pairs[@]} != coordinator_shards )); then
        echo "ERROR: ${coordinator_name} requires ${coordinator_shards} device pairs: ${coordinator_pairs}" >&2
        exit 1
    fi

    devices_seen=()
    for device_pair in "${device_pairs[@]}"; do
        validate_device_pair "${device_pair}"
        IFS=',' read -r device_a device_b <<< "${device_pair}"
        for device in "${device_a}" "${device_b}"; do
            for seen in "${devices_seen[@]:-}"; do
                if [[ -n "${seen}" && "${device}" == "${seen}" ]]; then
                    echo "ERROR: ${coordinator_name} device pairs overlap: ${coordinator_pairs}" >&2
                    exit 1
                fi
            done
            devices_seen+=("${device}")
        done
    done

    echo "===== Running ${coordinator_name} shards in parallel ====="
    coordinator_pids=()
    for (( shard=1; shard<=coordinator_shards; shard++ )); do
        device_pair="${device_pairs[$((shard - 1))]}"
        echo "${coordinator_config}:${shard}/${coordinator_shards} devices=${device_pair}"
        run_entry \
            "${coordinator_config}" "${shard}" "${coordinator_shards}" "${device_pair}" &
        coordinator_pids+=("$!")
    done

    set +e
    for coordinator_pid in "${coordinator_pids[@]}"; do
        wait "${coordinator_pid}"
        if (( $? != 0 )); then
            infrastructure_failed=1
        fi
    done
    set -e
else
    for entry in "${matrix[@]}"; do
        IFS=: read -r config shard num_shards <<< "${entry}"
        echo "===== Running ${config} shard ${shard}/${num_shards} ====="
        run_entry \
            "${config}" "${shard}" "${num_shards}" "${DCU_VISIBLE_DEVICES}"
    done
fi

failures=()
for entry in "${matrix[@]}"; do
    IFS=: read -r config shard num_shards <<< "${entry}"
    status_file="${REPORT_ROOT}/${config}-${shard}-of-${num_shards}/status.json"
    if [[ ! -f "${status_file}" ]]; then
        infrastructure_failed=1
        continue
    fi
    status_json="$(< "${status_file}")"
    if [[ "${status_json}" == *'"failure_kind":"test"'* ]]; then
        if [[ "${status_json}" =~ \"exit_code\":(-?[0-9]+) ]]; then
            failures+=("${config}:${shard}/${num_shards}:rc=${BASH_REMATCH[1]}")
        else
            failures+=("${config}:${shard}/${num_shards}:rc=unknown")
        fi
    elif [[ "${status_json}" != *'"failure_kind":"none"'* ]]; then
        infrastructure_failed=1
    fi
done

total_shards="${#matrix[@]}"
failed_shards="${#failures[@]}"
passed_shards="$((total_shards - failed_shards))"
test_failure_policy="report-only"
if [[ "${TEST_SUITE}" == "pr" ]]; then
    test_failure_policy="fail-after-summary"
fi
if (( single_entry == 1 )); then
    summary_file="${REPORT_ROOT}/${TEST_CONFIG}-${SHARD_NUMBER}-of-${NUM_TEST_SHARDS}/shard-summary.txt"
else
    summary_file="${REPORT_ROOT}/nightly-summary.txt"
fi

{
    echo "test_failure_policy=${test_failure_policy}"
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
        echo "| ${test_failure_policy} | ${total_shards} | ${passed_shards} | ${failed_shards} |"
        if (( failed_shards > 0 )); then
            echo
            echo "Failed test entries:"
            printf -- '- `%s`\n' "${failures[@]}"
        fi
    } >> "${GITHUB_STEP_SUMMARY}"
fi

if (( failed_shards > 0 )); then
    for failure in "${failures[@]}"; do
        echo "::warning title=DCU test failure::${failure}"
    done
    if [[ "${TEST_SUITE}" == "pr" ]]; then
        echo "PR test failures were recorded; the required job will fail after reporting and cleanup."
    else
        echo "Test failures are report-only; this shard job remains green."
    fi
fi

if (( infrastructure_failed != 0 )); then
    echo "ERROR: one or more test containers failed at the infrastructure level" >&2
    exit 1
fi

exit 0
