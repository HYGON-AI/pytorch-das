#!/usr/bin/env bash

set -Eeuo pipefail

DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:?DOCKER_IMAGE_NAME is required}"
TEST_SUITE="${TEST_SUITE:-smoke}"
REPORT_ROOT="${REPORT_ROOT:-${RUNNER_TEMP:-/tmp}/pytorch-2.12-dcu-reports}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-linux-rocky8.6-rocm-dcu-py3.10}"

case "${TEST_SUITE}" in
    smoke)
        matrix=("smoke:1:1")
        ;;
    nightly)
        # Keep this matrix aligned with
        # .github/workflows/rocm-nightly.yml on release/2.12.
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

mkdir -p "${REPORT_ROOT}"
failures=()

run_entry() {
    local config="$1"
    local shard="$2"
    local num_shards="$3"
    local container_name="pytorch-2.12-dcu-test-${GITHUB_RUN_ID:-local}-${config}-${shard}"
    local entry_dir="${REPORT_ROOT}/${config}-${shard}-of-${num_shards}"
    local hip_visible_devices="0"
    local test_started=0
    local conclusion="passed"

    mkdir -p "${entry_dir}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    # shellcheck disable=SC2086
    docker run --detach \
        --name "${container_name}" \
        --label "org.pytorch.dcu.nightly=${GITHUB_RUN_ID:-local}" \
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
    rc=${PIPESTATUS[0]}
    set -e

    if docker exec "${container_name}" \
        test -f /tmp/pytorch-nightly-test-started; then
        test_started=1
    fi

    if docker exec "${container_name}" \
        test -d /pytorch/test/test-reports; then
        mkdir -p "${entry_dir}/test-reports"
        # Extract as the runner user. docker cp would create root-owned files
        # and make the next actions/checkout cleanup fail with EACCES.
        docker exec "${container_name}" \
            tar -C /pytorch/test/test-reports -cf - . |
            tar --no-same-owner -C "${entry_dir}/test-reports" -xf - || true
    fi
    docker logs "${container_name}" > "${entry_dir}/container.log" 2>&1 || true
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    if (( rc != 0 )); then
        conclusion="failed"
        failures+=("${config}:${shard}/${num_shards}:rc=${rc}")
    fi
    printf \
        '{"config":"%s","shard":%s,"num_shards":%s,"exit_code":%s,"conclusion":"%s"}\n' \
        "${config}" "${shard}" "${num_shards}" "${rc}" "${conclusion}" \
        > "${entry_dir}/status.json"

    if (( rc != 0 && test_started == 0 )); then
        echo "ERROR: ${config} shard ${shard}/${num_shards} failed before tests started" >&2
        return "${rc}"
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
summary_file="${REPORT_ROOT}/nightly-summary.txt"

{
    echo "test_failure_policy=report-only"
    echo "total_shards=${total_shards}"
    echo "passed_shards=${passed_shards}"
    echo "failed_shards=${failed_shards}"
    if (( failed_shards > 0 )); then
        echo "failed_entries:"
        printf '  %s\n' "${failures[@]}"
    fi
} | tee "${summary_file}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "## DCU nightly test summary"
        echo
        echo "| Policy | Total shards | Passed | Failed |"
        echo "| --- | ---: | ---: | ---: |"
        echo "| Report only | ${total_shards} | ${passed_shards} | ${failed_shards} |"
        if (( failed_shards > 0 )); then
            echo
            echo "Failed entries:"
            printf -- '- `%s`\n' "${failures[@]}"
        fi
    } >> "${GITHUB_STEP_SUMMARY}"
fi

if (( failed_shards > 0 )); then
    for failure in "${failures[@]}"; do
        echo "::warning title=DCU nightly test failure::${failure}"
    done
    echo "Test failures are report-only; all shards completed and the workflow remains green."
fi

exit 0
