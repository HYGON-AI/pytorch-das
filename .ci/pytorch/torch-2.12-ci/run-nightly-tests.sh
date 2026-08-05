#!/usr/bin/env bash

set -Eeuo pipefail

DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:?DOCKER_IMAGE_NAME is required}"
ARTIFACT_ZIP="${ARTIFACT_ZIP:?ARTIFACT_ZIP is required}"
TEST_SUITE="${TEST_SUITE:-smoke}"
REPORT_ROOT="${REPORT_ROOT:-${RUNNER_TEMP:-/tmp}/pytorch-2.12-dcu-reports}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-linux-rocky8.6-rocm-dcu-py3.10}"

[[ -f "${ARTIFACT_ZIP}" ]] || {
    echo "ERROR: artifact not found: ${ARTIFACT_ZIP}" >&2
    exit 1
}

case "${TEST_SUITE}" in
    smoke)
        matrix=("smoke:1:1")
        ;;
    nightly)
        matrix=(
            "default:1:6" "default:2:6" "default:3:6"
            "default:4:6" "default:5:6" "default:6:6"
        )
        ;;
    full)
        matrix=(
            "default:1:6" "default:2:6" "default:3:6"
            "default:4:6" "default:5:6" "default:6:6"
            "distributed:1:3" "distributed:2:3" "distributed:3:3"
            "inductor:1:2" "inductor:2:2"
            "slow:1:2" "slow:2:2"
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

    if [[ "${config}" == "distributed" ]]; then
        hip_visible_devices="0,1,2,3"
    fi

    mkdir -p "${entry_dir}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    # shellcheck disable=SC2086
    docker run --detach \
        --name "${container_name}" \
        --label "org.pytorch.dcu.nightly=${GITHUB_RUN_ID:-local}" \
        ${HCU_DOCKER_GPU_ARGS:-} \
        --network host \
        --ipc host \
        --shm-size=64g \
        --env "BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" \
        --env "TEST_CONFIG=${config}" \
        --env "SHARD_NUMBER=${shard}" \
        --env "NUM_TEST_SHARDS=${num_shards}" \
        --env "HIP_VISIBLE_DEVICES=${hip_visible_devices}" \
        --env "PYTORCH_TEST_WITH_ROCM=1" \
        --env "PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda" \
        --env "CONTINUE_THROUGH_ERROR=1" \
        --env "NO_TD=1" \
        --env "MAX_JOBS=${MAX_JOBS:-4}" \
        "${DOCKER_IMAGE_NAME}" \
        tail -f /dev/null

    docker cp "${ARTIFACT_ZIP}" "${container_name}:/tmp/artifacts.zip"
    docker exec "${container_name}" bash -lc \
        'cd /pytorch && unzip -o /tmp/artifacts.zip && cd / && python3 -m pip install --no-deps --force-reinstall /pytorch/dist/*.whl'

    set +e
    if [[ "${config}" == "smoke" ]]; then
        docker exec "${container_name}" bash -lc \
            'cd / && python3 -c '\''import torch; assert torch.cuda.is_available(); assert torch.version.hip; x = torch.randn(32, 32, device="cuda"); y = x @ x; torch.cuda.synchronize(); print(torch.__version__); print(torch.version.hip); print(torch.cuda.get_device_name(0)); print(y.norm().item())'\'''
        rc=$?
    else
        docker exec "${container_name}" bash -lc \
            'cd /pytorch && .ci/pytorch/test.sh' \
            2>&1 | tee "${entry_dir}/console.log"
        rc=${PIPESTATUS[0]}
    fi
    set -e

    docker cp \
        "${container_name}:/pytorch/test/test-reports/." \
        "${entry_dir}/test-reports" >/dev/null 2>&1 || true
    docker logs "${container_name}" > "${entry_dir}/container.log" 2>&1 || true
    docker rm -f "${container_name}" >/dev/null 2>&1 || true

    if (( rc != 0 )); then
        failures+=("${config}:${shard}/${num_shards}")
    fi
}

for entry in "${matrix[@]}"; do
    IFS=: read -r config shard num_shards <<< "${entry}"
    echo "===== Running ${config} shard ${shard}/${num_shards} ====="
    run_entry "${config}" "${shard}" "${num_shards}"
done

if (( ${#failures[@]} > 0 )); then
    printf 'Failed nightly entries:\n'
    printf '  %s\n' "${failures[@]}"
    exit 1
fi

