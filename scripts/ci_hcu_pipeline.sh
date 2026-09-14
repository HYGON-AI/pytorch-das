#!/usr/bin/env bash

set -Eeuo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_env() {
    local name
    for name in "$@"; do
        [[ -n "${!name:-}" ]] || die "required environment variable is not set: ${name}"
    done
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

start_container() {
    local container="$1" image="$2" expected_python="$3"
    local actual_python

    docker rm -f "${container}" >/dev/null 2>&1 || true
    docker pull "${image}"
    docker run --detach \
        --name "${container}" \
        --label "org.pytorch.hcu.run=${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
        --network host --ipc host --init --shm-size=64g \
        --ulimit stack=10485760:83886080 --ulimit core=0 \
        --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
        --volume /opt/hyhal:/opt/hyhal:ro \
        "${image}" tail -f /dev/null

    actual_python="$(docker exec "${container}" python3 -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${actual_python}" == "${expected_python}" ]] || \
        die "${image} provides Python ${actual_python}; expected ${expected_python}"

    docker exec --env DTK_PKG_URL "${container}" bash -lc '
        set -euo pipefail
        if [[ -f /opt/dtk/env.sh && -f /opt/dtk/.info/rocm_version ]] &&
           [[ "$(tr -d "[:space:]" < /opt/dtk/.info/rocm_version)" == 26.04* ]]; then
            exit 0
        fi
        dtk_stage="$(mktemp -d /tmp/pytorch-dtk.XXXXXX)"
        trap '\''rm -rf -- "${dtk_stage}"'\'' EXIT
        mkdir -p "${dtk_stage}/extract"
        wget --quiet --show-progress --tries=3 --timeout=120 \
            --output-document="${dtk_stage}/dtk.tar.gz" -- "${DTK_PKG_URL}"
        tar -xzf "${dtk_stage}/dtk.tar.gz" -C "${dtk_stage}/extract"
        version_file="$(find "${dtk_stage}/extract" -type f -path "*/.info/rocm_version" -print -quit)"
        [[ -n "${version_file}" ]]
        dtk_source="${version_file%/.info/rocm_version}"
        [[ "$(tr -d "[:space:]" < "${version_file}")" == 26.04* ]]
        rm -rf -- /opt/dtk
        mkdir -p /opt/dtk
        cp -a "${dtk_source}/." /opt/dtk/
    '

    docker exec "${container}" mkdir -p /pytorch
    [[ -d .git ]] || die "run the build command from the checked-out repository root"
    tar \
        --exclude='./.git/config' \
        --exclude='./.git/config.worktree' \
        --exclude='./.git/hooks' \
        --exclude='./build' \
        --exclude='./dist' \
        -cf - . | docker exec -i "${container}" tar -C /pytorch -xf -
    docker exec "${container}" git config --global --add safe.directory /pytorch
    docker exec "${container}" git -C /pytorch config remote.origin.url "${SOURCE_REPO_URL}"
    [[ "$(docker exec "${container}" git -C /pytorch rev-parse HEAD)" == "${SOURCE_SHA}" ]] || \
        die "source commit copied into ${container} does not match ${SOURCE_SHA}"
}

build_wheels() {
    local container="$1" python_tag="$2"
    local wheels

    docker exec \
        --env BUILD_PIP_INDEX_URL \
        --env BUILD_PIP_TRUSTED_HOST \
        --env MAGMA_URL \
        --env MAGMA_SHA256 \
        --env AMDGPU_TARGETS \
        --env MAX_JOBS="$(nproc --ignore=2)" \
        "${container}" bash -lc \
        'cd /pytorch && bash scripts/build_hcu.sh' \
        2>&1 | tee "${LOCAL_RUN_ROOT}/logs/build-${python_tag}.log"

    docker cp "${container}:/pytorch/dist/." "${LOCAL_RUN_ROOT}/dist/${python_tag}/"
    mapfile -t wheels < <(find "${LOCAL_RUN_ROOT}/dist/${python_tag}" -maxdepth 1 -name '*.whl' -type f)
    [[ "${#wheels[@]}" -eq 2 ]] || die "expected linux and manylinux wheels for ${python_tag}"
    for wheel in "${wheels[@]}"; do
        [[ "$(basename "${wheel}")" == *"cp${python_tag#py}-cp${python_tag#py}"* ]] || \
            die "unexpected Python tag: ${wheel}"
    done
    ls -lh "${wheels[@]}"
}

command_build() {
    require_env GITHUB_RUN_ID GITHUB_RUN_ATTEMPT DTK_PKG_URL \
        SOURCE_REPO_URL SOURCE_SHA BUILD_IMAGE_PY310 BUILD_IMAGE_PY312 \
        CONTAINER_NAME_PY310 CONTAINER_NAME_PY312 LOCAL_RUN_ROOT \
        BUILD_PIP_INDEX_URL BUILD_PIP_TRUSTED_HOST MAGMA_URL MAGMA_SHA256 \
        AMDGPU_TARGETS

    start_container "${CONTAINER_NAME_PY310}" "${BUILD_IMAGE_PY310}" 3.10
    build_wheels "${CONTAINER_NAME_PY310}" py310
    start_container "${CONTAINER_NAME_PY312}" "${BUILD_IMAGE_PY312}" 3.12
    build_wheels "${CONTAINER_NAME_PY312}" py312
    docker rm -f "${CONTAINER_NAME_PY312}" >/dev/null
}

run_test() {
    local mode="$1" devices="$2" policy="$3"
    docker exec "${CONTAINER_NAME_PY310}" bash -lc '
        set -euo pipefail
        set +u; source /opt/dtk/env.sh; set -u
        cd /pytorch
        shopt -s nullglob; wheel=(dist/*manylinux*.whl); shopt -u nullglob
        [[ "${#wheel[@]}" -eq 1 ]]
        hcu-torch-ci run --mode "$1" --repo /pytorch \
            --torch-wheel "${wheel[0]}" --devices "$2" \
            --report-dir "/pytorch/test-reports/$1" --policy "$3"
    ' _ "${mode}" "${devices}" "${policy}" \
        2>&1 | tee "${LOCAL_RUN_ROOT}/logs/test-${mode}.log"
}

command_test() {
    local suite="${1:-}"
    require_env CONTAINER_NAME_PY310 LOCAL_RUN_ROOT HCU_CI_INDEX_URL \
        BUILD_PIP_INDEX_URL BUILD_PIP_TRUSTED_HOST
    [[ "${suite}" == pr || "${suite}" == full ]] || die "unsupported test suite: ${suite}"

    docker exec \
        --env BUILD_PIP_INDEX_URL \
        --env BUILD_PIP_TRUSTED_HOST \
        --env HCU_CI_INDEX_URL \
        "${CONTAINER_NAME_PY310}" bash -lc '
            python3 -m pip install --no-cache-dir --upgrade --force-reinstall \
                --index-url "${HCU_CI_INDEX_URL}" \
                --extra-index-url "${BUILD_PIP_INDEX_URL}" \
                --trusted-host "${BUILD_PIP_TRUSTED_HOST}" hcu-torch-ci
            hcu-torch-ci --help >/dev/null
        '

    if [[ "${suite}" == pr ]]; then
        local test_rc=0
        run_test mr '0,1,2,3,4,5,6,7' strict || test_rc=1
        run_test nn '0' strict || test_rc=1
        return "${test_rc}"
    fi
    run_test full '0,1,2,3,4,5,6,7' report-only
}

command_publish() {
    local publish_wheels
    require_env PYPI_PASSWORD DEVPI_URL DTK_PKG_NAME SOURCE_SHA \
        CONTAINER_NAME_PY310 LOCAL_RUN_ROOT BUILD_PIP_INDEX_URL \
        BUILD_PIP_TRUSTED_HOST

    mapfile -t publish_wheels < <(find "${LOCAL_RUN_ROOT}/dist" \
        -mindepth 2 -maxdepth 2 -type f -name '*manylinux*.whl' -print)
    [[ "${#publish_wheels[@]}" -eq 2 ]] || die "expected two manylinux wheels to publish"

    docker exec "${CONTAINER_NAME_PY310}" rm -rf /pytorch/publish-dist /pytorch/output
    docker exec "${CONTAINER_NAME_PY310}" mkdir -p /pytorch/publish-dist
    for wheel in "${publish_wheels[@]}"; do
        docker cp "${wheel}" "${CONTAINER_NAME_PY310}:/pytorch/publish-dist/"
    done

    printf '%s\n' "${PYPI_PASSWORD}" | docker exec -i \
        --env BUILD_PIP_INDEX_URL \
        --env BUILD_PIP_TRUSTED_HOST \
        --env DEVPI_URL \
        --env DTK_PKG_NAME \
        --env SOURCE_SHA \
        "${CONTAINER_NAME_PY310}" bash -lc '
            set -euo pipefail
            IFS= read -r password
            python3 -m pip install --no-cache-dir --upgrade \
                --index-url "${BUILD_PIP_INDEX_URL}" \
                --trusted-host "${BUILD_PIP_TRUSTED_HOST}" CIUpload
            python3 -c "import importlib.metadata; print(\"CIUpload:\", importlib.metadata.version(\"CIUpload\"))"
            cd /pytorch
            shopt -s nullglob; wheels=(publish-dist/*manylinux*.whl); shopt -u nullglob
            [[ "${#wheels[@]}" -eq 2 ]]
            dtk_version="$(tr -d ".[:space:]" < /opt/dtk/.info/rocm_version)"
            CIUpload REPAIR \
                --dtk_version "${dtk_version}" \
                --git_commit "${SOURCE_SHA:0:6}" \
                --outputdir output \
                -f "${wheels[@]}"
            shopt -s nullglob; repaired=(output/*manylinux*.whl); shopt -u nullglob
            [[ "${#repaired[@]}" -eq 2 ]]
            CIUpload NIGHTLY \
                --devpi_url "${DEVPI_URL}" \
                --password "${password}" \
                --dtk_pkg_name "${DTK_PKG_NAME}" \
                --record \
                --repo /pytorch \
                -f "${repaired[@]}"
        '
}

command_collect() {
    require_env LOCAL_RUN_ROOT CONTAINER_NAME_PY310
    if container_exists "${CONTAINER_NAME_PY310}"; then
        docker cp "${CONTAINER_NAME_PY310}:/pytorch/test-reports/." \
            "${LOCAL_RUN_ROOT}/reports/" 2>/dev/null || true
    fi
    find "${LOCAL_RUN_ROOT}" -maxdepth 5 -type f -ls 2>/dev/null || true
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        while IFS= read -r summary; do
            cat "${summary}" >> "${GITHUB_STEP_SUMMARY}"
            printf '\n' >> "${GITHUB_STEP_SUMMARY}"
        done < <(find "${LOCAL_RUN_ROOT}/reports" -type f -name summary.md -print 2>/dev/null | sort)
    fi
}

command_cleanup() {
    require_env CONTAINER_NAME_PY310 CONTAINER_NAME_PY312
    docker rm -f "${CONTAINER_NAME_PY310}" >/dev/null 2>&1 || true
    docker rm -f "${CONTAINER_NAME_PY312}" >/dev/null 2>&1 || true
}

case "${1:-}" in
    build) command_build ;;
    test) command_test "${2:-}" ;;
    publish) command_publish ;;
    collect) command_collect ;;
    cleanup) command_cleanup ;;
    *) die "usage: $0 {build|test <pr|full>|publish|collect|cleanup}" ;;
esac
