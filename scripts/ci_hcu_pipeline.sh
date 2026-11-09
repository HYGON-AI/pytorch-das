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

resolve_python_artifact_dir() {
    local configured_root="$1" python_tag="$2"
    local run_root dtk_id attempt candidate
    local -a wheels

    [[ "${GITHUB_RUN_ATTEMPT}" =~ ^[0-9]+$ ]] || \
        die "invalid GITHUB_RUN_ATTEMPT: ${GITHUB_RUN_ATTEMPT}"
    run_root="$(dirname "$(dirname "${configured_root}")")"
    dtk_id="$(basename "${configured_root}")"

    for ((attempt = GITHUB_RUN_ATTEMPT; attempt >= 1; attempt--)); do
        if (( attempt == GITHUB_RUN_ATTEMPT )); then
            candidate="${configured_root}/dist/${python_tag}"
        else
            candidate="${run_root}/${attempt}/${dtk_id}/dist/${python_tag}"
        fi
        shopt -s nullglob
        wheels=("${candidate}"/*manylinux*.whl)
        shopt -u nullglob
        if (( ${#wheels[@]} == 1 )); then
            if (( attempt != GITHUB_RUN_ATTEMPT )); then
                echo "Reusing ${python_tag} build artifacts from run attempt ${attempt}: ${candidate}" >&2
            fi
            printf '%s\n' "${candidate}"
            return 0
        fi
        (( ${#wheels[@]} == 0 )) || \
            die "expected one ${python_tag} manylinux wheel in ${candidate}, found ${#wheels[@]}"
    done

    die "cannot find ${python_tag} build artifacts for run ${GITHUB_RUN_ID} through attempt ${GITHUB_RUN_ATTEMPT}"
}

set_pip_trusted_host() {
    local host="${BUILD_PIP_INDEX_URL#*://}"
    host="${host%%/*}"
    host="${host%%:*}"
    [[ -n "${host}" ]] || die "cannot determine pip host from ${BUILD_PIP_INDEX_URL}"
    export BUILD_PIP_TRUSTED_HOST="${BUILD_PIP_TRUSTED_HOST:-${host}}"
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
        --env LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
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
        dtk_stage="$(mktemp -d /tmp/pytorch-dtk.XXXXXX)"
        trap '\''rm -rf -- "${dtk_stage}"'\'' EXIT
        mkdir -p "${dtk_stage}/extract"
        echo "Downloading DTK package: ${DTK_PKG_URL}"
        wget --quiet --tries=3 --timeout=120 \
            --output-document="${dtk_stage}/dtk.tar.gz" -- "${DTK_PKG_URL}"
        echo "DTK package downloaded"
        tar -xzf "${dtk_stage}/dtk.tar.gz" -C "${dtk_stage}/extract"
        version_file="$(find "${dtk_stage}/extract" -type f -path "*/.info/rocm_version" -print -quit)"
        [[ -n "${version_file}" ]]
        dtk_source="${version_file%/.info/rocm_version}"
        installed_dtk_version="$(tr -d "[:space:]" < "${version_file}")"
        [[ -n "${installed_dtk_version}" ]]
        echo "Installed DTK version: ${installed_dtk_version}"
        rm -rf -- /opt/dtk
        mkdir -p /opt/dtk
        cp -a "${dtk_source}/." /opt/dtk/
    '

}

copy_source() {
    local container="$1"

    require_env SOURCE_REPO_URL SOURCE_SHA
    docker exec "${container}" mkdir -p /pytorch
    [[ -d .git ]] || die "run the build command from the checked-out repository root"
    tar \
        --exclude='./.git/config' \
        --exclude='./.git/config.worktree' \
        --exclude='./.git/hooks' \
        --exclude='./build' \
        --exclude='./dist' \
        -cf - . | docker exec -i "${container}" \
            tar --no-same-owner -C /pytorch -xf -
    docker exec "${container}" git config --global --add safe.directory /pytorch
    docker exec "${container}" git -C /pytorch config remote.origin.url "${SOURCE_REPO_URL}"
}

build_wheels() {
    local container="$1" python_tag="$2"
    docker exec \
        --env PIP_INDEX_URL="${BUILD_PIP_INDEX_URL}" \
        --env PIP_TRUSTED_HOST="${BUILD_PIP_TRUSTED_HOST}" \
        --env PYTHONNOUSERSITE=1 \
        --env RESOURCE_SERVER_URL \
        --env MAGMA_URL \
        --env MAGMA_SHA256 \
        --env MAX_JOBS="$(nproc --ignore=2)" \
        "${container}" bash -c \
        'cd /pytorch && bash scripts/build_hcu.sh' \
        2>&1 | tee "${LOCAL_RUN_ROOT}/logs/build-${python_tag}.log"

    docker cp "${container}:/pytorch/dist/." "${LOCAL_RUN_ROOT}/dist/${python_tag}/"

}

build_one() {
    local container="$1" image="$2" expected_python="$3" python_tag="$4"

    start_container "${container}" "${image}" "${expected_python}"
    copy_source "${container}"
    build_wheels "${container}" "${python_tag}"
}

command_build() {
    local python_version="${1:-}" image container python_tag
    require_env GITHUB_RUN_ID GITHUB_RUN_ATTEMPT DTK_PKG_URL \
        SOURCE_REPO_URL SOURCE_SHA LOCAL_RUN_ROOT \
        BUILD_PIP_INDEX_URL RESOURCE_SERVER_URL

    case "${python_version}" in
        3.10)
            image="${BUILD_IMAGE_PY310:-}"
            container="${CONTAINER_NAME_PY310:-}"
            python_tag="py310"
            ;;
        3.12)
            image="${BUILD_IMAGE_PY312:-}"
            container="${CONTAINER_NAME_PY312:-}"
            python_tag="py312"
            ;;
        *)
            die "build requires a supported Python version (3.10 or 3.12), got: ${python_version:-<empty>}"
            ;;
    esac
    [[ -n "${image}" ]] || die "build image is not set for Python ${python_version}"
    [[ -n "${container}" ]] || die "container name is not set for Python ${python_version}"
    set_pip_trusted_host

    mkdir -p "${LOCAL_RUN_ROOT}/dist/${python_tag}" \
        "${LOCAL_RUN_ROOT}/logs" \
        "${LOCAL_RUN_ROOT}/reports"

    build_one "${container}" "${image}" "${python_version}" "${python_tag}"
}

run_test() {
    local mode="$1" devices="$2" policy="$3"
    docker exec "${CONTAINER_NAME}" bash -lc '
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

install_aicc() {
    local installer_url="${AICC_INSTALLER_URL:-${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh}"
    local major_version="${AICC_MAJOR_VERSION:-}"

    docker exec \
        --env AICC_INSTALLER_URL="${installer_url}" \
        --env AICC_MAJOR_VERSION="${major_version}" \
        "${CONTAINER_NAME}" bash -lc '
            set -euo pipefail
            if [ -z "${AICC_MAJOR_VERSION}" ]; then
                dtk_version="$(tr -d ".[:space:]" < /opt/dtk/.info/rocm_version)"
                case "${dtk_version}" in
                    2610*) AICC_MAJOR_VERSION="2.0.0" ;;
                    *) AICC_MAJOR_VERSION="1.0.0" ;;
                esac
            fi
            installer="$(mktemp)"
            trap '\''rm -f -- "${installer}"'\'' EXIT
            echo "Installing AICC ${AICC_MAJOR_VERSION} for test job"
            wget --quiet --tries=3 --timeout=120 \
                --output-document="${installer}" -- "${AICC_INSTALLER_URL}"
            bash "${installer}" --major "${AICC_MAJOR_VERSION}" --dtk-dir /opt/dtk
            test -x /opt/dtk/bin/aicc || {
                echo "ERROR: AICC was not installed at /opt/dtk/bin/aicc" >&2
                exit 1
            }
        '
}

command_test() {
    local mode="${1:-}" devices="${2:-}" policy="${3:-}"
    local artifact_root="${BUILD_ARTIFACT_ROOT:-${LOCAL_RUN_ROOT:-}}"
    local artifact_dir
    require_env GITHUB_RUN_ID GITHUB_RUN_ATTEMPT DTK_PKG_URL \
        SOURCE_REPO_URL SOURCE_SHA BUILD_IMAGE_PY310 CONTAINER_NAME \
        LOCAL_RUN_ROOT HCU_CI_INDEX_URL BUILD_PIP_INDEX_URL \
        RESOURCE_SERVER_URL
    set_pip_trusted_host

    artifact_dir="$(resolve_python_artifact_dir "${artifact_root}" py310)"
    mkdir -p "${LOCAL_RUN_ROOT}/logs" "${LOCAL_RUN_ROOT}/reports"

    start_container "${CONTAINER_NAME}" "${BUILD_IMAGE_PY310}" 3.10
    copy_source "${CONTAINER_NAME}"
    docker exec "${CONTAINER_NAME}" mkdir -p /pytorch/dist
    docker cp "${artifact_dir}/." "${CONTAINER_NAME}:/pytorch/dist/"
    install_aicc
    docker exec \
        --env BUILD_PIP_INDEX_URL \
        --env BUILD_PIP_TRUSTED_HOST \
        --env HCU_CI_INDEX_URL \
        "${CONTAINER_NAME}" bash -lc '
            set -euo pipefail
            python3 -m pip install --no-cache-dir --upgrade --force-reinstall \
                --index-url "${HCU_CI_INDEX_URL}" \
                --extra-index-url "${BUILD_PIP_INDEX_URL}" \
                --trusted-host "${BUILD_PIP_TRUSTED_HOST}" hcu-torch-ci
        '

    run_test "${mode}" "${devices}" "${policy}"
}

command_publish() {
    local artifact_dir python_tag
    local -a publish_wheels python_wheels
    require_env GITHUB_RUN_ID GITHUB_RUN_ATTEMPT \
        PYPI_PASSWORD DEVPI_URL DTK_PKG_NAME SOURCE_SHA \
        CONTAINER_NAME LOCAL_RUN_ROOT BUILD_IMAGE_PY310 \
        BUILD_PIP_INDEX_URL
    set_pip_trusted_host

    mkdir -p "${LOCAL_RUN_ROOT}"
    publish_wheels=()
    for python_tag in py310 py312; do
        artifact_dir="$(resolve_python_artifact_dir "${LOCAL_RUN_ROOT}" "${python_tag}")"
        shopt -s nullglob
        python_wheels=("${artifact_dir}"/*manylinux*.whl)
        shopt -u nullglob
        publish_wheels+=("${python_wheels[0]}")
    done

    start_container "${CONTAINER_NAME}" "${BUILD_IMAGE_PY310}" 3.10
    copy_source "${CONTAINER_NAME}"
    docker exec "${CONTAINER_NAME}" mkdir -p /pytorch/publish-dist
    for wheel in "${publish_wheels[@]}"; do
        docker cp "${wheel}" "${CONTAINER_NAME}:/pytorch/publish-dist/"
    done

    printf '%s\n' "${PYPI_PASSWORD}" | docker exec -i \
        --env BUILD_PIP_INDEX_URL \
        --env BUILD_PIP_TRUSTED_HOST \
        --env DEVPI_URL \
        --env DTK_PKG_NAME \
        --env SOURCE_SHA \
        "${CONTAINER_NAME}" bash -lc '
            set -euo pipefail
            IFS= read -r password
            python3 -m pip install --no-cache-dir --upgrade \
                --index-url "${BUILD_PIP_INDEX_URL}" \
                --trusted-host "${BUILD_PIP_TRUSTED_HOST}" CIUpload
            python3 -c "import importlib.metadata; print(\"CIUpload:\", importlib.metadata.version(\"CIUpload\"))"
            cd /pytorch
            shopt -s nullglob; wheels=(publish-dist/*manylinux*.whl); shopt -u nullglob
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
    require_env LOCAL_RUN_ROOT CONTAINER_NAME
    docker cp "${CONTAINER_NAME}:/pytorch/test-reports/." \
        "${LOCAL_RUN_ROOT}/reports/" 2>/dev/null || true
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        while IFS= read -r summary; do
            cat "${summary}" >> "${GITHUB_STEP_SUMMARY}"
            printf '\n' >> "${GITHUB_STEP_SUMMARY}"
        done < <(find "${LOCAL_RUN_ROOT}/reports" -type f -name summary.md -print 2>/dev/null | sort)
    fi
}

command_cleanup() {
    local container
    for container in "${CONTAINER_NAME:-}" "${CONTAINER_NAME_PY310:-}" "${CONTAINER_NAME_PY312:-}"; do
        [[ -n "${container}" ]] || continue
        docker rm -f "${container}" >/dev/null 2>&1 || true
    done
}

case "${1:-}" in
    build) command_build "${2:-}" ;;
    test) command_test "${2:-}" "${3:-}" "${4:-}" ;;
    publish) command_publish ;;
    collect) command_collect ;;
    cleanup) command_cleanup ;;
    *) die "usage: $0 {build|test <mr|nn|full> <devices> <strict|report-only>|publish|collect|cleanup}" ;;
esac
