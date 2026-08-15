#!/usr/bin/env bash

set -Eeuo pipefail

AICC_BASE_URL="${AICC_BASE_URL:-http://42.228.13.241:18000/ai_cc/nightly}"
AICC_DTK_DIR="${AICC_DTK_DIR:-/opt/dtk}"
AICC_INSTALLER_URL="${AICC_INSTALLER_URL:-}"
AICC_INSTALLER_SHA256="${AICC_INSTALLER_SHA256:-}"

resolve_latest_installer() {
    local index_html
    local candidate
    local timestamp
    local latest

    index_html="$(wget --quiet --output-document=- "${AICC_BASE_URL%/}/")"
    latest=""
    while IFS= read -r candidate; do
        timestamp="${candidate%.run}"
        timestamp="${timestamp##*_}"
        if [[ ! "${timestamp}" =~ ^[0-9]{8}([0-9]{4})?$ ]]; then
            continue
        fi
        if [[ -z "${latest}" || "${timestamp} ${candidate}" > "${latest}" ]]; then
            latest="${timestamp} ${candidate}"
        fi
    done < <(
        grep -oE 'dtk_llvm_[[:alnum:].-]+_[0-9]{8}([0-9]{4})?\.run' \
            <<< "${index_html}" | sort -u || true
    )

    if [[ -z "${latest}" ]]; then
        echo "ERROR: no dated dtk_llvm installer found at ${AICC_BASE_URL%/}/" >&2
        return 1
    fi
    printf '%s/%s\n' "${AICC_BASE_URL%/}" "${latest#* }"
}

if [[ -z "${AICC_INSTALLER_URL}" ]]; then
    AICC_INSTALLER_URL="$(resolve_latest_installer)"
fi

installer_name="${AICC_INSTALLER_URL%%\?*}"
installer_name="${installer_name##*/}"
if [[ ! "${installer_name}" =~ ^dtk_llvm_[[:alnum:].-]+_[0-9]{8}([0-9]{4})?\.run$ ]]; then
    echo "ERROR: unexpected AICC installer name: ${installer_name}" >&2
    exit 1
fi

marker_dir="${AICC_DTK_DIR}/.info"
name_marker="${marker_dir}/aicc_installer_name"
url_marker="${marker_dir}/aicc_installer_url"
sha_marker="${marker_dir}/aicc_installer_sha256"

if [[ -d "${AICC_DTK_DIR}/aillvm" && -f "${url_marker}" &&
    "$(< "${url_marker}")" == "${AICC_INSTALLER_URL}" ]]; then
    echo "AICC already installed: ${installer_name}"
    exit 0
fi

test -d "${AICC_DTK_DIR}"
installer_path="$(mktemp "/tmp/${installer_name}.XXXXXX")"
cleanup() {
    rm -f -- "${installer_path}"
}
trap cleanup EXIT

echo "Downloading AICC installer: ${AICC_INSTALLER_URL}"
case "${AICC_INSTALLER_URL}" in
    http://*|https://*)
        wget \
            --tries=5 \
            --timeout=120 \
            --output-document="${installer_path}" \
            "${AICC_INSTALLER_URL}"
        ;;
    *)
        test -f "${AICC_INSTALLER_URL}"
        cp "${AICC_INSTALLER_URL}" "${installer_path}"
        ;;
esac

actual_sha256="$(sha256sum "${installer_path}" | awk '{print $1}')"
if [[ -n "${AICC_INSTALLER_SHA256}" &&
    "${actual_sha256}" != "${AICC_INSTALLER_SHA256}" ]]; then
    echo "ERROR: AICC installer SHA256 mismatch" >&2
    echo "expected=${AICC_INSTALLER_SHA256}" >&2
    echo "actual=${actual_sha256}" >&2
    exit 1
fi

chmod 755 "${installer_path}"
echo "Installing ${installer_name} into ${AICC_DTK_DIR}"
set +o pipefail
yes | "${installer_path}" --dtk_dir "${AICC_DTK_DIR}"
installer_rc="${PIPESTATUS[1]}"
set -o pipefail
if (( installer_rc != 0 )); then
    echo "ERROR: AICC installer exited with code ${installer_rc}" >&2
    exit "${installer_rc}"
fi

if [[ ! -d "${AICC_DTK_DIR}/aillvm" ]]; then
    echo "ERROR: AICC installation did not create ${AICC_DTK_DIR}/aillvm" >&2
    exit 1
fi
if [[ -z "$(find "${AICC_DTK_DIR}/aillvm" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: ${AICC_DTK_DIR}/aillvm is empty after installation" >&2
    exit 1
fi

mkdir -p "${marker_dir}"
printf '%s\n' "${installer_name}" > "${name_marker}"
printf '%s\n' "${AICC_INSTALLER_URL}" > "${url_marker}"
printf '%s\n' "${actual_sha256}" > "${sha_marker}"

echo "AICC installation completed"
echo "installer=${installer_name}"
echo "sha256=${actual_sha256}"
find "${AICC_DTK_DIR}/aillvm" -maxdepth 2 -type f -print | sed -n '1,40p'
