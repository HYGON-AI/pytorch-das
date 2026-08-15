#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_ROOT}"' EXIT

mkdir -p "${TEMP_ROOT}/bin" "${TEMP_ROOT}/dtk" "${TEMP_ROOT}/fixture"

cat > "${TEMP_ROOT}/fixture/fake-installer.run" <<'INSTALLER'
#!/usr/bin/env bash
set -euo pipefail
test "${1:-}" = "--dtk_dir"
dtk_dir="${2:?missing DTK directory}"
mkdir -p "${dtk_dir}/aillvm/bin"
printf 'fixture\n' > "${dtk_dir}/aillvm/bin/clang"
INSTALLER
chmod 755 "${TEMP_ROOT}/fixture/fake-installer.run"

cat > "${TEMP_ROOT}/bin/wget" <<'MOCK_WGET'
#!/usr/bin/env bash
set -euo pipefail
output=""
for argument in "$@"; do
    case "${argument}" in
        --output-document=-)
            printf '%s\n' \
                '<a href="dtk_llvm_zzzzzz_202607311200.run">older</a>' \
                '<a href="dtk_llvm_aaaaaa_202608031615.run">middle</a>' \
                '<a href="dtk_llvm_bbbbbb_202608141230.run">latest</a>'
            exit 0
            ;;
        --output-document=*)
            output="${argument#*=}"
            ;;
    esac
done
test -n "${output}"
cp "${MOCK_AICC_INSTALLER}" "${output}"
printf '%s\n' "${*: -1}" > "${MOCK_AICC_DOWNLOAD_LOG}"
MOCK_WGET
chmod 755 "${TEMP_ROOT}/bin/wget"

PATH="${TEMP_ROOT}/bin:${PATH}" \
MOCK_AICC_INSTALLER="${TEMP_ROOT}/fixture/fake-installer.run" \
MOCK_AICC_DOWNLOAD_LOG="${TEMP_ROOT}/download-url.txt" \
AICC_BASE_URL="http://fixture/ai_cc/nightly" \
AICC_DTK_DIR="${TEMP_ROOT}/dtk" \
bash "${CI_DIR}/install-aicc.sh"

grep -Fxq \
    'http://fixture/ai_cc/nightly/dtk_llvm_bbbbbb_202608141230.run' \
    "${TEMP_ROOT}/download-url.txt"
grep -Fxq \
    'dtk_llvm_bbbbbb_202608141230.run' \
    "${TEMP_ROOT}/dtk/.info/aicc_installer_name"
test -s "${TEMP_ROOT}/dtk/.info/aicc_installer_sha256"
test -s "${TEMP_ROOT}/dtk/aillvm/bin/clang"

# A second call must use the marker and not download or reinstall the same build.
rm -f "${TEMP_ROOT}/download-url.txt" "${TEMP_ROOT}/fixture/fake-installer.run"
PATH="${TEMP_ROOT}/bin:${PATH}" \
MOCK_AICC_INSTALLER="${TEMP_ROOT}/fixture/fake-installer.run" \
MOCK_AICC_DOWNLOAD_LOG="${TEMP_ROOT}/download-url.txt" \
AICC_BASE_URL="http://fixture/ai_cc/nightly" \
AICC_DTK_DIR="${TEMP_ROOT}/dtk" \
bash "${CI_DIR}/install-aicc.sh"
test ! -e "${TEMP_ROOT}/download-url.txt"

echo "AICC latest-installer selection and idempotency passed"
