#!/usr/bin/env bash

set -Eeuo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_SMI_COMPAT_PATH="${AMD_SMI_COMPAT_PATH:-/usr/local/bin/amd-smi}"

if command -v amd-smi >/dev/null 2>&1; then
    echo "Using existing amd-smi: $(command -v amd-smi)"
    exit 0
fi

test -f "${CI_DIR}/amd-smi"
install -m 0755 "${CI_DIR}/amd-smi" "${AMD_SMI_COMPAT_PATH}"
echo "Installed HCU amd-smi compatibility command: ${AMD_SMI_COMPAT_PATH}"
