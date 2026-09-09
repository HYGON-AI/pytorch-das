#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PATCH_ROOT="${SCRIPT_DIR}/patches"
LIBRARIES=(fbgemm kineto)

die() {
  echo "error: $*" >&2
  exit 1
}

load_series() {
  local series_file="$1"
  local series_dir
  local entry

  series_dir="$(dirname -- "${series_file}")"
  PATCH_FILES=()

  while IFS= read -r entry || [[ -n "${entry}" ]]; do
    entry="${entry%$'\r'}"
    [[ -z "${entry}" || "${entry}" == \#* ]] && continue
    [[ -f "${series_dir}/${entry}" ]] ||
      die "patch file not found: ${series_dir}/${entry}"
    PATCH_FILES+=("${series_dir}/${entry}")
  done < "${series_file}"

  [[ ${#PATCH_FILES[@]} -gt 0 ]] || die "empty patch series: ${series_file}"
}

[[ $# -eq 0 ]] || die "usage: tools/third_party/apply_patches.sh"

for library in "${LIBRARIES[@]}"; do
  submodule_dir="${REPO_ROOT}/third_party/${library}"
  series_file="${PATCH_ROOT}/${library}/series"

  [[ -f "${series_file}" ]] || die "series file not found: ${series_file}"
  git -C "${submodule_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    die "submodule not initialized: third_party/${library}"
  load_series "${series_file}"

  if git -C "${submodule_dir}" apply --check "${PATCH_FILES[@]}" >/dev/null 2>&1; then
    git -C "${submodule_dir}" apply "${PATCH_FILES[@]}"
    echo "third_party/${library}: patched"
    continue
  fi

  REVERSE_PATCH_FILES=()
  for ((i=${#PATCH_FILES[@]} - 1; i >= 0; i--)); do
    REVERSE_PATCH_FILES+=("${PATCH_FILES[i]}")
  done
  if git -C "${submodule_dir}" apply --reverse --check \
      "${REVERSE_PATCH_FILES[@]}" >/dev/null 2>&1; then
    echo "third_party/${library}: already patched"
    continue
  fi

  die "third_party/${library}: patch cannot be applied"
done
