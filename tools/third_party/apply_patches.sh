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

# A single `git apply p1 p2` checks every patch against the original tree, so
# a later patch cannot touch a file an earlier one already changed. Apply the
# series one patch at a time on a scratch copy of the touched files instead.
series_applies() {
  local submodule_dir="$1"
  shift
  local apply_args=()
  local scratch
  local patch
  local path
  local status=0

  [[ "$1" == "--reverse" ]] && { apply_args=(--reverse); shift; }
  scratch="$(mktemp -d)"
  git init -q "${scratch}"

  for patch in "$@"; do
    while IFS=$'\t' read -r _ _ path; do
      [[ -f "${submodule_dir}/${path}" ]] || continue
      mkdir -p "${scratch}/$(dirname -- "${path}")"
      cp "${submodule_dir}/${path}" "${scratch}/${path}"
    done < <(git -C "${submodule_dir}" apply --numstat "${patch}")
  done

  for patch in "$@"; do
    git -C "${scratch}" apply ${apply_args[@]+"${apply_args[@]}"} "${patch}" \
        >/dev/null 2>&1 ||
      { status=1; break; }
  done
  rm -rf "${scratch}"
  return "${status}"
}

[[ $# -eq 0 ]] || die "usage: tools/third_party/apply_patches.sh"

for library in "${LIBRARIES[@]}"; do
  submodule_dir="${REPO_ROOT}/third_party/${library}"
  series_file="${PATCH_ROOT}/${library}/series"

  [[ -f "${series_file}" ]] || die "series file not found: ${series_file}"
  git -C "${submodule_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    die "submodule not initialized: third_party/${library}"
  load_series "${series_file}"

  if series_applies "${submodule_dir}" "${PATCH_FILES[@]}"; then
    for patch_file in "${PATCH_FILES[@]}"; do
      git -C "${submodule_dir}" apply "${patch_file}"
    done
    echo "third_party/${library}: patched"
    continue
  fi

  REVERSE_PATCH_FILES=()
  for ((i=${#PATCH_FILES[@]} - 1; i >= 0; i--)); do
    REVERSE_PATCH_FILES+=("${PATCH_FILES[i]}")
  done
  if series_applies "${submodule_dir}" --reverse "${REVERSE_PATCH_FILES[@]}"; then
    echo "third_party/${library}: already patched"
    continue
  fi

  die "third_party/${library}: patch cannot be applied"
done
