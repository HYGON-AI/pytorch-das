#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(realpath -- "${repo_root}")"
prepared_root="$(realpath -- "${HCU_PREPARED_SOURCE_ROOT:-/pytorch}")"
backup_root="${repo_root}/.hcu-source-backup"
stamp_file="${repo_root}/.hcu-source-preparation.env"
rocm_path="${ROCM_PATH:-/opt/dtk}"

if [[ "${HCU_ALLOW_SOURCE_PREPARE:-0}" != "1" ]]; then
  echo "Set HCU_ALLOW_SOURCE_PREPARE=1 to prepare a disposable HCU CI checkout" >&2
  exit 1
fi
if [[ "${prepared_root}" == "${repo_root}" || "${prepared_root}" == "${repo_root}/"* ]]; then
  echo "HCU prepared source root must be outside the current checkout" >&2
  exit 1
fi
if [[ -e "${backup_root}" || -e "${stamp_file}" ]]; then
  echo "The current checkout has already been prepared for an HCU wheel build" >&2
  exit 1
fi

source_version="$(tr -d '[:space:]' < "${repo_root}/version.txt")"
prepared_version="$(tr -d '[:space:]' < "${prepared_root}/version.txt")"
dtk_version="$(tr -d '[:space:]' < "${rocm_path}/.info/rocm_version")"
[[ "${source_version}" == 2.12.* ]]
[[ "${prepared_version}" == 2.12.* ]]
[[ "${dtk_version}" == 26.04* ]]

required_paths=(
  "${prepared_root}/third_party/miniz-3.0.2"
  "${prepared_root}/third_party/gloo/tools/amd_build/build_amd.py"
  "${prepared_root}/third_party/mslk/include/mslk/utils/hip/tuning_cache.cuh"
  "${prepared_root}/android/libs/fbjni"
  "${prepared_root}/install_dir"
  "${prepared_root}/torch/lib/libmagma.so"
  "${prepared_root}/torch/lib/libmkl_core.so"
  "${prepared_root}/torch/lib/libmkl_def.so"
  "${prepared_root}/torch/lib/libmkl_gf_lp64.so"
  "${prepared_root}/torch/lib/libmkl_gnu_thread.so"
  "${prepared_root}/torch/lib/libmkl_intel_lp64.so"
  "${prepared_root}/torch/lib/libmkl_vml_def.so"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing /torch2.12 prepared build input: ${path}" >&2
    exit 1
  fi
done

install -d -m 0755 "${backup_root}" "${repo_root}/android/libs"
mv -- "${repo_root}/third_party" "${backup_root}/third_party"
mv -- "${repo_root}/android/libs/fbjni" "${backup_root}/fbjni"
if [[ -e "${repo_root}/install_dir" ]]; then
  mv -- "${repo_root}/install_dir" "${backup_root}/install_dir"
fi

cp -a -- "${prepared_root}/third_party" "${repo_root}/third_party"
cp -a -- "${prepared_root}/android/libs/fbjni" "${repo_root}/android/libs/fbjni"
cp -a -- "${prepared_root}/install_dir" "${repo_root}/install_dir"
install -d -m 0755 "${repo_root}/torch/lib"
cp -a -- "${prepared_root}"/torch/lib/libmagma.so "${repo_root}/torch/lib/"
cp -a -- "${prepared_root}"/torch/lib/libmkl_*.so "${repo_root}/torch/lib/"

sed -i "s#/opt/rocm#${rocm_path}#g" "${repo_root}/cmake/Dependencies.cmake"
chmod 0755 "${repo_root}/third_party/gloo/tools/amd_build/build_amd.py"

source_sha="${SOURCE_SHA:-$(git -C "${repo_root}" rev-parse HEAD)}"
third_party_seed_sha256="$(
  sha256sum \
    "${repo_root}/third_party/mslk/include/mslk/utils/hip/tuning_cache.cuh" |
    awk '{print $1}'
)"
{
  printf "source_profile=torch2.12-image-overlay-v1\n"
  printf "source_sha=%s\n" "${source_sha}"
  printf "prepared_source_root=%s\n" "${prepared_root}"
  printf "prepared_source_version=%s\n" "${prepared_version}"
  printf "dtk_version=%s\n" "${dtk_version}"
  printf "third_party_seed_sha256=%s\n" "${third_party_seed_sha256}"
} > "${stamp_file}"

printf "Prepared source profile: torch2.12-image-overlay-v1\n"
printf "Prepared source root: %s\n" "${prepared_root}"
printf "Third-party seed SHA256: %s\n" "${third_party_seed_sha256}"
