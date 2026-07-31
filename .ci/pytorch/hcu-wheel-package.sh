#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

rocm_path="${ROCM_PATH:-/opt/dtk}"
version_file="${repo_root}/version.txt"
dtk_version_file="${rocm_path}/.info/rocm_version"

test -f "${version_file}"
test -f "${dtk_version_file}"
test "${HCU_REUSE_OLD_WHL:-0}" = "0"
test -f "${repo_root}/.hcu-source-preparation.env"
grep -qx "source_profile=torch2.12-image-overlay-v1" \
  "${repo_root}/.hcu-source-preparation.env"

base_version="$(tr -d '[:space:]' < "${version_file}")"
dtk_version="$(tr -d '[:space:]' < "${dtk_version_file}")"
dtk_version_number="${dtk_version//./}"
actual_source_sha="$(git rev-parse HEAD)"
source_sha="${SOURCE_SHA:-${actual_source_sha}}"
if [[ "${source_sha}" != "${actual_source_sha}" ]]; then
  echo "SOURCE_SHA ${source_sha} does not match checkout HEAD ${actual_source_sha}" >&2
  exit 1
fi
source_sha_short="${source_sha:0:7}"
package_type="${HCU_PACKAGE_TYPE:-ci}"

case "${package_type}" in
  ci | release)
    package_version="${base_version}+das.opt1.dtk${dtk_version_number}"
    ;;
  nightly)
    package_version="${base_version}+das.opt1.dtk${dtk_version_number}.$(date -u +%y%m%d%H%M).g${source_sha_short}"
    ;;
  *)
    echo "Unsupported HCU_PACKAGE_TYPE: ${package_type}" >&2
    exit 1
    ;;
esac

export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION:-${package_version}}"
export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-1}"
export ROCM_PATH="${rocm_path}"
export DTKROOT="${rocm_path}"
export HIP_PATH="${rocm_path}/hip"
export PATH="${rocm_path}/bin:${rocm_path}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${rocm_path}/lib:${rocm_path}/lib64:${rocm_path}/magma/lib:/opt/intel/oneapi/mkl/2024.2/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${rocm_path}:${CMAKE_PREFIX_PATH:-}"

if [[ -f "${rocm_path}/env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090,SC1091
  source "${rocm_path}/env.sh"
  set -u
fi

if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-13/enable
  set -u
fi

export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export CMAKE_C_COMPILER="${CC}"
export CMAKE_CXX_COMPILER="${CXX}"

hip_compiler=""
for candidate in "${rocm_path}/llvm/bin/clang++" "${rocm_path}/bin/clang++"; do
  if [[ -x "${candidate}" ]]; then
    hip_compiler="${candidate}"
    break
  fi
done
if [[ -z "${hip_compiler}" ]]; then
  echo "Unable to find the DTK clang++ compiler under ${rocm_path}" >&2
  exit 1
fi
test -f "${rocm_path}/lib/cmake/hip-lang/hip-lang-config.cmake"

export HIPCXX="${hip_compiler}"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_HIP_COMPILER=${hip_compiler} -DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocm_path} -DCMAKE_PREFIX_PATH=${rocm_path}"
export CXXFLAGS="${CXXFLAGS:-} -ffunction-sections -fdata-sections"
export LDFLAGS="${LDFLAGS:-} -Wl,--gc-sections"

export USE_ROCM_CK_GEMM="${USE_ROCM_CK_GEMM:-0}"
export USE_ROCM_CK_SDPA="${USE_ROCM_CK_SDPA:-0}"
export USE_CUFILE="${USE_CUFILE:-0}"
export USE_AOTRITON_FA="${USE_AOTRITON_FA:-0}"
export USE_MEM_EFF_ATTENTION="${USE_MEM_EFF_ATTENTION:-0}"
export USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-1}"
export USE_FBGEMM="${USE_FBGEMM:-ON}"
export USE_MKLDNN="${USE_MKLDNN:-0}"
export USE_MKL="${USE_MKL:-ON}"
export USE_NINJA="${USE_NINJA:-ON}"
export USE_CUDA="${USE_CUDA:-0}"
export USE_KINETO="${USE_KINETO:-1}"
export BUILD_STATIC_RUNTIME_BENCHMARK="${BUILD_STATIC_RUNTIME_BENCHMARK:-OFF}"
export BUILD_TEST="${BUILD_TEST:-OFF}"
export USE_MAGMA="${USE_MAGMA:-ON}"
export NO_CAFFE2_OPS="${NO_CAFFE2_OPS:-1}"
export USE_RCCL="${USE_RCCL:-1}"
export USE_DISTRIBUTED="${USE_DISTRIBUTED:-1}"
export USE_GLOO_IBVERBS="${USE_GLOO_IBVERBS:-1}"
export USE_GLOO="${USE_GLOO:-1}"
export USE_IBVERBS="${USE_IBVERBS:-1}"
export USE_C10D_MPI="${USE_C10D_MPI:-1}"
export USE_MPI="${USE_MPI:-1}"
export USE_OPENMP="${USE_OPENMP:-1}"
export USE_MIOPEN="${USE_MIOPEN:-1}"
export USE_ROCM="${USE_ROCM:-1}"
export USE_NUMPY="${USE_NUMPY:-1}"
export USE_GLOG="${USE_GLOG:-1}"
export USE_GFLAGS="${USE_GFLAGS:-1}"

test -x "${CC}"
test -x "${CXX}"
test -f "${rocm_path}/magma/lib/libmagma.so"
command -v mpicc

python3 -m pip install --disable-pip-version-check \
  "auditwheel==${HCU_AUDITWHEEL_VERSION:-6.7.0}" \
  "packaging==${HCU_PACKAGING_VERSION:-24.2}" \
  "patchelf==${HCU_PATCHELF_VERSION:-0.17.2.4}" \
  "wheel==${HCU_WHEEL_VERSION:-0.47.0}"

printf "HCU package type: %s\n" "${package_type}"
printf "PyTorch package version: %s\n" "${PYTORCH_BUILD_VERSION}"
printf "Source SHA: %s\n" "${source_sha}"
printf "DTK version: %s\n" "${dtk_version}"
printf "C compiler: %s\n" "$("${CC}" --version | head -n 1)"
printf "C++ compiler: %s\n" "$("${CXX}" --version | head -n 1)"
printf "HIP compiler: %s\n" "$("${hip_compiler}" --version | head -n 1)"

.ci/pytorch/build.sh

shopt -s nullglob
compiler_wheels=(dist/*.whl)
shopt -u nullglob
if (( ${#compiler_wheels[@]} != 1 )); then
  echo "Expected exactly one compiler wheel in dist, found ${#compiler_wheels[@]}" >&2
  printf "%s\n" "${compiler_wheels[@]}"
  exit 1
fi

prepared_lib_dir="${HCU_PREPARED_SOURCE_ROOT:-/pytorch}/torch/lib"
runtime_libraries=(
  libmagma.so
  libmkl_core.so
  libmkl_def.so
  libmkl_gf_lp64.so
  libmkl_gnu_thread.so
  libmkl_intel_lp64.so
  libmkl_vml_def.so
)
for library in "${runtime_libraries[@]}"; do
  test -f "${prepared_lib_dir}/${library}"
done

# setup.py clean removes the libraries seeded in torch/lib by source
# preparation. Repack the compiler wheel with the exact /torch2.12 runtime
# libraries before auditwheel resolves SONAMEs and manylinux dependencies.
wheel_work="$(mktemp -d /tmp/hcu-wheel-enrich.XXXXXX)"
compiler_wheel="${wheel_work}/$(basename "${compiler_wheels[0]}")"
mv -- "${compiler_wheels[0]}" "${compiler_wheel}"
python3 -m wheel unpack --dest "${wheel_work}/unpacked" "${compiler_wheel}"
shopt -s nullglob
unpacked_wheels=("${wheel_work}/unpacked"/*)
shopt -u nullglob
if (( ${#unpacked_wheels[@]} != 1 )); then
  echo "Expected exactly one unpacked HCU wheel" >&2
  exit 1
fi
install -d -m 0755 "${unpacked_wheels[0]}/torch/lib"
for library in "${runtime_libraries[@]}"; do
  cp -a -- "${prepared_lib_dir}/${library}" "${unpacked_wheels[0]}/torch/lib/"
done
python3 -m wheel pack --dest-dir dist "${unpacked_wheels[0]}"

shopt -s nullglob
raw_wheels=(dist/*.whl)
shopt -u nullglob
if (( ${#raw_wheels[@]} != 1 )); then
  echo "Expected exactly one enriched raw HCU wheel in dist, found ${#raw_wheels[@]}" >&2
  printf "%s\n" "${raw_wheels[@]}"
  exit 1
fi

install -d -m 0755 dist/raw
mv -- "${raw_wheels[0]}" dist/raw/

python3 -m pip install --disable-pip-version-check \
  --force-reinstall \
  --no-deps \
  dist/raw/*.whl

torch_path="$(
  cd /
  python3 -c '
import importlib.util

spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Unable to find the installed torch package")
print(spec.submodule_search_locations[0])
'
)"
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/2024.2/lib:${torch_path}/lib:${LD_LIBRARY_PATH:-}"

glibc_version="$(ldd --version | awk 'NR == 1 {print $NF}')"
platform="manylinux_${glibc_version//./_}_x86_64"
exclude_libraries=(
  libc10.so
  libc10_hip.so
  libgalaxyhip.so.5
  libhipblas.so.0
  libhipblas.so.2
  libhipblaslt.so.0
  libhipfft.so
  libhipfft.so.0
  libhipnn.so
  libhiprand.so.1
  libhipsolver.so.0
  libhipsparse.so.0
  libhipsparse.so.1
  libMIOpen.so.1
  libnuma.so.1
  libomp.so
  librccl.so.1
  librocblas.so.0
  librocblas.so.4
  librocfft-device-0.so.0
  librocfft-device-1.so.0
  librocfft-device-2.so.0
  librocfft-device-3.so.0
  librocfft.so.0
  librocm_smi64.so.2
  librocrand.so.1
  librocsolver.so.0
  librocsparse.so.0
  librocsparse.so.1
  libtorch_cpu.so
  libtorch_hip.so
  libtorch_python.so
  libtorch.so
)
auditwheel_args=()
for library in "${exclude_libraries[@]}"; do
  auditwheel_args+=(--exclude "${library}")
done

auditwheel repair \
  --plat "${platform}" \
  "${auditwheel_args[@]}" \
  --wheel-dir dist \
  dist/raw/*.whl

shopt -s nullglob
repaired_wheels=(dist/*.whl)
shopt -u nullglob
if (( ${#repaired_wheels[@]} != 1 )); then
  echo "Expected exactly one repaired HCU wheel in dist, found ${#repaired_wheels[@]}" >&2
  printf "%s\n" "${repaired_wheels[@]}"
  exit 1
fi

# oneMKL loads its dispatcher libraries with dlopen rather than DT_NEEDED, so
# auditwheel cannot discover them. MAGMA's repaired RPATH points at torch.libs;
# preserve the exact SONAME filenames that the oneMKL loader requests.
finalize_work="$(mktemp -d /tmp/hcu-wheel-finalize.XXXXXX)"
auditwheel_wheel="${finalize_work}/$(basename "${repaired_wheels[0]}")"
mv -- "${repaired_wheels[0]}" "${auditwheel_wheel}"
python3 -m wheel unpack --dest "${finalize_work}/unpacked" "${auditwheel_wheel}"
shopt -s nullglob
finalize_wheels=("${finalize_work}/unpacked"/*)
shopt -u nullglob
if (( ${#finalize_wheels[@]} != 1 )); then
  echo "Expected exactly one unpacked manylinux HCU wheel" >&2
  exit 1
fi
install -d -m 0755 "${finalize_wheels[0]}/torch.libs"
cp -a -- \
  "${prepared_lib_dir}/libmkl_def.so" \
  "${finalize_wheels[0]}/torch.libs/libmkl_def.so.2"
cp -a -- \
  "${prepared_lib_dir}/libmkl_vml_def.so" \
  "${finalize_wheels[0]}/torch.libs/libmkl_vml_def.so.2"
python3 -m wheel pack --dest-dir dist "${finalize_wheels[0]}"

shopt -s nullglob
repaired_wheels=(dist/*.whl)
shopt -u nullglob
if (( ${#repaired_wheels[@]} != 1 )); then
  echo "Expected exactly one finalized HCU wheel in dist, found ${#repaired_wheels[@]}" >&2
  printf "%s\n" "${repaired_wheels[@]}"
  exit 1
fi

auditwheel show "${repaired_wheels[0]}"
ls -lh dist/raw/*.whl "${repaired_wheels[0]}"
