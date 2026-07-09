This directory records the third-party dependency entry points, submodule
references, and repository-side compliance materials used by the PyTorch tree.
It should not be interpreted as the complete third-party source delivery for
Hygon PyTorch source release.

For Hygon PyTorch source releases, the actual build may replace this directory
with a matched standalone third-party source bundle before compilation. In
practice, the build workflow may remove the repository copy of `third_party/`
and then extract the matched standalone bundle into the repository root. See
../THIRD_PARTY_BUNDLE.md for the current bundle information, extraction path,
and compliance notes.

Accordingly, this repository copy of `third_party/` mainly serves three
purposes:

- preserve dependency entry points and submodule references
- preserve repository-side compliance reference materials
- define the compliance baseline that the standalone third-party bundle is
  expected to carry forward

Relevant submodule-backed components currently referenced by this repository
include:

- third_party/cutlass (core, excluding examples/python/CuTeDSL)
  Upstream: https://github.com/NVIDIA/cutlass
  License: BSD-3-Clause
  Preserved license text: third_party/licenses/cutlass.LICENSE
  Usage: CUTLASS-backed attention integrations and other CUTLASS core usage.

- third_party/cutlass/examples/python/CuTeDSL
  Upstream: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
  License: NVIDIA EULA
  Preserved license text: third_party/licenses/cutlass.CuTeDSL.EULA.txt
  Usage: Mirrored into PyTorch Inductor vendored CuTeDSL kernels during build.
  Distribution note: Separately governed component; do not treat it as part of
  the CUTLASS BSD-3-Clause core license scope.
  Scope note: This EULA boundary applies to the CuTeDSL subtree and the
  build-mirrored grouped_gemm.py path, not to every file under
  torch/_inductor/kernel/vendored_templates/cutedsl.

- third_party/composable_kernel
  Upstream: https://github.com/ROCm/composable_kernel
  License: MIT
  Preserved license text: third_party/licenses/composable_kernel.LICENSE
  Usage: HIP flash-attention CK backend integration.

- third_party/flash-attention
  Upstream: https://github.com/Dao-AILab/flash-attention
  License: BSD-3-Clause
  Preserved license text: third_party/licenses/flash-attention.LICENSE
  Usage: HIP flash-attention integration under
  aten/src/ATen/native/transformers/hip/flash_attn.

- third_party/aiter
  Upstream: https://github.com/ROCm/aiter
  License: MIT
  Preserved license text: third_party/licenses/aiter.LICENSE
  Usage: AITER-backed HIP flash-attention integration.

The submodule references remain declared in .gitmodules. Local preserved copies
of the relevant upstream license texts are stored under third_party/licenses/ so
that the repository retains licensing information even when submodules are not
initialized. When a standalone third-party source bundle is used, the extracted
bundle becomes the complete third-party source delivery for build and
redistribution, and it must preserve the applicable upstream license texts,
copyright notices, NOTICE files, and Hygon modification records.

The extracted standalone bundle must also carry the current top-level
third_party compliance materials from this repository, including:

- third_party/README.md
- third_party/LICENSES_BUNDLED.txt
- third_party/licenses/
- third_party/attributions/

If the repository copy of these compliance materials is updated, the matched
standalone third-party bundle is expected to refresh the corresponding files as
part of the next published bundle revision.
