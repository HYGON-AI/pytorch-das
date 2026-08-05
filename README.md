# PyTorch for Hygon

## Open Source Attribution

This repository is based on the official PyTorch project and contains
Hygon-maintained modifications based on the official PyTorch 2.12.0 release.

- Upstream project: PyTorch
- Upstream source: https://github.com/pytorch/pytorch/tree/v2.12.0
- Upstream documentation: https://docs.pytorch.org/docs/2.12/
- Upstream README preserved in this repository: [README_ORIGIN.md](README_ORIGIN.md)
- License: BSD-3-Clause

`README_ORIGIN.md` is preserved as an upstream reference copy for comparison and
review. It keeps upstream wording as-is and does not define the public release
messaging for this Hygon repository.

The original PyTorch copyright, license terms, and disclaimers are preserved in
[LICENSE](LICENSE). Third-party notices are preserved in [NOTICE](NOTICE) where
applicable.

Modified by Hygon Information Technology Co., Ltd.

## Modification Scope

Hygon modifications in this repository are intended to adapt PyTorch for Hygon
software and hardware environments. The scope includes Hygon backend integration,
build and packaging adjustments, operator and kernel adaptations, and related
documentation or metadata updates.

Unless explicitly stated in a specific source file or third-party notice, files
that originate from upstream PyTorch remain governed by the upstream PyTorch
license terms.

## Third-Party Sources

Third-party libraries are managed as Git submodules, consistent with upstream
PyTorch. Initialize them before building:

```bash
git submodule sync
git submodule update --init --recursive
tools/third_party/apply_patches.sh
```

The licenses for those libraries are provided in their respective repositories.

## Build and Installation

Build and installation should follow the release notes or packaging instructions
for the corresponding Hygon PyTorch distribution. For general PyTorch build
concepts, refer to the upstream PyTorch documentation and the preserved upstream
README.

After installation, verify the package with:

```bash
python -c "import torch; print(torch.__version__)"
```

## References

- [Official PyTorch repository](https://github.com/pytorch/pytorch)
- [Official PyTorch documentation](https://pytorch.org/docs/stable/)
- [Preserved upstream README](README_ORIGIN.md)
