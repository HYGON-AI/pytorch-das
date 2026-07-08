# PyTorch for Hygon

## Open Source Attribution

This repository is based on the official PyTorch project and contains
Hygon-maintained modifications for the PyTorch 2.12 development line.

- Upstream project: PyTorch
- Upstream source: https://github.com/pytorch/pytorch
- Upstream documentation: https://pytorch.org/docs/stable/
- Upstream README preserved in this repository: [README_ORIGIN.md](README_ORIGIN.md)
- License: BSD-3-Clause

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

This release uses a split source-release model:

- the PyTorch repository provides the main source tree, repository notices, and
  release-facing documentation
- a matched standalone third-party source bundle provides the `third_party/`
  source tree used for build and redistribution

Before build, download the matched standalone third-party source bundle for this
repository release. See [THIRD_PARTY_BUNDLE.md](THIRD_PARTY_BUNDLE.md) for the
current bundle identifier, extraction path, release responsibilities, and
compliance notes.

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
