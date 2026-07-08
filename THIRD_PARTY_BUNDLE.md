# Third-Party Source Bundle

This PyTorch source package uses a separately published third-party source
bundle for build and redistribution.

The upstream PyTorch repository references third-party components through git
submodules and related build scripts. For Hygon PyTorch releases, the
`third_party` source tree may be provided as a standalone bundle because some
components require local fixes, platform adaptation, or controlled downloads.

## Current Bundle

- Bundle name: `third_party_mayp_0602.zip`
- Bundle URL:
  `http://42.228.13.241:18000/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip`
- Bundle size: `460014977` bytes
- Last modified: `2026-06-02 07:04:22 GMT`
- Extract path: `third_party/`

The exact third-party bundle used for an official source release must be
recorded in the release record, together with a stable checksum such as
SHA-256. If a newer third-party bundle is released for bug fixes or performance
optimizations, it must be published and recorded as a distinct bundle revision.

## Usage

Before building this PyTorch source package, extract the matched third-party
bundle into the PyTorch source tree.

```bash
rm -rf third_party
unzip third_party_mayp_0602.zip
```

If the archive does not contain a top-level `third_party/` directory, extract it
with the correct target path instead:

```bash
mkdir -p third_party
unzip third_party_mayp_0602.zip -d third_party
```

## Compliance

The standalone third-party bundle is responsible for preserving each
third-party component's original license text, copyright notice, NOTICE file,
and Hygon modification records where applicable.

For third-party license details and Hygon modifications, see the inventory,
license summary, and modification records included in the extracted
`third_party` bundle.
