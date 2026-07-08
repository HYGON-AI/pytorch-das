# Third-Party Source Bundle

This PyTorch source package uses a separately published third-party source
bundle for build and redistribution.

The upstream PyTorch repository references third-party components through git
submodules and related build scripts. For Hygon PyTorch releases, the
`third_party` source tree may be provided as a standalone bundle because some
components require local fixes, platform adaptation, or controlled downloads.

## Reference Bundle

- Bundle name: `third_party_mayp_0602.zip`
- Bundle URL:
  `http://42.228.13.241:18000/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip`
- Bundle size: `460014977` bytes
- Last modified: `2026-06-02 07:04:22 GMT`
- SHA-256:
  `61e5c3eacb5cc02e5f97336591f2f243296df1f59c1f043d9e90d87b22c084ce`
- Extract path: `third_party/`

The exact third-party bundle used for an official source release must be
recorded in the release record, together with a stable checksum such as
SHA-256. If a newer third-party bundle is released for bug fixes or performance
optimizations, it must be published and recorded as a distinct bundle revision.

## Refreshed Candidate Bundle

- Candidate name: `third_party_mayp_0602_compliance_refresh_20260708.zip`
- Candidate size: `460027747` bytes
- Candidate SHA-256:
  `9bc67b244daac24ab6e85c4b52e1d77d67ccca48bc34bc5b9645fd3b6ddc556c`
- Status: Local refreshed candidate generated during the 2026-07-08 compliance
  pass by overlaying the current repository compliance materials onto the
  reference bundle.

The refreshed candidate must be published as a distinct bundle revision before
an official public release can use it.

## Release Record

See [THIRD_PARTY_BUNDLE_RELEASE_RECORD.md](THIRD_PARTY_BUNDLE_RELEASE_RECORD.md)
for the current verification record, bundle status, and publish gates.

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

The `third_party_mayp_0602.zip` bundle referenced above is a base snapshot. The
official release bundle must also include the current repository copies of:

- `third_party/README.md`
- `third_party/LICENSES_BUNDLED.txt`
- `third_party/licenses/`
- `third_party/attributions/`

Do not publish the reference bundle directly unless those current compliance
materials are present in the extracted `third_party` tree.
