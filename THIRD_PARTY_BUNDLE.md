# Third-Party Source Bundle

This Hygon PyTorch release uses a split source-release model:

- the repository release provides the main PyTorch source tree, repository
  notices, and release-facing documentation
- a separately published standalone bundle provides the `third_party/` source
  tree used for build and redistribution

The upstream PyTorch repository references many third-party components through
git submodules and related build scripts. For Hygon PyTorch releases, the
`third_party` tree may be published separately because some components require
local fixes, platform adaptation, controlled downloads, or release-specific
source refresh.

## Repo-Side Responsibilities

The repository release is responsible for:

- identifying the matched standalone bundle revision for this repository release
- publishing the bundle name, URL, checksum, and extraction path once the
  bundle is officially released
- preserving repository-side notices and attribution records for source files
  that live outside `third_party/`

The repository release does not itself guarantee that the reference bundle or a
local candidate bundle is release-ready. That decision depends on the published
bundle contents and final verification status recorded below.

## Bundle-Side Responsibilities

The standalone `third_party` bundle owner is responsible for publishing a
bundle revision whose extracted `third_party/` tree matches the source bundle
expected by this repository release.

At minimum, the published bundle must include:

- `third_party/README.md`
- `third_party/LICENSES_BUNDLED.txt`
- `third_party/licenses/`
- `third_party/attributions/`
- the actual third-party source trees required for build, including any
  modified third-party libraries that are distributed in source form

If the bundle contents change, the bundle owner must publish a distinct bundle
revision with a new bundle name or version identifier and a new checksum.

## Current Bundle Status

### Reference Bundle

- Bundle name: `third_party_mayp_0602.zip`
- Bundle URL:
  `http://42.228.13.241:18000/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip`
- Bundle size: `460014977` bytes
- Last modified: `2026-06-02 07:04:22 GMT`
- SHA-256:
  `61e5c3eacb5cc02e5f97336591f2f243296df1f59c1f043d9e90d87b22c084ce`
- Extract path: `third_party/`

This bundle is a known base snapshot, not the final release-ready bundle for
the current repository-side compliance state.

### Current Review Candidate

- Candidate name: `third_party_mayp_0602_compliance_refresh_20260708.zip`
- Candidate size: `460027747` bytes
- Candidate SHA-256:
  `9bc67b244daac24ab6e85c4b52e1d77d67ccca48bc34bc5b9645fd3b6ddc556c`
- Status: locally generated review candidate assembled from the reference
  bundle plus the current repository-side compliance materials

This candidate describes the expected content shape for the final standalone
bundle, but it is not yet a published public bundle revision.

## Usage

Before building this PyTorch source package, extract the matched standalone
bundle into the repository root so that the bundle provides the `third_party/`
directory expected by build scripts.

```bash
rm -rf third_party
unzip <matched-standalone-bundle>.zip
```

If the archive does not contain a top-level `third_party/` directory, extract
it with the correct target path instead:

```bash
mkdir -p third_party
unzip <matched-standalone-bundle>.zip -d third_party
```

## Release Record

See [THIRD_PARTY_BUNDLE_RELEASE_RECORD.md](THIRD_PARTY_BUNDLE_RELEASE_RECORD.md)
for the current verification record, reference bundle gaps, bundle-owner action
items, and final publish gate.
