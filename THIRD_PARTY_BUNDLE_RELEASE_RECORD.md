# Third-Party Bundle Release Record

## Reference Bundle

- Bundle name: `third_party_mayp_0602.zip`
- Source URL:
  `http://42.228.13.241:18000/thirdparty/pytorch/pytorch2.12/third_party_mayp_0602.zip`
- Size: `460014977` bytes
- Last modified: `2026-06-02 07:04:22 GMT`
- SHA-256:
  `61e5c3eacb5cc02e5f97336591f2f243296df1f59c1f043d9e90d87b22c084ce`

## Verification Date

- Verified on: `2026-07-08`

## Verification Result

The reference bundle is reachable and its size, timestamp, and SHA-256 are
known. However, the reference bundle is not yet sufficient to satisfy the
current repository-side open-source compliance materials.

Observed gaps relative to the current repository state:

1. The extracted bundle does not contain `third_party/licenses/`.
2. The extracted bundle does not contain `third_party/attributions/`.
3. The extracted bundle ships older top-level `third_party/README.md` and
   `third_party/LICENSES_BUNDLED.txt` content than the current repository.

## Refreshed Candidate Bundle

- Bundle name: `third_party_mayp_0602_compliance_refresh_20260708.zip`
- Size: `460027747` bytes
- SHA-256:
  `9bc67b244daac24ab6e85c4b52e1d77d67ccca48bc34bc5b9645fd3b6ddc556c`
- Status: local candidate generated from the reference bundle plus the current
  repository compliance materials.

Verified candidate contents:

1. `third_party/README.md`
2. `third_party/LICENSES_BUNDLED.txt`
3. `third_party/licenses/`
4. `third_party/attributions/`

The refreshed candidate is the current review target. It still needs an
official published URL or release handoff before public release is complete.

## Required Release Gate

Before public release, regenerate or refresh the standalone third-party bundle
so that the extracted `third_party` tree contains:

1. The current `third_party/README.md`
2. The current `third_party/LICENSES_BUNDLED.txt`
3. The full `third_party/licenses/` directory
4. The full `third_party/attributions/` directory

Do not treat the reference bundle as release-ready. Public release should use
the refreshed candidate bundle (or an equivalent published revision with the
same compliance materials), and that published revision must be hashed and
recorded here as a distinct bundle revision.
