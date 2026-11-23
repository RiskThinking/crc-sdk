# Development

[Docs index](README.md) · [Project README](../README.md)

## Setup and checks

The lockfile pins the published `crc-framework` release recorded in
`crc_sdk/contracts/api/release.json`, so no framework checkout or Rust
toolchain is needed.

```sh
uv sync --locked --all-extras
uv run python tools/api_contracts/check.py
uv run pytest
uv run python -I tools/check_examples.py
uv run mypy
uv run ruff check .
uv run ruff format --check .
```

To install without uv, create a virtual environment and run
`python -m pip install --pre -e ".[zarr,netcdf,raster,test]"`; `--pre` is
needed while the framework requirement is a prerelease.

To work against unreleased framework changes, check out
[crc-framework-rs](https://github.com/RiskThinking/crc-framework-rs) next to
this repository and replace the locked release in the environment:

```sh
uv sync --locked --all-extras
uv pip install -e ../crc-framework-rs
```

The next `uv sync` restores the locked release. The installed-contract check
(`tools/api_contracts/check.py --runtime`) reports a framework version that
differs from the ledger.

The test suite uses offline source fixtures. The example check runs the Python
examples from the README and guides, using a fresh directory for each guide.
It supplies a local release index to the CRC Open example and syntax-checks
the explicitly marked live provider snippets.

Check the generated artifact definitions. The last command needs a
framework checkout at the release tag in the ledger.

```sh
uv run python tools/generate_dataset_contracts.py --check
uv run python tools/generate_evidence_contracts.py --check
uv run python tools/generate_collection_contracts.py --check
uv run python tools/generate_evaluation_contracts.py --check
uv run python ../crc-framework-rs/tools/sync_data_contracts.py --sdk . --check
```

## Conventions

- Recipes are immutable; builders and `explain()` must not do I/O.
- Optional dependencies are imported lazily and raise an `ImportError` naming
  the extra.
- Every adapter records licence, attribution, retrieval time and checksums in
  canonical metadata or its cache manifest.
- Numerical algorithms belong in [crc-framework](https://github.com/RiskThinking/crc-framework-rs);
  this package handles data access, orchestration and contracts.
- The [Python interface contract](api-reference.md) is authoritative. Remove
  superseded interfaces rather than retaining aliases, deprecated wrappers, or
  historical parser paths. Framework types have their canonical imports in
  `crc_framework`.
- Closed library choices use enums; user identifiers and data names remain
  open. Ordinary workflows require no preparation gate or invented metadata.

## CI

CI runs contracts, lint, and typing checks. It tests the SDK with the locked
framework release on Linux with Python 3.11–3.14 and on Windows.
CI verifies imports and contracts in a fresh base wheel installation, then
runs the offline suite and guide examples against the installed wheels with
locked test dependencies. A separate Linux Python 3.12 job checks the minimum
supported PyArrow version.

CI checks out the framework at the release tag in the ledger only for its
contract tools. To test against unreleased framework code, dispatch the
workflow with a `framework_ref`; CI then builds that framework revision from
source and uses it instead of the locked release.

## Releases

1. Apply the package and schema versions from
   `crc_sdk/contracts/api/release.json` to `pyproject.toml`, generated contracts,
   lockfiles, and release checks.
   Run `uv lock` after changing dependencies.
2. Publish the framework version recorded in the ledger. Merge the SDK and
   push the SDK package tag, which is `v` plus the `pyproject.toml` version,
   such as `v1.0.0a1` or `v1.0.0`.
3. The SDK release workflow checks that the tag matches the SDK package
   version and runs CI with the locked framework release. It
   builds the wheel and source distribution, installs the wheel with the exact
   published framework version, and verifies the installed contracts, tests,
   and examples against local fixtures. The release publishes the verified
   artifacts to GitHub and PyPI.

The data publisher verifies CRC Open shards and the release index before
updating the publication pointer. SDK release tests exercise the reader with a
local Dataset 2.0 release. [CRC Open discovery](sources.md#discover-a-crc-open-release)

## Contributor credits

The roster in `.all-contributorsrc` generates the README table:

```sh
npx --yes all-contributors-cli@6.26.1 add HANDLE code,review
npx --yes all-contributors-cli@6.26.1 generate
```

Edit the roster, not the generated table.
