# Development

[Docs index](README.md) · [Project README](../README.md)

## Setup and checks

Development resolves `crc-framework` from the sibling checkout
`../crc-framework-rs`. Check out the matching framework source there before
installing the SDK environment. The `tool.uv.sources` entry is a development
source override only; built SDK wheels retain the declared
`crc-framework>=1.0.0,<2` dependency and do not contain a filesystem dependency.

```sh
uv sync --locked --all-extras
uv run python tools/api_contracts/check.py
uv run pytest
uv run mypy
uv run ruff check .
uv run ruff format --check .
```

Without uv: `python -m venv .venv && .venv/bin/pip install -e ".[zarr,netcdf,raster,test]"`.

Tests use offline source fixtures. Generated dataset and catalog resources are
checked together with the paired framework contracts:

```sh
uv run python tools/generate_dataset_contracts.py --check
uv run python tools/generate_evidence_contracts.py --check
uv run pytest -q tests/test_dataset_schema.py tests/test_declarations.py tests/test_catalog.py
uv run pytest -q tests/test_execution_scope.py tests/test_recipes.py tests/test_terminals.py
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

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests:

- lint on Python 3.12: `ruff check`, `ruff format --check` and `mypy`;
- tests on Python 3.11–3.14 with the locked dependencies and paired framework
  source (3.14 is
  non-blocking while native wheels catch up);
- paired editable and wheel installation checks;
- clean installs of the built SDK and framework wheels, plus a Windows job.

## Releases

1. Apply the package and schema versions from
   `crc_sdk/contracts/api/release.json` to `pyproject.toml`, generated contracts,
   lockfiles, and release checks. Superseded unshipped definitions are replaced
   in place; their existence is not a reason for an additional version bump.
   Run `uv lock` after changing dependencies.
2. Merge, then create a GitHub release with tag `v<version>` (for example
   `v1.0.0` or `v1.1.0a1`). Prerelease tags are marked automatically.
3. `release.yaml` reruns CI, checks the tag matches the package version, builds
   one universal wheel and an sdist, and requires
   `tools/api_contracts/check.py --runtime` against the installed wheel before
   publication. Declaration checks do not prove the implementation is complete.

## Contributor credits

The roster in `.all-contributorsrc` generates the README table:

```sh
npx --yes all-contributors-cli@6.26.1 add HANDLE code,review
npx --yes all-contributors-cli@6.26.1 generate
```

Edit the roster, not the generated table.
