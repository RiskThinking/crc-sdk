# Development

[Docs index](README.md) · [Project README](../README.md)

## Setup and checks

```sh
uv sync --all-extras
uv run pytest
uv run mypy
uv run ruff check .
uv run ruff format --check .
```

Without uv: `python -m venv .venv && .venv/bin/pip install -e ".[zarr,netcdf,raster,geometry,test]"`.

Tests are offline by default. Opt-in live checks:
`CRC_OPEN_LIVE=1 uv run pytest -q tests/test_crc_open_public.py`.

## Conventions

- Plans are immutable; builders and `explain()` must not do I/O.
- Optional dependencies are imported lazily and raise an `ImportError` naming
  the extra.
- Every adapter records licence, attribution, retrieval time and checksums in
  canonical metadata or its cache manifest.
- Numerical algorithms belong in [crc-framework](https://github.com/RiskThinking/crc-framework-rs);
  this package handles data access, orchestration and contracts.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests:

- lint (ruff, mypy) and tests on Python 3.11–3.14 with the locked
  dependencies (3.14 is non-blocking while native wheels catch up);
- tests against the minimum supported crc-framework;
- a clean install of the built wheel, plus a Windows job.

## Releases

1. Bump `version` in `pyproject.toml` (and the `crc-framework` range if
   needed); run `uv lock`.
2. Merge, then create a GitHub release with tag `v<version>` (for example
   `v1.0.0` or `v1.1.0a1`). Prerelease tags are marked automatically.
3. `release.yaml` reruns CI, checks the tag matches the package version, builds
   one universal wheel and an sdist, attaches them to the release and publishes
   to PyPI.

## Contributor credits

The roster in `.all-contributorsrc` generates the README table:

```sh
npx --yes all-contributors-cli@6.26.1 add HANDLE code,review
npx --yes all-contributors-cli@6.26.1 generate
```

Edit the roster, not the generated table.
