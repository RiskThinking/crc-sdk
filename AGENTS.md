# Agent guide

crc-sdk is the Python data-access and workflow layer of Climate Risk Commons.
Numerical algorithms live in the `crc-framework` dependency
([crc-framework-rs](https://github.com/RiskThinking/crc-framework-rs)).

## Read first

- [docs/README.md](docs/README.md): one-line index of every page; open only the
  page you need.
- [docs/concepts.md](docs/concepts.md): plans, cache modes, canonical dataset
  contract and probability conventions. Most questions start here.
- Docstrings are the parameter reference (`crc_sdk/workflows/*.py`).

## Layout

`crc_sdk/workflows` (user-facing plans) → `providers` (discovery, licences) →
`connectors` (format readers, canonical I/O, DuckDB) → `fitting`, `impacts`,
`geometry`, `types`, `schema`. Tests mirror modules under `tests/`.

## Checks

```sh
uv sync --all-extras
uv run pytest -q
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

Tests are offline; never add tests that require the network without an opt-in
environment variable.

## Rules

- Do not break public APIs exported in `__all__`; add, deprecate, then remove.
- Builders and `explain()` must not do I/O.
- Write outputs atomically (temporary file, then `os.replace`).
- Record licence, attribution, retrieval time and checksums for every source.
- Docs, code and commit messages are in English.
