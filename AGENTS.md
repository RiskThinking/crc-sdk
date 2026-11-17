# Agent guide

crc-sdk is the Python data-access and workflow layer of Climate Risk Commons.
Numerical algorithms live in the `crc-framework` dependency
([crc-framework-rs](https://github.com/RiskThinking/crc-framework-rs)).

## Read first

- [docs/README.md](docs/README.md): one-line index of every page; open only the
  page you need.
- [docs/api-reference.md](docs/api-reference.md): canonical imports, interface
  ownership, and contract checks.
- `crc_sdk/contracts/api/public-api.json`: authoritative constructors, methods,
  properties, enums, and removal inventory.
- `crc_sdk/contracts/api/release.json`: package and format version ledger.

## Layout

`_declarations` and `_catalog` define science and exact identity. `_dataset_schema`
and `_schema` validate artifact metadata, canonical rows, Arrow and Parquet.
`_providers`, `_connectors`, and `_geometry` contain private source-decoding and
spatial algorithms. Canonical public imports live at `crc_sdk`.

## Checks

```sh
uv sync --locked --all-extras
uv run python tools/api_contracts/check.py
uv run pytest -q
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

Tests are offline; never add tests that require the network without an opt-in
environment variable.

## Rules

- Implement the declared Python interface directly. Superseded exports,
  historical schemas, compatibility adapters, and alternate names are removed;
  do not add deprecation paths or tests requiring their retention.
- Numerical types, measurements, and fitting specifications belong to
  `crc_framework`; do not re-export them from the SDK or copy their policies.
- Closed library choices use typed enums. Custom identities and data column
  names remain open exact strings, without implicit normalization or taxonomy
  registration.
- Input constructors accept the required science directly. Ordinary data access
  does not require invented scenarios, periods, or custom schema authoring.
- Builders and `explain()` must not do I/O.
- Result terminals execute their complete upstream graph automatically. Resolve
  each source and distinct modelling node once per execution.
- Write outputs atomically and preserve existing destinations on failure. Local
  writes default to no overwrite.
- Record licence, attribution, retrieval time and checksums for every source.
- Keep diagnostics and missing-data counts through every terminal. Collected
  evidence and immutable reports remain usable after execution closes.
- Version once against the released baseline. Replace unshipped definitions in
  place; do not raise another major version because a branch draft used it.
- An installable release must pass `tools/api_contracts/check.py --runtime`
  against the built wheel. Declaration validation alone is not runtime evidence.
- Do not add review documents, implementation-phase narration, or references to
  external planning files to product code and documentation.
- Docs, code and commit messages are in English.
