# DuckDB versions

[Docs index](README.md) · [Sources and native evidence](sources.md)

The SDK supports DuckDB 1.4 through 2.x. Tests run against both major versions.
The SDK does not check which version is installed, so you get the speed of the
version you install.

You only see DuckDB in two places:

- `Assets.sql(...)` runs your SELECT query.
- `Runtime(connection=...)` lends the SDK your own connection.

DuckDB 2.0 changes some SQL and connection behaviour. It only affects your own
queries and connections. SDK results are the same on both versions.

| Your SQL or connection | DuckDB 2.x | To keep 1.x behaviour |
| --- | --- | --- |
| Arrow lambdas, `x -> x + 1` | Rejected. Write `lambda x: x + 1`, which 1.x also accepts | `SET lambda_syntax = 'ENABLE_SINGLE_ARROW'` |
| `a // 0` and `a % 0` | Raise an error instead of returning NULL | Wrap the expression in `TRY(...)`, or `SET error_on_division_by_zero = false` |
| A connection passed as `Runtime(connection=...)` | Must not hold an unfinished `execute()` result. Read results to the end, for example with `fetchall()`, before running the SDK | Not needed; reading to the end works on both versions |
| Database files you write | DuckDB 1.x cannot open them | Connect with `config={"storage_compatibility_version": "v1.4.0"}` |

The SDK does not write DuckDB database files. It only uses DuckDB's temporary
spill directory, which it removes when execution ends.

## Stay on DuckDB 1.x

New installs get DuckDB 2.x. To keep 1.x, add an upper bound next to the SDK:

```sh
python -m pip install --upgrade crc-sdk "duckdb<2"
```

In a project, add `duckdb>=1.4,<2` to your own dependencies. The SDK behaves the
same on either version.
