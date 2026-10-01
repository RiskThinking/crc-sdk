"""Runs inside either venv; the same file exercises 0.7.x and 0.8.x.

  write-cdf   fit real distribution rows (hot_days | rflood | cyclone)
  write-efas  canonicalize a small real JRC EFAS window
  read        probe every read path against one canonical file
  compare     curve-column equality between two canonical files
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROBS = [0.5, 0.9, 0.98, 0.99, 0.999]
RP9 = np.array([2, 5, 10, 25, 50, 100, 250, 500, 1000])


def _digest(values) -> str:
    arr = np.round(np.nan_to_num(np.asarray(values, dtype=float), nan=-1.0), 6)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def write_cdf(a) -> dict:
    from crc_sdk.connectors import write_hazard_stream
    from crc_sdk.fitting import CDFColumnSchema, CDFCurveFitPolicy, fit_cdf_quantile_batches
    from crc_sdk.types import SourceProvenance

    table = pq.read_table(a.input)
    kwargs = {}
    columns = CDFColumnSchema()
    if a.kind == "hot_days":
        probabilities = np.linspace(0.0, 1.0, 11)  # real file has 1001; checked below
        probabilities = np.linspace(0.0, 1.0, len(table["cdf_quantiles"][0]))
        family, unit, lb = "gumbel_r", "days", None
        hazard = "hot_days"
        kwargs = dict(minimum_informative_value=1.0, minimum_informative_knots=4,
                      minimum_distinct_informative_values=2, fallback_families=("genextreme",),
                      parametric_failure_action="tabulated")
    elif a.kind == "rflood":
        probabilities = 1.0 - 1.0 / RP9
        rows = [np.maximum(np.asarray(v, float), 0.0) if v is not None and len(v) == 9 else None
                for v in table["perc_95"].to_pylist()]
        keep = [i for i, r in enumerate(rows) if r is not None]
        table = pa.table({
            "hex_id": table["hex_id"].take(keep), "index_name": ["rflood"] * len(keep),
            "year": table["year"].take(keep), "pathway": table["pathway"].take(keep),
            "cdf_quantiles": pa.array([rows[i].tolist() for i in keep], pa.list_(pa.float64())),
        })
        family, unit, lb = "genextreme", "m", None
        kwargs = dict(atom_policy="none", parametric_failure_action="tabulated")
    elif a.kind == "cyclone":
        n = 1001
        target = np.linspace(0, 1, n)
        out, ids = [], []
        for i, v in enumerate(table["full_distribution"].to_pylist()):
            if not v:
                continue
            s = np.sort(np.asarray(v, float))
            out.append(np.full(n, s[0]) if len(s) == 1 else np.interp(target, np.linspace(0, 1, len(s)), s))
            ids.append(i)
        table = pa.table({
            "hex_id": table["hex_id"].take(ids), "index_name": ["cyclone"] * len(ids),
            "year": pa.array([2025] * len(ids), pa.int32()), "pathway": ["pooled"] * len(ids),
            "cdf_quantiles": pa.array([o.tolist() for o in out], pa.list_(pa.float64())),
        })
        probabilities = target
        family, unit, lb = "gumbel_r", "m/s", None
        kwargs = dict(fallback_families=("genextreme",), parametric_failure_action="tabulated")
    else:
        raise SystemExit(f"unknown kind {a.kind}")
    table = table.set_column(table.schema.get_field_index("year"), "year", table["year"].cast(pa.int32()))
    params = dict(h3_resolution=5, family=family, value_unit=unit, value_semantics=a.kind,
                  producer="compat-harness", creation_version="compat",
                  source=SourceProvenance(provider="fixture", dataset=a.kind),
                  source_id="compat", on_fit_failure="skip", max_workers=2, **kwargs)
    if a.schema_version and "schema_version" in inspect.signature(CDFCurveFitPolicy).parameters:
        params["schema_version"] = a.schema_version
    result = fit_cdf_quantile_batches(table, probabilities.tolist(), CDFCurveFitPolicy(**params), columns=columns)
    write_hazard_stream(result.stream, a.out, max_workers=1, overwrite=True)
    s = result.summary
    return {"rows": s.source_rows, "canonical": s.canonical_rows, "skipped": s.skipped_rows}


def write_efas(a) -> dict:
    from crc_sdk.workflows import HazardDataset, JRCFloodPolicy

    plan = (HazardDataset.efas(version="latest").for_area((7.9, 50.0, 8.1, 50.1))
            .cache(a.cache, mode="reuse").source_periods("all")
            .canonicalize(policy=JRCFloodPolicy.curated(h3_resolution=8)))
    plan.materialize(a.out)
    return {"rows": pq.ParquetFile(a.out).metadata.num_rows}


def _probe(name, fn, out):
    try:
        out[name] = {"ok": True, "value": fn()}
    except Exception as error:  # noqa: BLE001 - the point is to record any failure
        text = "".join(traceback.format_exception_only(type(error), error)).strip()
        out[name] = {"ok": False, "error": text[:300]}


def read(a) -> dict:
    import duckdb
    from crc_sdk.connectors import read_hazard_dataset, read_hazard_metadata
    from crc_sdk.workflows import curve_quantiles, distribution_from_hazard_row

    out: dict = {}
    _probe("metadata", lambda: read_hazard_metadata(a.input).schema_version, out)
    _probe("read_hazard_dataset", lambda: read_hazard_dataset(a.input).num_rows, out)

    def raw_quantiles():
        table = pq.read_table(a.input)
        q = curve_quantiles(table, PROBS)
        return _digest(q)

    _probe("table_curve_quantiles", raw_quantiles, out)
    _probe("duckdb_columns", lambda: duckdb.sql(
        f"select count(*), count(distinct cell_index) from read_parquet('{a.input}')").fetchone(), out)

    def one_row():
        table = pq.read_table(a.input).slice(0, 1)
        return _digest(distribution_from_hazard_row(table).quantiles(PROBS))

    _probe("distribution_from_hazard_row", one_row, out)

    def local_provider():
        from crc_sdk.workflows import HazardDataset
        HazardDataset.local(a.input)
        return "constructed"

    _probe("HazardDataset.local", local_provider, out)
    raw = pq.read_schema(a.input).metadata or {}
    meta = json.loads(raw.get(b"crc.hazard.metadata", b"{}"))
    out["_file"] = {"schema_version": meta.get("schema_version"), "creation_version": meta.get("creation_version"),
                    "rows": pq.ParquetFile(a.input).metadata.num_rows}
    return out


def compare(a) -> dict:
    cols = ["hazard_name", "horizon", "pathway", "cell_index", "curve_kind", "curve_type", "curve_shape",
            "curve_location", "curve_scale", "curve_atom_probability", "curve_atom_location"]
    key = ["hazard_name", "horizon", "pathway", "cell_index"]
    x = pq.read_table(a.input, columns=cols).sort_by([(k, "ascending") for k in key])
    y = pq.read_table(a.other, columns=cols).sort_by([(k, "ascending") for k in key])
    if x.num_rows != y.num_rows:
        return {"identical": False, "rows": [x.num_rows, y.num_rows]}
    kind_diff = int(sum(p != q for p, q in zip(x["curve_kind"].to_pylist(), y["curve_kind"].to_pylist())))
    type_diff = int(sum(p != q for p, q in zip(x["curve_type"].to_pylist(), y["curve_type"].to_pylist())))
    worst = 0.0
    for c in cols[6:]:
        p = np.array(x[c].to_pylist(), dtype=float)
        q = np.array(y[c].to_pylist(), dtype=float)
        both = ~(np.isnan(p) & np.isnan(q))
        if both.any():
            d = np.abs(np.nan_to_num(p[both], nan=1e300) - np.nan_to_num(q[both], nan=-1e300))
            worst = max(worst, float(np.max(d)))
    return {"identical": kind_diff == 0 and type_diff == 0 and worst == 0.0, "rows": x.num_rows,
            "kind_mismatches": kind_diff, "type_mismatches": type_diff, "max_abs_param_diff": worst}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-cdf"); w.add_argument("--kind", required=True); w.add_argument("--input", required=True)
    w.add_argument("--out", required=True); w.add_argument("--schema-version")
    e = sub.add_parser("write-efas"); e.add_argument("--out", required=True); e.add_argument("--cache", required=True)
    r = sub.add_parser("read"); r.add_argument("--input", required=True)
    c = sub.add_parser("compare"); c.add_argument("--input", required=True); c.add_argument("--other", required=True)
    a = p.parse_args()
    print(json.dumps({"write-cdf": write_cdf, "write-efas": write_efas, "read": read, "compare": compare}[a.cmd](a)))


if __name__ == "__main__":
    main()
