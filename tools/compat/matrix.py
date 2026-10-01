"""Drive worker.py across venvs and print the compatibility matrix.

  python tools/compat/matrix.py --venvs DIR --data DIR --work DIR [--json out.json]
Writers: old (published), new (local, schema 1.3), new12 (local, schema 1.2).
Readers: old, new. Datasets: real proprietary slices + one open JRC EFAS window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = str(Path(__file__).with_name("worker.py"))


def run(venv: Path, *args: str) -> dict:
    proc = subprocess.run([str(venv / "bin" / "python"), WORKER, *args], capture_output=True, text=True)
    if proc.returncode:
        return {"error": (proc.stderr.strip().splitlines() or ["failed"])[-1][:300]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venvs", required=True)
    ap.add_argument("--data", required=True, help="dir with hot_days.parquet rflood_dhaka.parquet cyclone_miami.parquet")
    ap.add_argument("--work", required=True)
    ap.add_argument("--json")
    ap.add_argument("--skip-open", action="store_true")
    a = ap.parse_args()
    venv = {"old": Path(a.venvs) / "old", "new": Path(a.venvs) / "new"}
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    datasets = {"hot_days": data / "hot_days.parquet", "rflood": data / "rflood_dhaka.parquet",
                "cyclone": data / "cyclone_miami.parquet"}
    writers = {"old": ("old", None), "new": ("new", None), "new12": ("new", "1.2")}
    files: dict[tuple[str, str], Path] = {}
    report: dict = {"writes": {}, "reads": {}, "equivalence": {}}
    for ds, src in datasets.items():
        for w, (v, sv) in writers.items():
            out = work / f"{ds}.{w}.parquet"
            args = ["write-cdf", "--kind", ds, "--input", str(src), "--out", str(out)]
            if sv:
                args += ["--schema-version", sv]
            res = run(venv[v], *args)
            report["writes"][f"{ds}/{w}"] = res
            if "error" not in res:
                files[(ds, w)] = out
    if not a.skip_open:
        for w, v in (("old", "old"), ("new", "new")):
            out = work / f"efas.{w}.parquet"
            res = run(venv[v], "write-efas", "--out", str(out), "--cache", str(work / f"efas-cache-{w}"))
            report["writes"][f"efas/{w}"] = res
            if "error" not in res:
                files[("efas", w)] = out
    for (ds, w), path in files.items():
        for r in ("old", "new"):
            report["reads"][f"{ds}/{w} -> {r}"] = run(venv[r], "read", "--input", str(path))
    for ds in [*datasets, "efas"]:
        for other in ("new", "new12"):
            if (ds, "old") in files and (ds, other) in files:
                report["equivalence"][f"{ds}: old vs {other}"] = run(
                    venv["new"], "compare", "--input", str(files[(ds, "old")]), "--other", str(files[(ds, other)]))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
    probes = ["metadata", "read_hazard_dataset", "table_curve_quantiles", "duckdb_columns",
              "distribution_from_hazard_row", "HazardDataset.local"]
    print("| file -> reader | file schema | " + " | ".join(probes) + " |")
    print("|---|---|" + "---|" * len(probes))
    for k, v in report["reads"].items():
        if "error" in v:
            print(f"| {k} | - | " + " | ".join(["ERR"] * len(probes)) + " |"); continue
        cells = ["ok" if v[p]["ok"] else "**FAIL**" for p in probes]
        print(f"| {k} | {v['_file']['schema_version']} | " + " | ".join(cells) + " |")
    print()
    for k, v in report["reads"].items():
        for p in probes:
            if isinstance(v, dict) and p in v and not v[p]["ok"]:
                print(f"- {k} / {p}: {v[p]['error']}")
    print()
    for k, v in report["equivalence"].items():
        print(f"- {k}: {v}")
    print()
    for k, v in report["writes"].items():
        print(f"- write {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
