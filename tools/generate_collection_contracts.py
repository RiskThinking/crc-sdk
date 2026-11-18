"""Generate the collection JSON Schema and structural conformance cases."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from crc_sdk._collection_schema import (
    CollectionMember,
    CollectionMetadata,
    CollectionResource,
    CollectionSelection,
    collection_metadata_schema,
)
from crc_sdk._dataset_schema import (
    DatasetMetadata,
    ExecutionCounts,
    ExecutionReport,
    ReleaseReference,
    SourceIdentity,
    SpatialContract,
    SupportKind,
)
from crc_sdk._declarations import DataSpec, Period, SeriesSpec, Variable
from crc_sdk._inspection import SeriesReference

ROOT = Path(__file__).resolve().parents[1] / "crc_sdk" / "contracts"


def collection_fixtures() -> dict[str, object]:
    period = Period.window(1991, 2020)
    variable = Variable("company:depth")
    source = SourceIdentity(
        provider="project",
        product="depth",
        model="model-a",
        member="r1",
        release=ReleaseReference("first", "sha256:" + "c" * 64),
    )
    report = ExecutionReport(
        True,
        True,
        counts=ExecutionCounts(requested=1, produced=1),
        source_identities=(source,),
    )
    dataset = DatasetMetadata(
        spec=DataSpec([variable], series=[SeriesSpec(variable, period=period)]),
        spatial=SpatialContract(supports=(SupportKind.POINT,)),
        report=report,
        source=source,
    )
    selected = CollectionMetadata(
        members=(CollectionMember(id="company baseline", resource_ids=("r1",)),),
        resources=(
            CollectionResource(
                id="r1",
                metadata=dataset,
                path=f"resources/{'a' * 64}.parquet",
                sha256="sha256:" + "b" * 64,
            ),
        ),
        selection=CollectionSelection(
            members=("company baseline",),
            variables=(variable,),
            periods=(period,),
            series=(
                SeriesReference(
                    variable,
                    period=period,
                    release=source.release,
                    model=source.model,
                    member=source.member,
                ),
            ),
        ),
        report=report,
    )
    empty = CollectionMetadata(
        members=(),
        resources=(),
        selection=CollectionSelection(members=(), variables=()),
        report=ExecutionReport(True, True),
    )
    document = json.loads(selected.to_json())
    absent_resource = deepcopy(document)
    absent_resource["members"][0]["resource_ids"] = ["unknown"]
    invalid_version = deepcopy(document)
    invalid_version["schema_version"] = "0.9"
    dangling_digest = deepcopy(document)
    del dangling_digest["resources"][0]["path"]
    external_path = deepcopy(document)
    external_path["resources"][0]["path"] = "../unowned.parquet"
    unknown_record = deepcopy(document)
    unknown_record["workflow"] = "removed"
    wrong_period = deepcopy(document)
    wrong_period["selection"]["periods"][0]["id"] = "period:" + "0" * 64
    wrong_model = deepcopy(document)
    wrong_model["selection"]["series"][0]["model"] = "model-b"
    return {
        "accepted": [
            {"name": "typed-selection", "document": document},
            {"name": "empty-selection", "document": json.loads(empty.to_json())},
        ],
        "rejected": [
            {"name": name, "structural": structural, "document": record}
            for name, structural, record in (
                ("unregistered-resource", False, absent_resource),
                ("unsupported-version", True, invalid_version),
                ("digest-without-path", True, dangling_digest),
                ("external-resource", True, external_path),
                ("unknown-record", True, unknown_record),
                ("wrong-period-id", False, wrong_period),
                ("wrong-source-model", False, wrong_model),
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, value in (
        (ROOT / "schemas" / "collection-1.0.schema.json", collection_metadata_schema()),
        (ROOT / "fixtures" / "collection-1.0.json", collection_fixtures()),
    ):
        content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"stale generated collection contract: {path}")
        else:
            path.write_bytes(content)
    print("Collection schema and fixture verification passed")


if __name__ == "__main__":
    main()
