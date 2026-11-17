"""Generate the canonical evidence metadata JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crc_sdk._evidence import evidence_metadata_schema

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crc_sdk/contracts/schemas/evidence-1.0.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = (
        json.dumps(evidence_metadata_schema(), indent=2, allow_nan=False) + "\n"
    ).encode()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != content:
            raise SystemExit(f"stale generated evidence metadata schema: {OUTPUT}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(content)
    print(
        "Evidence metadata schema verified"
        if args.check
        else "Evidence metadata schema generated"
    )


if __name__ == "__main__":
    main()
