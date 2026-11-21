"""Generate the insured annual loss artifact JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crc_sdk._loss_artifact import ANNUAL_LOSS_VERSION
from crc_sdk._schema.loss_contract import annual_loss_json_schema

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = (
        ROOT
        / "crc_sdk/contracts/schemas"
        / f"annual-loss-{ANNUAL_LOSS_VERSION}.schema.json"
    )
    content = (
        json.dumps(annual_loss_json_schema(), indent=2, allow_nan=False) + "\n"
    ).encode()
    if args.check:
        if not path.exists() or path.read_bytes() != content:
            raise SystemExit(f"stale generated annual loss contract: {path}")
    else:
        path.write_bytes(content)
    print("Annual loss schema verification passed")


if __name__ == "__main__":
    main()
