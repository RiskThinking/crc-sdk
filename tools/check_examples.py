"""Run the Python examples in the SDK guides with isolated working directories."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

from crc_framework import DepthAboveGround, LengthUnit, Tail

from crc_sdk import Dataset, Period, Probability, Variable
from crc_sdk._crc_open import release_extent

PYTHON_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
LIVE_EXAMPLE = "<!-- live-example -->"


def _extent(path: Path) -> dict[str, float] | None:
    extent = release_extent(path)
    return None if extent is None else dataclasses.asdict(extent)


def write_example_release() -> None:
    """Supply a local publisher index for the CRC Open discovery example."""
    dataset = Dataset.from_quantiles(
        {
            "support_id": ["harbour", "harbour"],
            "longitude": [24.0, 24.0],
            "latitude": [60.0, 60.0],
            "probability": [0.9, 0.99],
            "value": [0.6, 1.2],
        },
        variable=Variable(
            "company:flood-depth",
            measurement=DepthAboveGround(unit=LengthUnit.METRE),
        ),
        period=Period.window(2041, 2070),
        probability=Probability.annual(tail=Tail.UPPER),
    ).write("published-depth.parquet")
    assert dataset.path is not None and dataset.metadata is not None
    digest = hashlib.sha256(dataset.path.read_bytes()).hexdigest()
    document = {
        "schema_version": "1.1",
        "id": "example-release",
        "datasets": [
            {
                "id": "published-depth",
                "uri": dataset.path.name,
                "sha256": f"sha256:{digest}",
                "metadata": json.loads(dataset.metadata.to_json()),
                "extent": _extent(dataset.path),
            }
        ],
    }
    Path("release-index.json").write_text(json.dumps(document), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pages = [root / "README.md", *sorted(root.glob("docs/*.md"))]
    for page in pages:
        content = page.read_text(encoding="utf-8")
        matches = list(PYTHON_FENCE.finditer(content))
        if not matches:
            continue
        with TemporaryDirectory() as directory, chdir(directory):
            if page == root / "docs/sources.md":
                write_example_release()
            namespace = {"__name__": "__main__"}
            checked = 0
            live = 0
            for match in matches:
                line = content.count("\n", 0, match.start(1))
                code = compile("\n" * line + match[1], str(page), "exec")
                if content[: match.start()].rstrip().endswith(LIVE_EXAMPLE):
                    live += 1
                    continue
                exec(code, namespace)
                checked += 1
        print(f"{page.relative_to(root)}: {checked} examples, {live} live snippets")


if __name__ == "__main__":
    main()
