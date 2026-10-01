#!/usr/bin/env bash
# Build two throwaway venvs: `old` (published crc-sdk, default 0.7.1) and `new`
# (this checkout, editable). Usage: tools/compat/setup.sh [DIR] [OLD_VERSION]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${1:-${TMPDIR:-/tmp}/crc-sdk-compat}"
OLD="${2:-0.7.1}"
EXTRAS="geometry,raster,netcdf,zarr"
mkdir -p "$DIR"
[ -d "$DIR/old" ] || uv venv -q --python 3.12 "$DIR/old"
[ -d "$DIR/new" ] || uv venv -q --python 3.12 "$DIR/new"
uv pip install -q --python "$DIR/old/bin/python" "crc-sdk[$EXTRAS]==$OLD"
uv pip install -q --python "$DIR/new/bin/python" -e "$ROOT[$EXTRAS]"
for v in old new; do
  "$DIR/$v/bin/python" -I -c "import importlib.metadata as m; print('$v', m.version('crc-sdk'), 'crc-framework', m.version('crc-framework'))"
done
echo "$DIR"
