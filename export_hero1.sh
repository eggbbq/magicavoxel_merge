#!/usr/bin/env bash
set -euo pipefail

# Read hero2.vox beside this script; write all exports to the current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$PWD"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f "$SCRIPT_DIR/hero2.vox" ]]; then
  echo "Input file does not exist: $SCRIPT_DIR/hero2.vox" >&2
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export BTP_VOX_REUSE_SUBRECT_LIMIT="${BTP_VOX_REUSE_SUBRECT_LIMIT:-0}"

"$PYTHON_BIN" -m btp_vox.cli \
  --input "$SCRIPT_DIR/hero2.vox" \
  --output "$OUTPUT_DIR/hero2.gltf" \
  --format gltf \
  --tex-out "$OUTPUT_DIR/hero2.png" \
  --uv-out "$OUTPUT_DIR/hero2_uv.json" \
  --uv2 \
  --character-apart \
  --handedness right \
  --scale 0.02 \
  --pivot bottom_center \
  --tex-pot \
  --tex-fmt rgb \
  --tex-layout global \
  --tex-compress-solid-quads \
  --tex-reuse-subrects \
  --face-alias-uv-remap

printf 'Exported hero2.gltf, hero2.bin, hero2.png, hero2_uv.json to %s\n' "$OUTPUT_DIR"
