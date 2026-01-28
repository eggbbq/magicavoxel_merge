#!/usr/bin/env bash
set -euo pipefail

# Batch convert all .vox files in a directory to .glb using magicavoxel_merge.
#
# Edit IN_DIR / OUT_DIR / JOBS below, then run:
#   ./batch_convert.sh
#
# Notes:
# - Uses xargs parallelism (jobs defaults to number of CPU cores).
# - Output filenames follow input stem: <stem>.glb
# - If you want external textures, set TEXTURE_OUT=1 (writes <stem>.png alongside .glb)

IN_DIR="/Users/graylian/workspace/VoxPLC"
OUT_DIR="/Users/graylian/workspace/project_sh/pfl_cc/assets/assets/4_models"
# Max parallel jobs. Leave empty to auto-detect CPU cores.
JOBS="5"

if [[ ! -d "$IN_DIR" ]]; then
  echo "Input dir does not exist: $IN_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

if [[ -z "$JOBS" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
  else
    JOBS="4"
  fi
fi

# Export settings (match test_convert.sh defaults)
MODE="atlas" # atlas, palette
SCALE="0.02"
ATLAS_PAD="0" # atlas padding in texels
ATLAS_INSET="0" # atlas inset in texels
ATLAS_STYLE="baked" # baked, solid
ATLAS_TEXEL_SCALE="1" # texture texel scale
ATLAS_LAYOUT="by-model" # by-model, by-palette
CENTER_FLAG="--center" # center the model
HANDEDNESS="right" # right, left
AXIS="y_up" # y_up, z_up
AVG_NORMALS_ATTR="none" # none, color, tangent

# Set TEXTURE_OUT=1 to export external png beside the glb
TEXTURE_OUT="${TEXTURE_OUT:-0}"

if [[ -z "${HANDEDNESS:-}" ]]; then
  HANDEDNESS="right"
fi

if [[ -z "${AXIS:-}" ]]; then
  AXIS="y_up"
fi

export OUT_DIR MODE SCALE ATLAS_PAD ATLAS_INSET ATLAS_STYLE ATLAS_TEXEL_SCALE ATLAS_LAYOUT CENTER_FLAG HANDEDNESS AXIS AVG_NORMALS_ATTR TEXTURE_OUT

convert_one() {
  local in_vox="$1"
  local base
  base="$(basename "$in_vox")"
  base="${base%.vox}"

  local out_glb="$OUT_DIR/${base}.glb"

  if [[ "$TEXTURE_OUT" == "1" ]]; then
    local out_png="$OUT_DIR/${base}.png"
    python -m magicavoxel_merge \
      "$in_vox" "$out_glb" \
      --axis "$AXIS" \
      --mode "$MODE" \
      --scale "$SCALE" \
      --atlas-pad "$ATLAS_PAD" \
      --atlas-inset "$ATLAS_INSET" \
      --atlas-style "$ATLAS_STYLE" \
      --atlas-texel-scale "$ATLAS_TEXEL_SCALE" \
      --atlas-layout "$ATLAS_LAYOUT" \
      --handedness "$HANDEDNESS" \
      --avg-normals-attr "$AVG_NORMALS_ATTR" \
      $CENTER_FLAG \
      --texture-out "$out_png" \
      --weld \
      --cull-mv-faces top,bottom \
      --merge-strategy maxrect
  else
    python -m magicavoxel_merge \
      "$in_vox" "$out_glb" \
      --axis "$AXIS" \
      --mode "$MODE" \
      --scale "$SCALE" \
      --atlas-pad "$ATLAS_PAD" \
      --atlas-inset "$ATLAS_INSET" \
      --atlas-style "$ATLAS_STYLE" \
      --atlas-texel-scale "$ATLAS_TEXEL_SCALE" \
      --atlas-layout "$ATLAS_LAYOUT" \
      --handedness "$HANDEDNESS" \
      --avg-normals-attr "$AVG_NORMALS_ATTR" \
      $CENTER_FLAG \
      --weld \
      --cull-mv-faces top,bottom \
      --merge-strategy maxrect
  fi
}

export -f convert_one

# Use NUL delimiters to handle spaces in filenames.
VOX_COUNT="$(find "$IN_DIR" -type f -iname "*.vox" | wc -l | tr -d ' ')"
if [[ "$VOX_COUNT" == "0" ]]; then
  echo "No .vox files found under: $IN_DIR" >&2
  exit 0
fi

echo "Converting $VOX_COUNT .vox files from '$IN_DIR' -> '$OUT_DIR' (jobs=$JOBS, mode=$MODE, atlas_style=$ATLAS_STYLE)"

find "$IN_DIR" -type f \( -iname "*.vox" \) -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -lc 'convert_one "$0"' \
  || exit $?

# Optional listing
if [[ "$TEXTURE_OUT" == "1" ]]; then
  ls -lh "$OUT_DIR"/*.glb "$OUT_DIR"/*.png 2>/dev/null || true
else
  ls -lh "$OUT_DIR"/*.glb 2>/dev/null || true
fi
