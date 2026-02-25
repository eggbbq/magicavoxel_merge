#!/usr/bin/env bash
set -euo pipefail

# Batch convert all .vox files inside IN_DIR into GLB + atlas PNG + UV JSON using btp_vox.
# Usage: customize IN_DIR/OUT_DIR and run ./btp_vox/batch_convert.sh

IN_DIR="/Users/graylian/workspace/VoxPLC"
OUT_DIR="/Users/graylian/workspace/project_sh/voxel_world/assets/vox"
JOBS="${JOBS:-4}"

export IN_DIR OUT_DIR JOBS

mkdir -p "$OUT_DIR"

btp_convert_one() {
  local in_vox="$1"
  local stem
  stem="$(basename "$in_vox")"
  stem="${stem%.vox}"

  local out_model="$OUT_DIR/${stem}.glb"
  local out_png="$OUT_DIR/${stem}.png"
  local out_uv="$OUT_DIR/${stem}_uv.json"

#   --pivot corner 
#   --pivot bottom_center
#   --pivot center
#   --atlas-layout by-model | global
#   --format glb | gltf


  local args=(
    "$in_vox"
    "$out_model"
    --format "glb"
    --scale "0.02"
    --uv-json-out "$out_uv"
    --atlas-pot
    --pivot center
  )

  python -m btp_vox.cli "${args[@]}"
}

export -f btp_convert_one

find "$IN_DIR" -type f -name "*.vox" -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -lc 'btp_convert_one "$0"'
