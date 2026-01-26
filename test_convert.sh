#!/usr/bin/env bash
set -euo pipefail

INPUT_VOX="/Users/graylian/workspace/Vox/vox/body_map.vox"
OUT_DIR="/Users/graylian/workspace/project_sh/pfl_cc/assets/assets/4_models"

mkdir -p "$OUT_DIR"

# Atlas export (external texture for easy editing/testing)
python -m magicavoxel_merge \
  "$INPUT_VOX" "$OUT_DIR/body_map_atlas_external.glb" \
  --mode atlas \
  --scale 0.02 \
  --weld \
  --atlas-pad 4 \
  --atlas-inset 2 \
  --atlas-layout by-model \
  --center \
#   --texture-out "$OUT_DIR/body_map_atlas.png" \

