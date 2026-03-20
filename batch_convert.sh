#!/usr/bin/env bash
set -euo pipefail

DIR_IN="/Users/graylian/workspace/VoxPLC"
# DIR_OUT="/Users/graylian/workspace/magicavox_model_export/Assets/Vox"
DIR_OUT="/Users/graylian/Documents/LayaProject/assets/vox"
JOBS="${JOBS:-4}"

mkdir -p "$DIR_OUT"

btp_convert_one_plat() {
  local in_vox="$1"
  local base stem

  base="$(basename "$in_vox")"
  stem="${base%.vox}"

  local out_model="$DIR_OUT/${stem}.gltf"
  local out_tex="$DIR_OUT/${stem}.png"
  local out_uv="$DIR_OUT/${stem}_uv.json"

  local args=(
    --input           $in_vox
    --output          $out_model
    --uv-out          $out_uv
    --uv2
    
    --plat-top-cutout

    --tex-pot
    --tex-fmt         rgba
    --tex-out         $out_tex
    --tex-layout      global
    --tex-reuse-subrects


    --scale           0.02
    --pivot           bottom_center
    --format          gltf
  )

  BTP_VOX_TIMINGS=1 python -m btp_vox.cli "${args[@]}"
}

btp_convert_one_normal() {
  local in_vox="$1"
  local base stem

  base="$(basename "$in_vox")"
  stem="${base%.vox}"

  local out_model="$DIR_OUT/${stem}.gltf"
  local out_tex="$DIR_OUT/${stem}.png"
  local out_uv="$DIR_OUT/${stem}_uv.json"

# --cull            b

  local args=(
    --input           $in_vox
    --output          $out_model
    --uv-out          $out_uv
    --uv2
    --character-flat

    --tex-pot
    --tex-fmt         rgb
    --tex-out         $out_tex
    --tex-layout      global
    --tex-reuse-subrects

    --scale           0.02
    --pivot           bottom_center
    --format          gltf
  )

  BTP_VOX_TIMINGS=1 python -m btp_vox.cli "${args[@]}"
}

btp_convert_one() {
  local in_vox="$1"
  local base
  base="$(basename "$in_vox")"
  if [[ "$base" == *"-plat.vox" ]]; then
    btp_convert_one_plat "$in_vox"
  else
    btp_convert_one_normal "$in_vox"
  fi
}

export -f btp_convert_one
export -f btp_convert_one_plat
export -f btp_convert_one_normal
export DIR_IN DIR_OUT

find "$DIR_IN" -type f -name "*.vox" -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -lc 'btp_convert_one "$0"'
