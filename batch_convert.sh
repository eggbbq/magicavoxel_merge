#!/usr/bin/env bash
set -euo pipefail

# Local defaults for this repo. Override with env vars when needed.
DIR_IN="${DIR_IN:-../Mistedge/vox/vox}"
DIR_OUT="${DIR_OUT:-../Mistedge/assets/vox}"
JOBS="${JOBS:-4}"
BTP_VOX_TIMINGS="${BTP_VOX_TIMINGS:-1}"

# Optional perf knobs for reuse_map (keep as env vars if needed).
BTP_VOX_FAST_PACK_THRESHOLD="${BTP_VOX_FAST_PACK_THRESHOLD:-128}"
BTP_VOX_REUSE_SUBRECT_LIMIT="${BTP_VOX_REUSE_SUBRECT_LIMIT:-5000}"
BTP_VOX_REUSE_MAX_CANDIDATES="${BTP_VOX_REUSE_MAX_CANDIDATES:-256}"

if [[ ! -d "$DIR_IN" ]]; then
  echo "Input directory does not exist: $DIR_IN" >&2
  exit 1
fi

mkdir -p "$DIR_OUT"

btp_get_fixed_size() {
  local in_vox="$1"
  local base stem key entry
  base="$(basename "$in_vox")"
  stem="${base%.vox}"

  for key in "$in_vox" "$base" "$stem"; do
    while IFS= read -r entry; do
      [[ -z "$entry" ]] && continue
      if [[ "${entry%%=*}" == "$key" ]]; then
        printf '%s\n' "${entry#*=}"
        return 0
      fi
    done <<< "${BTP_FIXED_SIZE_RULES:-}"
  done
  return 1
}

btp_is_plat_t_file() {
  local in_vox="$1"
  local base stem key entry
  base="$(basename "$in_vox")"
  stem="${base%.vox}"

  for key in "$in_vox" "$base" "$stem"; do
    while IFS= read -r entry; do
      [[ -z "$entry" ]] && continue
      if [[ "$entry" == "$key" ]]; then
        return 0
      fi
    done <<< "${BTP_PLAT_T_FILES:-}"
  done
  return 1
}


# 配置 固定纹理尺寸 [填入vox文件名]
fixed_sizes=(
  "terrain2=512x512"
)

# 配置 强制采用plat-t模式导出 [填入vox文件名]
plat_t_files=(
  # "terrain2"
)

btp_convert_one() {
  local in_vox="$1"
  local base stem fixed_size=""

  base="$(basename "$in_vox")"
  stem="${base%.vox}"

  local out_model="$DIR_OUT/${stem}.gltf"
  local out_tex="$DIR_OUT/${stem}.png"
  local out_uv="$DIR_OUT/${stem}_uv.json"

  local args=(
    --input           "$in_vox"
    --output          "$out_model"
    --uv-out          "$out_uv"
    --uv2
    --character-flat
    --tex-pot
    --tex-fmt         rgb
    --tex-out         "$out_tex"
    --tex-layout      global
    --tex-compress-solid-quads
    --face-alias-uv-remap
    --tex-reuse-subrects
    --scale           0.02
    --pivot           bottom_center
    --format          gltf
  )
  if btp_is_plat_t_file "$in_vox"; then
    args+=(--plat-top-cutout)
  fi
  if fixed_size="$(btp_get_fixed_size "$in_vox")"; then
    args+=(--tex-fixed-size "$fixed_size")
  fi

  BTP_VOX_TIMINGS="$BTP_VOX_TIMINGS" python -m btp_vox.cli "${args[@]}"
}

BTP_BATCH_CONFIG_DECLS="$(declare -p fixed_sizes plat_t_files)"
BTP_FIXED_SIZE_RULES="$(printf '%s\n' "${fixed_sizes[@]+"${fixed_sizes[@]}"}")"
BTP_PLAT_T_FILES="$(printf '%s\n' "${plat_t_files[@]+"${plat_t_files[@]}"}")"

export -f btp_convert_one
export -f btp_get_fixed_size
export -f btp_is_plat_t_file
export DIR_OUT BTP_VOX_TIMINGS BTP_VOX_FAST_PACK_THRESHOLD
export BTP_VOX_REUSE_SUBRECT_LIMIT BTP_VOX_REUSE_MAX_CANDIDATES
export BTP_FIXED_SIZE_RULES BTP_PLAT_T_FILES

find "$DIR_IN" -type f -name "*.vox" -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -lc 'btp_convert_one "$0"'
