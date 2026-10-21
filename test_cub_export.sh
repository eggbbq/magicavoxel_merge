#!/usr/bin/env bash
set -euo pipefail

# 第 2 步：导出模型和绘画用展开贴图。路径和导出参数直接在下面修改。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

in_vox="$SCRIPT_DIR/test_cube.vox"
out_dir="$SCRIPT_DIR/exports/test_cube_paint"
out_model="$out_dir/test_cube_paint.gltf"
out_tex="$out_dir/test_cube_paint.png"

if [[ ! -f "$in_vox" ]]; then
  echo "Input file does not exist: $in_vox" >&2
  exit 1
fi

mkdir -p "$out_dir"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 一行一个参数：直接修改数值，或用 # 注释整行开关选项。
args=(
  --input              "$in_vox"
  --output             "$out_model"
  --format             gltf
  --uv-mode            paint
  --character-apart
  --handedness         right
  --scale              0.02
  --pivot              bottom_center
  --tex-texel-scale    8           # 每体素面单位的像素倍率，原始密度填 1
  --tex-pad            0           # 展开片外边距
  --tex-pot                        # POT；关闭时注释这一整行
  --tex-fmt            auto        # auto / rgba / rgb
  --tex-out            "$out_tex"
  --uv-out             "$out_dir/test_cube_paint_uv.json"
  --uv-template-out    "$out_dir/test_cube_paint_uv.svg"
  --uv-islands-out     "$out_dir/test_cube_paint_islands.json"
)

"$PYTHON_BIN" -m btp_vox.cli "${args[@]}"

printf 'Exported model, texture, SVG guide and UV JSON to %s\n' "$out_dir"
