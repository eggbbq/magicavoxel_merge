#!/usr/bin/env bash
set -euo pipefail

# 第 4 步：读取加工后的模型和贴图，生成新模型和紧凑贴图。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

in_model="$SCRIPT_DIR/exports/test_cube_paint/test_cube_paint.gltf"
in_tex="$SCRIPT_DIR/exports/test_cube_paint/test_cube_paint.png"
out_dir="$SCRIPT_DIR/exports/test_cube_runtime"
out_model="$out_dir/test_cube_runtime.gltf"
out_tex="$out_dir/test_cube_runtime.png"

if [[ ! -f "$in_model" ]]; then
  echo "Input model does not exist: $in_model" >&2
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 一行一个参数：直接修改数值，或用 # 注释整行开关选项。
args=(
  --input              "$in_model"
  --texture            "$in_tex"  # 注释这一行可自动读取模型引用的贴图
  --output             "$out_model"
  --tex-out            "$out_tex"
  --tex-pad            0           # 纹理块四周边距
  --tex-pot
#   --no-tex-pot         1            # 非 POT；需要 POT 时改成 --tex-pot
  --deduplicate                  # 取消注释，复用像素完全相同的纹理块
  --merge-similar                  # 合并相似块（也匹配旋转/翻转），注释此行关闭
  --similarity-threshold 0.98       # 0～1，越高越严格；1 只合并完全相同的块
  --similarity-max-error 25         # 单通道最大色差 0～255；alpha 必须完全一致
  --compress-solid-patches         # 纯色区块压成单像素采样，注释此行关闭
  --pack               dense       # 更充分装箱；shelf 更快
  --report-out         "$out_dir/repack_report.json"
)

"$PYTHON_BIN" -m btp_vox.repack "${args[@]}"
printf 'Exported new model and compact texture to %s\n' "$out_dir"
