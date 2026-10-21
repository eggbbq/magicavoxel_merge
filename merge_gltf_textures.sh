#!/usr/bin/env bash
set -euo pipefail

# 多个模型共享一张颜色贴图，输出模型分别保存。直接修改模型列表和逐行参数。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

in_models=(
  "$SCRIPT_DIR/hero1.gltf"
  "$SCRIPT_DIR/hero2.gltf"
)
out_dir="$SCRIPT_DIR/exports/shared_atlas"
out_tex="$out_dir/shared_atlas.png"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --input                  "${in_models[@]}"
  --output-dir             "$out_dir"
  --tex-out                "$out_tex"
  --tex-pad                0          # 每块纹理的边距
  --tex-pot                           # 改为 --no-tex-pot 输出非 POT
  --merge-similar                     # 注释此行关闭相似合并
  --similarity-threshold   0.98       # 1 只合并完全相同的块
  --similarity-max-error   16         # 单通道最大色差；alpha 必须完全一致
  # --deduplicate                    # 关闭相似合并后，可单独开启完全相同区块去重
  --compress-solid-patches           # 注释此行关闭纯色压缩
  --pack                   dense
  --report-out             "$out_dir/merge_report.json"
)

"$PYTHON_BIN" -m btp_vox.merge_textures "${args[@]}"
printf 'Exported separate glTF models sharing one PNG to %s\n' "$out_dir"
