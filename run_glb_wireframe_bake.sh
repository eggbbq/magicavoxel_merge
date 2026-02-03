#!/usr/bin/env bash
set -euo pipefail

# === 配置参数 ===
INPUT_GLB="/Users/graylian/workspace/magicavox_model_export/Assets/Vox/test.glb"
OUTPUT_GLB="/Users/graylian/workspace/magicavox_model_export/Assets/Vox/test_bake.glb"
SCALE_FACTOR="0.02"
EXPAND_STEP="10"
MIN_OFFSET="512"
MIN_SCALE="1000.0"
SIZE_SCALE="1000.0"
# 如需调整输入、输出或编码策略，直接修改上面的变量。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/glb_wireframe_bake.py"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "[ERROR] 输入 GLB 不存在: ${INPUT_GLB}" >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] 找不到 Python 工具: ${PY_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_GLB}")"

echo "[INFO] 处理 ${INPUT_GLB} -> ${OUTPUT_GLB}" 
python "${PY_SCRIPT}" \
  "${INPUT_GLB}" \
  "${OUTPUT_GLB}" \
  --scale-factor "${SCALE_FACTOR}" \
  --expand-step "${EXPAND_STEP}" \
  --min-offset "${MIN_OFFSET}" \
  --min-scale "${MIN_SCALE}" \
  --size-scale "${SIZE_SCALE}"

echo "[DONE] UV2 bake 完成。"
