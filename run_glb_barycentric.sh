#!/usr/bin/env bash
set -euo pipefail

# === 参数在此声明 ===
INPUT_GLB="/Users/graylian/workspace/magicavox_model_export/Assets/Vox/cube.glb"
OUTPUT_GLB="/Users/graylian/workspace/magicavox_model_export/Assets/Vox/cube_cracked.glb"
SHRINK=0.85
NORMAL_EPS=0.001
PLANE_EPS=0.001
# === 如需修改，直接编辑上面这些变量 ===

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/glb_shrink_barycentric.py"

if [[ ! -f "${INPUT_GLB}" ]]; then
  echo "[ERROR] 输入 GLB 不存在: ${INPUT_GLB}" >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] 找不到 Python 工具: ${PY_SCRIPT}" >&2
  exit 1
fi

python "${PY_SCRIPT}" \
  "${INPUT_GLB}" \
  "${OUTPUT_GLB}" \
  --shrink "${SHRINK}" \
  --normal-eps "${NORMAL_EPS}" \
  --plane-eps "${PLANE_EPS}"
