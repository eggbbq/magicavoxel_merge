#!/usr/bin/env bash
set -euo pipefail

# === 参数配置 ===
INPUT_DIR="/Users/graylian/workspace/magicavox_model_export/Assets/Vox"
OUTPUT_DIR="/Users/graylian/workspace/magicavox_model_export/Assets/Vox"
SIZE_SCALE="100.0"
AXIS_EPS="0.001"
# 如需调整输入/输出或编码参数，直接修改上方变量。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/glb_wireframe_mark.py"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] 输入目录不存在: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] 找不到 Python 工具: ${PY_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

count=0
found_any=0
while IFS= read -r -d '' input_glb; do
  found_any=1
  base="$(basename "${input_glb}")"
  stem="${base%.glb}"
  output_glb="${OUTPUT_DIR}/${stem}.glb"
  echo "[INFO] 处理 ${input_glb} -> ${output_glb}"
  python "${PY_SCRIPT}" \
    "${input_glb}" \
    "${output_glb}" \
    --size-scale "${SIZE_SCALE}" \
    --axis-eps "${AXIS_EPS}" || exit $?
  count=$((count + 1))
done < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.glb" -print0)

if [[ "${found_any}" == "0" ]]; then
  echo "[INFO] 输入目录中没有 .glb 文件: ${INPUT_DIR}" >&2
  exit 0
fi

echo "[DONE] 已处理 ${count} 个 GLB。"
