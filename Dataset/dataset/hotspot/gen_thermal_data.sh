#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

# 让后续传入的相对路径都以本脚本目录为基准
cd "${SCRIPT_DIR}"

START_ID="${1:-}"
END_ID="${2:-}"

if [[ -z "${START_ID}" || -z "${END_ID}" ]]; then
  echo "用法: $(basename "$0") START_ID END_ID"
  echo "示例: $(basename "$0") 1 1000"
  exit 2
fi
if ! [[ "${START_ID}" =~ ^[0-9]+$ && "${END_ID}" =~ ^[0-9]+$ ]]; then
  echo "错误: START_ID/END_ID 必须是非负整数"
  exit 2
fi
if (( START_ID > END_ID )); then
  echo "错误: START_ID 不能大于 END_ID"
  exit 2
fi

# 注意：按你的要求，这里都使用相对路径（相对于本脚本所在目录 Dataset/dataset/hotspot）
INPUT_DIR_REL="../output/placement"
OUTPUT_ROOT_REL="../output/thermal/hotspot_config"
CONFIGS_ROOT_REL="../output/thermal/hotspot_config"

# 1) gen_flp_trace.py: 生成 system_i_config/，并在每个布局下生成 system_1.ptrace..system_10.ptrace
"${PY}" "${SCRIPT_DIR}/gen_flp_trace.py" \
  --input_dir "${INPUT_DIR_REL}" \
  --output_root "${OUTPUT_ROOT_REL}" \
  --start_id "${START_ID}" \
  --end_id "${END_ID}"

# 2) gen_powercsv.py: 对每个 system_i_config 下的 system_j.ptrace 生成 system_power_i_j.csv / system_totalpower_i_j.csv
"${PY}" "${SCRIPT_DIR}/gen_powercsv.py" \
  --grid 64 \
  --startid "${START_ID}" \
  --endid "${END_ID}"

# 3) run_hotspot.py: 对每个 system_i_config 下的 system_j.ptrace 做热仿真，生成 Chiplet_{j}.grid.steady
"${PY}" "${SCRIPT_DIR}/run_hotspot.py" \
  --configs_root "${CONFIGS_ROOT_REL}" \
  --start_id "${START_ID}" \
  --end_id "${END_ID}"

# 4) process_csvdata.py: 提取 Layer 2 温度到 tempcsv，输出 system_temp_i_j.csv
"${PY}" "${SCRIPT_DIR}/process_csvdata.py"
