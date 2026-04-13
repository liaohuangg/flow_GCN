#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

INPUT_DIR="${SCRIPT_DIR}/../output/placement"
OUTPUT_ROOT="${SCRIPT_DIR}/../output/thermal/hotspot_config"

START_ID="${1:-}"
END_ID="${2:-}"

if [[ -n "${START_ID}" || -n "${END_ID}" ]]; then
  if [[ -z "${START_ID}" || -z "${END_ID}" ]]; then
    echo "用法: $(basename "$0") [START_ID END_ID]"
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
fi

mkdir -p "${OUTPUT_ROOT}"

shopt -s nullglob
for json_path in "${INPUT_DIR}"/system_*.json; do
  base="$(basename "${json_path}" .json)"
  id="${base#system_}"
  if [[ "${id}" == "${base}" || ! "${id}" =~ ^[0-9]+$ ]]; then
    continue
  fi
  if [[ -n "${START_ID}" ]]; then
    if (( id < START_ID || id > END_ID )); then
      continue
    fi
  fi

  out_dir="${OUTPUT_ROOT}/${base}_config"
  mkdir -p "${out_dir}"

  "${PY}" "${SCRIPT_DIR}/gen_flp_trace.py" \
    --json_path "${json_path}" \
    --output_dir "${out_dir}" \
    --output_flp "${out_dir}/${base}.flp" \
    --output_ptrace "${out_dir}/${base}.ptrace"
done

#/root/workspace/flow_GCN/Dataset/dataset/hotspot/gen.sh 1 1000