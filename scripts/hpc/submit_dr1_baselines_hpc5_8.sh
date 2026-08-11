#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/dr1_validation/baseline_variants.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dr1_validation/baselines}
MODE=${MODE:-smoke}

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${MODE}" == "smoke" ]]; then
  python3 "${PROJECT_ROOT}/scripts/validation/run_dr1_baselines.py" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_ROOT}" \
    --mode smoke
  echo "Smoke/preflight complete: ${OUTPUT_ROOT}"
  echo "Full run template: MODE=full sbatch --array=0-7 scripts/hpc/submit_dr1_baselines_hpc5_8.sh"
  exit 0
fi

BASELINES=(B0 B1 B2 B3 B4 B5 B6 B7)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BASELINE=${BASELINES[${TASK_ID}]}
HOSTNAME_NOW=$(hostname)
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python3 "${PROJECT_ROOT}/scripts/validation/run_dr1_baselines.py" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/${BASELINE}" \
  --mode full \
  --baseline "${BASELINE}" \
  > "${LOG_DIR}/dr1_baseline_${BASELINE}_${HOSTNAME_NOW}.log" 2>&1
