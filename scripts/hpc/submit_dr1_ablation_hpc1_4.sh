#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/dr1_validation/core_ablation_variants.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dr1_validation/ablation_core6}
MODE=${MODE:-smoke}

mkdir -p "${OUTPUT_ROOT}/logs"

if [[ "${MODE}" == "smoke" ]]; then
  python3 "${PROJECT_ROOT}/scripts/validation/run_dr1_ablation.py" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_ROOT}" \
    --mode smoke
  python3 "${PROJECT_ROOT}/scripts/run_tile_ablation.py" \
    --config "${PROJECT_ROOT}/config/tile_ablation.yaml" \
    --output-dir "${OUTPUT_ROOT}/tile_core6_smoke" \
    --variants full_method no_ridge_continuity no_weak_edge_anti_chaining no_artifact_penalties no_host_support no_lobe_peak_host_contradiction \
    --run-layer2 \
    --skip-host-query \
    --max-host-queries 0
  python3 "${PROJECT_ROOT}/scripts/validation/dr1_ablation_core6_evaluate.py" \
    --config "${PROJECT_ROOT}/configs/dr1_validation/core_ablation_variants.yaml" \
    --output-dir "${OUTPUT_ROOT}/core6_formal_eval"
  echo "Smoke/preflight complete: ${OUTPUT_ROOT}"
  echo "Full run template: MODE=full sbatch --array=0-5 scripts/hpc/submit_dr1_ablation_hpc1_4.sh"
  exit 0
fi

VARIANTS=(full_method no_ridge_continuity no_weak_edge_anti_chaining no_artifact_penalties no_host_support no_lobe_peak_host_contradiction)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
VARIANT=${VARIANTS[${TASK_ID}]}
HOSTNAME_NOW=$(hostname)
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python3 "${PROJECT_ROOT}/scripts/validation/run_dr1_ablation.py" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/${VARIANT}" \
  --mode full \
  --variant "${VARIANT}" \
  > "${LOG_DIR}/dr1_ablation_${VARIANT}_${HOSTNAME_NOW}.log" 2>&1
