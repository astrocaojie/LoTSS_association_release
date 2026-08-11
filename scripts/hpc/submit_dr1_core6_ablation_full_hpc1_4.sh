#!/usr/bin/env bash
#SBATCH --job-name=dr1_core6_ablation
#SBATCH --account=astro2
#SBATCH --partition=c32d4m1tp3
#SBATCH --nodelist=hpc-[1-4]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/dr1_validation/hpc_submission/logs/dr1_core6_ablation_%j.out
#SBATCH --error=outputs/dr1_validation/hpc_submission/logs/dr1_core6_ablation_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/LoTSS_scratch}
ORIGINAL_DATA_ROOT=${ORIGINAL_DATA_ROOT:-${PROJECT_ROOT}/data/LoTSS_DR3}
BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dr1_core6_ablation}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/lotss_dr3_full}
MANIFEST=${MANIFEST:-${SOURCE_OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv}
VARIANTS=(full_method no_ridge_continuity no_weak_edge_anti_chaining no_artifact_penalties no_host_support no_lobe_peak_host_contradiction)
VARIANT=${VARIANT:-${1:-}}

if [[ "${BASE_OUTPUT_ROOT}" == *"/full_method_formal"* ]]; then
  echo "Refusing to write core6 ablations under full_method_formal: ${BASE_OUTPUT_ROOT}" >&2
  exit 3
fi

VALID_VARIANT=0
for allowed in "${VARIANTS[@]}"; do
  if [[ "${VARIANT}" == "${allowed}" ]]; then
    VALID_VARIANT=1
    break
  fi
done
if (( VALID_VARIANT == 0 )); then
  echo "Set VARIANT to one of: ${VARIANTS[*]}" >&2
  echo "Example: MODE=full VARIANT=full_method sbatch ${BASH_SOURCE[0]}" >&2
  exit 2
fi

OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${VARIANT}"
CONFIG_DIR="${OUTPUT_ROOT}/config"
LOG_DIR="${OUTPUT_ROOT}/logs"
FINAL_DIR="${OUTPUT_ROOT}/association/catalogs/final_science_catalogs"
PREDICTION_DIR="${OUTPUT_ROOT}/dr1_validation"
RUN_START_EPOCH=$(date +%s)
RUN_START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
RUN_START_TAG=$(date -u +"%Y%m%dT%H%M%SZ")
JOB_TAG=${SLURM_JOB_ID:-manual_${RUN_START_TAG}}

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${PREDICTION_DIR}"
mkdir -p "${OUTPUT_ROOT}/pybdsf" "${OUTPUT_ROOT}/manifests"
if [[ ! -e "${OUTPUT_ROOT}/pybdsf/raw" ]]; then
  ln -s "${SOURCE_OUTPUT_ROOT}/pybdsf/raw" "${OUTPUT_ROOT}/pybdsf/raw"
fi
if [[ ! -e "${OUTPUT_ROOT}/pybdsf/processed" ]]; then
  ln -s "${SOURCE_OUTPUT_ROOT}/pybdsf/processed" "${OUTPUT_ROOT}/pybdsf/processed"
fi
if [[ ! -e "${OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv" && -e "${SOURCE_OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv" ]]; then
  ln -s "${SOURCE_OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv" "${OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv"
fi

"${PYTHON}" - <<PY
from pathlib import Path
import yaml

project = Path("${PROJECT_ROOT}")
base = yaml.safe_load((project / "configs/real_lotss_conservative.yaml").read_text()) or {}
variants = yaml.safe_load((project / "configs/dr1_validation/core_ablation_variants.yaml").read_text()) or {}
switches = {}
for row in variants.get("variants", []):
    if row.get("name") == "${VARIANT}":
        switches = row.get("ablation", {}) or {}
        break
base["ablation"] = {**(base.get("ablation", {}) or {}), **switches}
Path("${CONFIG_DIR}").mkdir(parents=True, exist_ok=True)
(Path("${CONFIG_DIR}") / "real_lotss_conservative_${VARIANT}.yaml").write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
PY

CONFIG="${CONFIG_DIR}/real_lotss_conservative_${VARIANT}.yaml"
METADATA_PATH="${OUTPUT_ROOT}/run_metadata_${VARIANT}_${JOB_TAG}.json"
METADATA_LATEST_PATH="${OUTPUT_ROOT}/run_metadata_${VARIANT}.json"

write_metadata() {
  local stage="$1"
  local status="$2"
  set +e
  "${PYTHON}" - "${METADATA_PATH}" "${METADATA_LATEST_PATH}" "${stage}" "${status}" "${RUN_START_EPOCH}" "${RUN_START_UTC}" "${PROJECT_ROOT}" "${VARIANT}" "${OUTPUT_ROOT}" "${CONFIG}" "${PREDICTION_DIR}/${VARIANT}_predictions.csv.gz" <<'PY'
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

metadata_path = Path(sys.argv[1])
metadata_latest_path = Path(sys.argv[2])
stage = sys.argv[3]
status = sys.argv[4]
start_epoch = float(sys.argv[5])
start_utc = sys.argv[6]
project_root = Path(sys.argv[7])
variant = sys.argv[8]
output_root = Path(sys.argv[9])
config = Path(sys.argv[10])
prediction_catalogue = Path(sys.argv[11])

def git_value(args):
    try:
        return subprocess.check_output(["git", "-C", str(project_root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

deps = {}
for module_name in ["numpy", "pandas", "astropy", "pyarrow", "matplotlib", "yaml"]:
    try:
        module = importlib.import_module(module_name)
        deps[module_name] = getattr(module, "__version__", "installed_version_unknown")
    except Exception as exc:
        deps[module_name] = f"unavailable: {exc.__class__.__name__}"

now = time.time()
payload = {
    "stage": stage,
    "status": status,
    "variant": variant,
    "hostname": platform.node(),
    "project_root": str(project_root),
    "output_root": str(output_root),
    "config_copy": str(config),
    "prediction_catalogue": str(prediction_catalogue),
    "git_commit": git_value(["rev-parse", "HEAD"]),
    "git_status_short": git_value(["status", "--short"]),
    "python_executable": sys.executable,
    "python_version": sys.version.replace("\n", " "),
    "dependency_versions": deps,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "run_start_utc": start_utc,
    "metadata_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    "runtime_seconds": round(now - start_epoch, 3),
}
metadata_path.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
metadata_path.write_text(text, encoding="utf-8")
metadata_latest_path.write_text(text, encoding="utf-8")
PY
  local rc=$?
  set -e
  return "${rc}"
}

on_error() {
  local exit_code=$?
  write_metadata "end" "failed_exit_${exit_code}" || true
  exit "${exit_code}"
}

trap on_error ERR
write_metadata "start" "running" || true

if [[ "${MODE:-smoke}" == "smoke" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/run_lotss_dr3_full.py" \
    --data-root "${DATA_ROOT}" \
    --original-data-root "${ORIGINAL_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --skip-data-benchmark \
    --input-format fits \
    --release-tag release \
    --use-existing-pybdsf \
    --resume \
    --num-workers 4 \
    --limit 1 \
    --debug-sample-figures 0 \
    > "${LOG_DIR}/smoke_${VARIANT}_${JOB_TAG}.out" 2> "${LOG_DIR}/smoke_${VARIANT}_${JOB_TAG}.err"
  write_metadata "end" "smoke_success" || true
  exit 0
fi

"${PYTHON}" "${PROJECT_ROOT}/scripts/run_lotss_dr3_full.py" \
  --data-root "${DATA_ROOT}" \
  --original-data-root "${ORIGINAL_DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --skip-data-benchmark \
  --input-format fits \
  --release-tag release \
  --use-existing-pybdsf \
  --query-wise-host \
  --resume \
  --num-workers 32 \
  --debug-sample-figures 0 \
  > "${LOG_DIR}/full_${VARIANT}_${JOB_TAG}.out" 2> "${LOG_DIR}/full_${VARIANT}_${JOB_TAG}.err"

"${PYTHON}" "${PROJECT_ROOT}/paper/build_result_catalogs.py" \
  --partials-dir "${OUTPUT_ROOT}/association/catalogs/partials" \
  --output-dir "${FINAL_DIR}" \
  --overwrite \
  > "${LOG_DIR}/build_final_catalogs_${VARIANT}_${JOB_TAG}.out" 2> "${LOG_DIR}/build_final_catalogs_${VARIANT}_${JOB_TAG}.err"

"${PYTHON}" "${PROJECT_ROOT}/scripts/validation/build_dr1_prediction_catalogue.py" \
  --parent-catalog "${FINAL_DIR}/parent_host_catalog.parquet" \
  --local-catalog "${FINAL_DIR}/local_group_catalog.parquet" \
  --output "${PREDICTION_DIR}/${VARIANT}_predictions.csv.gz" \
  > "${LOG_DIR}/build_dr1_predictions_${VARIANT}_${JOB_TAG}.out" 2> "${LOG_DIR}/build_dr1_predictions_${VARIANT}_${JOB_TAG}.err"

write_metadata "end" "full_success" || true
