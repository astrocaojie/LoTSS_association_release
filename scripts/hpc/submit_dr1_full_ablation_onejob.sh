#!/usr/bin/env bash
#SBATCH --job-name=dr1_full_ablation_onejob
#SBATCH --account=astro2
#SBATCH --partition=c32d4m1tp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/dr1_validation/hpc_submission/logs/dr1_full_ablation_onejob_%j.out
#SBATCH --error=outputs/dr1_validation/hpc_submission/logs/dr1_full_ablation_onejob_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/LoTSS_scratch}
ORIGINAL_DATA_ROOT=${ORIGINAL_DATA_ROOT:-${PROJECT_ROOT}/data/LoTSS_DR3}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/lotss_dr3_full}
MANIFEST=${MANIFEST:-${SOURCE_OUTPUT_ROOT}/manifests/lotss_dr3_fits_manifest.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dr1_validation/ablation_full_A0_A10}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:-${PROJECT_ROOT}/configs/real_lotss_conservative.yaml}
VARIANT_CONFIG=${VARIANT_CONFIG:-${PROJECT_ROOT}/configs/dr1_validation/ablation_full_A0_A10_catalogues.yaml}
PARALLEL_VARIANTS=${PARALLEL_VARIANTS:-2}
WORKERS_PER_VARIANT=${WORKERS_PER_VARIANT:-16}

VARIANTS=(
  full_method
  no_ridge_continuity
  no_weak_edge_anti_chaining
  no_artifact_penalties
  no_host_support
  no_lobe_peak_host_contradiction
  no_pa_alignment
  no_ellipse_overlap
  no_multithreshold_contour_connectivity
  no_stage2_relative_scale_constraints
  no_stage2_endpoint_filtering
)

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/failed" "${PROJECT_ROOT}/outputs/dr1_validation/hpc_submission/logs"
JOB_START_EPOCH=$(date +%s)
JOB_START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
JOB_ID=${SLURM_JOB_ID:-manual_$(date -u +"%Y%m%dT%H%M%SZ")}
FAILED_FILE="${OUTPUT_ROOT}/ablation_failed_variants.csv"
echo "variant,status,exit_code,log,metadata" > "${FAILED_FILE}"

write_job_metadata() {
  local status="$1"
  "${PYTHON}" - "${OUTPUT_ROOT}/ablation_job_metadata.json" "${status}" "${JOB_START_EPOCH}" "${JOB_START_UTC}" "${PROJECT_ROOT}" "${OUTPUT_ROOT}" "${VARIANT_CONFIG}" <<'PY'
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
start_epoch = float(sys.argv[3])
start_utc = sys.argv[4]
project = Path(sys.argv[5])
output = Path(sys.argv[6])
config = Path(sys.argv[7])

def git(args):
    try:
        return subprocess.check_output(["git", "-C", str(project), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

deps = {}
for name in ["numpy", "pandas", "astropy", "pyarrow", "matplotlib", "yaml"]:
    try:
        module = importlib.import_module(name)
        deps[name] = getattr(module, "__version__", "installed_version_unknown")
    except Exception as exc:
        deps[name] = f"unavailable: {exc.__class__.__name__}"

payload = {
    "experiment": "dr1_full_ablation_A0_A10",
    "status": status,
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hostname": platform.node(),
    "git_commit": git(["rev-parse", "HEAD"]),
    "project_root": str(project),
    "output_root": str(output),
    "variant_config": str(config),
    "dependency_versions": deps,
    "run_start_utc": start_utc,
    "metadata_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "runtime_seconds": round(time.time() - start_epoch, 3),
    "validation_policy": "DR1 component-only bbox-containment support; no optical IDs.",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_variant() {
  local variant="$1"
  local vroot="${OUTPUT_ROOT}/${variant}"
  local config_dir="${vroot}/config"
  local log_dir="${vroot}/logs"
  local final_dir="${vroot}/association/catalogs/final_science_catalogs"
  local prediction_dir="${vroot}/dr1_validation"
  local start_epoch
  start_epoch=$(date +%s)
  mkdir -p "${config_dir}" "${log_dir}" "${prediction_dir}" "${vroot}/pybdsf" "${vroot}/manifests"
  [[ -e "${vroot}/pybdsf/raw" ]] || ln -s "${SOURCE_OUTPUT_ROOT}/pybdsf/raw" "${vroot}/pybdsf/raw"
  [[ -e "${vroot}/pybdsf/processed" ]] || ln -s "${SOURCE_OUTPUT_ROOT}/pybdsf/processed" "${vroot}/pybdsf/processed"

  "${PYTHON}" - "${CONFIG_TEMPLATE}" "${VARIANT_CONFIG}" "${variant}" "${config_dir}/real_lotss_conservative_${variant}.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

base = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
variant_config = yaml.safe_load(Path(sys.argv[2]).read_text()) or {}
variant = sys.argv[3]
output = Path(sys.argv[4])
switches = {}
for row in variant_config.get("variants", []):
    if row.get("name") == variant:
        switches = row.get("ablation", {}) or {}
        break
base["ablation"] = {**(base.get("ablation", {}) or {}), **switches}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
PY

  "${PYTHON}" - "${vroot}/run_metadata_${variant}.json" "running" "${start_epoch}" "${PROJECT_ROOT}" "${variant}" "${vroot}" "${config_dir}/real_lotss_conservative_${variant}.yaml" <<'PY'
import importlib, json, os, platform, subprocess, sys, time
from pathlib import Path
path = Path(sys.argv[1]); status = sys.argv[2]; start = float(sys.argv[3]); project = Path(sys.argv[4])
variant = sys.argv[5]; output = Path(sys.argv[6]); config = Path(sys.argv[7])
def git(args):
    try:
        return subprocess.check_output(["git", "-C", str(project), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
deps = {}
for name in ["numpy", "pandas", "astropy", "pyarrow", "matplotlib", "yaml"]:
    try:
        module = importlib.import_module(name); deps[name] = getattr(module, "__version__", "installed_version_unknown")
    except Exception as exc:
        deps[name] = f"unavailable: {exc.__class__.__name__}"
payload = {
    "variant": variant, "status": status, "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hostname": platform.node(), "git_commit": git(["rev-parse", "HEAD"]),
    "output_root": str(output), "config_copy": str(config), "dependency_versions": deps,
    "metadata_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "runtime_seconds": round(time.time() - start, 3),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  "${PYTHON}" "${PROJECT_ROOT}/scripts/run_lotss_dr3_full.py" \
    --data-root "${DATA_ROOT}" \
    --original-data-root "${ORIGINAL_DATA_ROOT}" \
    --output-root "${vroot}" \
    --config "${config_dir}/real_lotss_conservative_${variant}.yaml" \
    --manifest "${MANIFEST}" \
    --skip-data-benchmark \
    --input-format fits \
    --release-tag release \
    --use-existing-pybdsf \
    --query-wise-host \
    --resume \
    --num-workers "${WORKERS_PER_VARIANT}" \
    --debug-sample-figures 0 \
    > "${log_dir}/full_${variant}_${JOB_ID}.out" 2> "${log_dir}/full_${variant}_${JOB_ID}.err"

  "${PYTHON}" "${PROJECT_ROOT}/paper/build_result_catalogs.py" \
    --partials-dir "${vroot}/association/catalogs/partials" \
    --output-dir "${final_dir}" \
    --overwrite \
    > "${log_dir}/build_final_catalogs_${variant}_${JOB_ID}.out" 2> "${log_dir}/build_final_catalogs_${variant}_${JOB_ID}.err"

  "${PYTHON}" "${PROJECT_ROOT}/scripts/validation/build_dr1_prediction_catalogue.py" \
    --parent-catalog "${final_dir}/parent_host_catalog.parquet" \
    --local-catalog "${final_dir}/local_group_catalog.parquet" \
    --output "${prediction_dir}/${variant}_predictions.csv.gz" \
    > "${log_dir}/build_dr1_predictions_${variant}_${JOB_ID}.out" 2> "${log_dir}/build_dr1_predictions_${variant}_${JOB_ID}.err"

  "${PYTHON}" - "${vroot}/run_metadata_${variant}.json" "success" "${start_epoch}" "${PROJECT_ROOT}" "${variant}" "${vroot}" "${config_dir}/real_lotss_conservative_${variant}.yaml" <<'PY'
import importlib, json, os, platform, subprocess, sys, time
from pathlib import Path
path = Path(sys.argv[1]); status = sys.argv[2]; start = float(sys.argv[3]); project = Path(sys.argv[4])
variant = sys.argv[5]; output = Path(sys.argv[6]); config = Path(sys.argv[7])
def git(args):
    try:
        return subprocess.check_output(["git", "-C", str(project), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
deps = {}
for name in ["numpy", "pandas", "astropy", "pyarrow", "matplotlib", "yaml"]:
    try:
        module = importlib.import_module(name); deps[name] = getattr(module, "__version__", "installed_version_unknown")
    except Exception as exc:
        deps[name] = f"unavailable: {exc.__class__.__name__}"
payload = {
    "variant": variant, "status": status, "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hostname": platform.node(), "git_commit": git(["rev-parse", "HEAD"]),
    "output_root": str(output), "config_copy": str(config), "dependency_versions": deps,
    "metadata_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "runtime_seconds": round(time.time() - start, 3),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_job_metadata "running"
active=0
for variant in "${VARIANTS[@]}"; do
  (
    set +e
    run_variant "${variant}"
    code=$?
    if (( code != 0 )); then
      echo "${variant},failed,${code},${OUTPUT_ROOT}/${variant}/logs/full_${variant}_${JOB_ID}.err,${OUTPUT_ROOT}/${variant}/run_metadata_${variant}.json" >> "${FAILED_FILE}"
    else
      echo "${variant},success,0,${OUTPUT_ROOT}/${variant}/logs/full_${variant}_${JOB_ID}.err,${OUTPUT_ROOT}/${variant}/run_metadata_${variant}.json" >> "${FAILED_FILE}"
    fi
    exit 0
  ) &
  active=$((active + 1))
  if (( active >= PARALLEL_VARIANTS )); then
    wait -n || true
    active=$((active - 1))
  fi
done
wait || true

"${PYTHON}" "${PROJECT_ROOT}/scripts/validation/dr1_ablation_full_evaluate.py" \
  --config "${VARIANT_CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --mode full \
  > "${OUTPUT_ROOT}/logs/evaluate_${JOB_ID}.out" 2> "${OUTPUT_ROOT}/logs/evaluate_${JOB_ID}.err" || true

write_job_metadata "finished_results_pending_review"
