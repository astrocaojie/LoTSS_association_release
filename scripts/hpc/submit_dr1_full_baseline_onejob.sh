#!/usr/bin/env bash
#SBATCH --job-name=dr1_full_baseline_onejob
#SBATCH --account=astro2
#SBATCH --partition=c32d4m1tp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/dr1_validation/hpc_submission/logs/dr1_full_baseline_onejob_%j.out
#SBATCH --error=outputs/dr1_validation/hpc_submission/logs/dr1_full_baseline_onejob_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python3}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dr1_validation/baselines_full_B0_B7}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/dr1_validation/baseline_variants.yaml}
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/outputs/dr1_validation/hpc_submission/logs"

JOB_START_EPOCH=$(date +%s)
JOB_START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
JOB_ID=${SLURM_JOB_ID:-manual_$(date -u +"%Y%m%dT%H%M%SZ")}

"${PYTHON}" "${PROJECT_ROOT}/scripts/validation/run_dr1_baselines.py" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --mode full \
  > "${OUTPUT_ROOT}/logs/run_baselines_${JOB_ID}.out" 2> "${OUTPUT_ROOT}/logs/run_baselines_${JOB_ID}.err"

"${PYTHON}" - "${OUTPUT_ROOT}/baseline_job_metadata.json" "${JOB_START_EPOCH}" "${JOB_START_UTC}" "${PROJECT_ROOT}" "${OUTPUT_ROOT}" "${CONFIG}" <<'PY'
import json, os, platform, subprocess, sys, time
from pathlib import Path
path = Path(sys.argv[1]); start = float(sys.argv[2]); start_utc = sys.argv[3]; project = Path(sys.argv[4])
output = Path(sys.argv[5]); config = Path(sys.argv[6])
def git(args):
    try:
        return subprocess.check_output(["git", "-C", str(project), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
payload = {
    "experiment": "dr1_full_baseline_B0_B7",
    "status": "finished_results_pending_review",
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hostname": platform.node(),
    "git_commit": git(["rev-parse", "HEAD"]),
    "project_root": str(project),
    "output_root": str(output),
    "config": str(config),
    "run_start_utc": start_utc,
    "metadata_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "runtime_seconds": round(time.time() - start, 3),
    "validation_policy": "DR1 component-only bbox-containment support; no optical IDs.",
    "implemented_methods": ["B0", "B1", "B2", "B3", "B4"],
    "skipped_methods": ["B5", "B6", "B7"],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
