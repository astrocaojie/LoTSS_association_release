#!/usr/bin/env bash
set -euo pipefail

python scripts/run_pipeline.py \
  --h5-path "${H5_PATH:?set H5_PATH}" \
  --gaus-catalog "${GAUS_CATALOG:?set GAUS_CATALOG}" \
  --config configs/real_lotss_conservative.yaml \
  --output-dir outputs/smoke_association \
  --limit "${LIMIT:-20}" \
  --make-figures \
  --overwrite \
  --association-mode
