# Script Reference

The scripts are grouped by use case. Run them from the repository root after
installing the package with `python -m pip install -e ".[dev]"`.

## Core User Commands

- `scripts/run_pipeline.py`: run local Gaussian association for one H5 cutout
  file and one PyBDSF Gaussian catalogue.
- `scripts/run_parent_linking.py`: add large-scale parent-link candidates to
  an existing local-association output directory.
- `scripts/visualize_results.py`: regenerate overview and zoom figures from
  existing output catalogues.

## Survey-Scale LoTSS Processing

- `scripts/run_lotss_dr3_full.py`: production wrapper for LoTSS DR3 tiles.
- `scripts/run_ready_association.py`: process fields with already prepared
  PyBDSF catalogues.
- `scripts/run_missing_pybdsf_lotss_dr3.py`: run PyBDSF only for missing or
  incomplete field catalogues.
- `scripts/build_lotss_dr3_manifest.py`: build a manifest for FITS/H5/PyBDSF
  field products.
- `scripts/stage_lotss_dr3_official_gaussians.py`: stage official LoTSS DR3
  Gaussian catalogues into per-field inputs.
- `scripts/merge_pybdsf_gaussian_catalogs.py`: merge per-field PyBDSF Gaussian
  catalogues.
- `scripts/check_pybdsf_coverage.py`: check catalogue coverage against a field
  manifest.
- `scripts/report_lotss_dr3_progress.py`: summarize production-run progress.

## Inspection and Small Utilities

- `scripts/inspect_h5.py`: print H5 groups, datasets, shapes, and detected
  cutout keys.
- `scripts/print_gaus_catalog_columns.py`: inspect Gaussian catalogue columns
  and detected aliases.
- `scripts/build_segmentation_maps.py`: build reusable S/N segmentation maps.
- `scripts/match_gaussians_to_cutouts.py`: test catalogue-to-cutout matching.
- `scripts/build_component_graph.py`: build a diagnostic component graph for a
  small sample.
- `scripts/export_extended_sources.py`: lightweight compatibility wrapper for
  running the pipeline.

## Validation and Baselines

- `scripts/run_tile_baselines.py`: compare distance, contour, PyBDSF-island,
  unconstrained graph, and full local-association baselines on a tile sample.
- `scripts/run_tile_ablation.py`: run configurable tile-level ablations.
- `scripts/prepare_dr1_ablation.py`: prepare DR1 ablation manifests and
  configs.
- `scripts/run_dr1_ablation_shard.py`: run one DR1 ablation shard.
- `scripts/evaluate_dr1_ablation.py`: evaluate DR1 ablation outputs.
- `scripts/merge_dr1_ablation_results.py`: merge DR1 ablation summaries.
- `scripts/audit_dr1_ablation_configs.py`: check that ablation config switches
  correspond to code paths.
- `scripts/summarize_dr1_variant_smoke.py`: summarize smoke-test metadata for
  DR1 variants.
- `scripts/check_dr1_real_success.py`: compact success check for a DR1 shard.
- `scripts/build_dr1_validation_h5.py`: build a DR1 validation H5 subset.
- `scripts/validation/*.py`: formal DR1 prediction, sanity, ablation,
  baseline, and preflight tools.

## Annotation and Human Review

- `scripts/build_annotation_manifest.py`: create a manifest for visual review.
- `scripts/annotation_server.py`: serve a local annotation interface.
- `scripts/export_annotations.py`: export JSONL annotations to CSV.
- `scripts/summarize_annotations.py`: summarize exported annotations.

## HPC Examples

The `scripts/hpc/` files are SLURM examples for larger validation or baseline
runs. They are provided as examples and should be edited for local queue names,
paths, environments, and resource limits.
