# LoTSS Association

LoTSS Association is a rule-based Python package for associating PyBDSF
Gaussian components in LoTSS radio images. It groups local radio emission,
measures interpretable association evidence, and proposes parent links for
large double-lobe or extended systems.

The pipeline is designed for reproducible survey processing and validation. It
does not require a machine-learning model: decisions are based on beam-aware
geometry, multi-threshold radio-contour support, ridge and bridge continuity,
artifact penalties, and optional WISE/CatWISE host evidence.

The input graph starts from PyBDSF Gaussian components, not from the PyBDSF
source catalogue. Existing PyBDSF source identifiers can be preserved for
diagnostics, but the association decisions are rebuilt from component geometry,
image support, and validation rules in this repository.

## Method Overview

1. Read LoTSS cutouts or FITS-derived cutouts and match PyBDSF Gaussian
   components into each image.
2. Build beam-normalized pairwise evidence between Gaussian components,
   including distance, morphology, multi-threshold contour connectivity,
   bridge/ridge support, and artifact penalties.
3. Form conservative local association groups from strong edges, with weak
   edges used only as limited attachments.
4. Propose parent-scale candidates for large separated systems and record the
   evidence needed for visual review and validation.

## Repository Layout and Entry Points

- `lofar_det_vsex/`: reusable package modules for IO, segmentation,
  association, parent-linking, validation, plotting, and utilities.
- `scripts/run_pipeline.py`: single H5 cutout-file entry point.
- `scripts/run_lotss_dr3_full.py`: full LoTSS DR3 production wrapper.
- `scripts/run_ready_association.py`: process fields with existing PyBDSF
  catalogues.
- `scripts/run_missing_pybdsf_lotss_dr3.py`: optional PyBDSF backfill helper.
- `scripts/validation/`: DR1 component-reference validation tools.
- `scripts/hpc/`: SLURM templates for larger survey runs.
- `configs/real_lotss_conservative.yaml`: recommended conservative
  configuration.
- `docs/`: method notes and release notes.
- `tests/`: unit tests for the reusable pipeline components.

The most useful documentation files are:

- `docs/association_strategy.md`: local Gaussian-component association logic.
- `docs/parent_association.md`: parent-linking candidate stage and outputs.
- `docs/algorithm_notes.md`: background on PyBDSF components, segmentation,
  and graph-based grouping.
- `docs/design.md`: input handling, catalogue fields, and visualization
  conventions.

## Installation

```bash
git clone https://github.com/astrocaojie/LoTSS_association.git LoTSS_association
cd LoTSS_association
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`PyBDSF` is optional. It is only needed when the repository must generate
missing Gaussian catalogues. If PyBDSF Gaussian catalogues already exist, the
association pipeline can run without PyBDSF.

## Required Inputs

For a normal association run, provide:

- LoTSS cutouts in H5 format, or full FITS images for the production wrapper.
- PyBDSF Gaussian catalogues with RA/Dec and shape/flux columns.
- Optional WISE/CatWISE internet access for host-query support.
- Optional LoTSS DR1 component-reference catalogue for validation.

Large FITS/H5/catalogue products are intentionally not committed. Store local
data under `data/` or pass paths explicitly on the command line.

For reproducible runs, keep the command line, YAML configuration, image
manifest, Gaussian-catalogue manifest, and validation reference manifest next
to the output catalogues.

## Quick Run

```bash
python scripts/run_pipeline.py \
  --h5-path data/example/lotss_cutouts.h5 \
  --gaus-catalog data/example/pybdsf_gaussians.fits \
  --config configs/real_lotss_conservative.yaml \
  --output-dir outputs/example_association \
  --limit 20 \
  --make-figures \
  --overwrite \
  --association-mode
```

Main outputs are written under `outputs/example_association/catalogs/`:

- `radio_association_groups.csv` and `.parquet`
- `radio_association_edges.csv` and `.parquet`
- `radio_association_components.csv` and `.parquet`
- `lofar_det_vsex_merged_sources.csv` and `.parquet`

`radio_association_edges` is the main diagnostic table: it records the positive
and negative evidence for each tested component pair. `radio_association_groups`
is the recommended catalogue for science use after validation and visual
quality control.

## Full LoTSS DR3 Run

The production wrapper scans image roots, finds or records PyBDSF catalogues,
processes each field, and merges final science catalogues.

```bash
python scripts/run_lotss_dr3_full.py \
  --original-data-root data/LoTSS_DR3 \
  --data-root data/LoTSS_scratch \
  --h5-root data/lotss_cutout_2048 \
  --output-root outputs/lotss_dr3_full \
  --config configs/real_lotss_conservative.yaml \
  --use-existing-pybdsf \
  --resume \
  --num-workers 4
```

Example environment variables are provided in `examples/paths.example.env`.

## Parent-Linking

Parent-linking is a candidate stage for large separated radio systems. It does
not rewrite the local Gaussian groups. The main outputs are:

- `large_scale_parent_candidates.csv`
- `large_scale_parent_edges.parquet`
- `parent_link_diagnostics.csv`
- `needs_visual_check.csv`

WISE/CatWISE host matches are recorded as supporting diagnostics. They should
be interpreted together with the radio morphology, bridge/ridge evidence,
artifact flags, and visual review products.

## Validation and Calibration

The DR1 validation path uses the LoTSS DR1 component catalogue as
radio-component reference support. It does not use DR1 optical-ID tables as
association truth.

```bash
python scripts/validation/dr1_full_method_sanity.py \
  --dr1-catalogue data/dr1/lotss_dr1_component_catalogue.csv.gz \
  --parent-catalog outputs/lotss_dr3_full/association/catalogs/final_science_catalogs/parent_host_catalog.parquet \
  --local-catalog outputs/lotss_dr3_full/association/catalogs/final_science_catalogs/local_group_catalog.parquet \
  --output-dir outputs/dr1_validation/full_method_sanity
```

For full ablation and baseline jobs, inspect `scripts/hpc/` and run the
preflight first:

```bash
python scripts/validation/preflight_dr1_full_experiments.py \
  --manifest outputs/lotss_dr3_full/manifests/lotss_dr3_fits_manifest.csv \
  --dr1-catalogue data/dr1/lotss_dr1_component_catalogue.csv.gz
```

For full scientific reporting, describe the calibration choices, reference
sample, and cross-survey limitations in the accompanying paper or release note.

## Testing

```bash
pytest
```

The tests avoid large survey data and cover the core scoring, clustering,
baseline, and validation utilities.

## Citation

If you use this code, cite the repository and the associated paper or data
release. `CITATION.cff` contains the repository metadata and should be updated
with the final author list, DOI, and paper title before publication.

## License

This package ships with an MIT license template. Confirm the final license with
all project contributors before publishing.
