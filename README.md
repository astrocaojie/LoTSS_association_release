# LoTSS Association

LoTSS Association is a rule-based Python package for associating PyBDSF
Gaussian components in LoTSS radio images. It groups local radio emission,
measures interpretable association evidence, and proposes parent links for
large double-lobe or extended systems.

The pipeline is designed for reproducible survey processing and validation. It
does not require a machine-learning model: decisions are based on beam-aware
geometry, multi-threshold radio-contour support, ridge and bridge continuity,
artifact penalties, and optional WISE/CatWISE host evidence.

## Features

- Local Gaussian-component association for H5 cutouts or FITS-derived cutouts.
- Physics-aware parent-link candidates for large separated radio structures.
- Optional host-query support through WISE/CatWISE catalogues.
- DR1 component-reference validation utilities and baseline comparisons.
- Lightweight tests that exercise clustering, scoring, contour assignment, and
  validation metrics without requiring large survey products.

## Repository Layout

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
- `docs/`: methodology and release notes.
- `tests/`: unit tests for the reusable pipeline components.

## Installation

```bash
git clone https://github.com/astrocaojie/LoTSS_association_release.git
cd LoTSS_association_release
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

## Validation

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

## Testing

```bash
pytest
```

The tests avoid large survey data and cover the core scoring, clustering,
baseline, and validation utilities.

## Citation

If you use this code, cite the repository and the associated paper or data
release. Update `CITATION.cff` with the final author list, DOI, repository URL,
and paper title before public release.

## License

This package ships with an MIT license template. Confirm the final license with
all project contributors before publishing.
