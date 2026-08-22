# Software Package Overview

This repository is organized as a reusable source-association package plus
survey-processing scripts. The reusable Python package is `lotss_association`;
the `scripts/` directory contains command-line entry points for common
workflows and larger LoTSS production or validation runs.

## Public Package Layers

The package is split into four practical layers:

1. **Input and normalization**
   - `lotss_association.io`: inspect H5 files and read radio cutouts.
   - `lotss_association.catalog`: read and normalize PyBDSF Gaussian catalogues.
   - `lotss_association.matching`: match catalogue components into each cutout.
2. **Local Gaussian association**
   - `lotss_association.segmentation`: build S/N maps and multi-threshold
     connected-component labels.
   - `lotss_association.beam`: compute beam-aware distances and projected
     beam widths.
   - `lotss_association.morphology`: classify compact, resolved, and lobe-like
     Gaussian components.
   - `lotss_association.association`: score candidate Gaussian pairs and form
     conservative local association groups.
3. **Parent-candidate stage**
   - `lotss_association.local_sanity`: diagnose possible local overmerges.
   - `lotss_association.parent_seed`: identify local groups that can serve as
     large-scale parent endpoints.
   - `lotss_association.parent_links`: generate and score parent-link
     candidates for large separated systems.
   - `lotss_association.host_query` and `lotss_association.host_support`: query
     and score WISE/CatWISE host evidence as diagnostics.
4. **Validation and reporting**
   - `lotss_association.validation`: DR1 component-reference validation helpers.
   - `lotss_association.baseline_demo`: distance, contour, PyBDSF-island, and
     unconstrained-graph baselines.
   - `lotss_association.ablation`: ablation-result summaries.
   - `lotss_association.visualize`: overview and zoom diagnostic plots.
   - `lotss_association.annotation`: optional human-review manifest and export
     helpers.

## Stable and Experimental Areas

The recommended public path is:

```text
catalog/io -> segmentation -> association -> parent_links -> validation/visualization
```

The following modules are useful but should be treated as validation or
workflow support rather than the minimal public API:

- `lotss_association.baseline_demo`
- `lotss_association.ablation`
- `lotss_association.annotation`
- `lotss_association.parent_links_endpoint_guarded`

These modules are included so that survey tests and paper-validation products
can be reproduced from the same repository, but normal users can start with
`scripts/run_pipeline.py` and `scripts/run_parent_linking.py`.

## Reuse Boundaries

The core scoring functions are survey-aware but not hard-coded to a single
local filesystem. Reuse on another radio survey normally requires:

- a Gaussian/component catalogue with sky positions, fluxes, and size columns;
- image cutouts or FITS tiles with a reliable beam and WCS;
- a YAML configuration tuned to the survey resolution, sensitivity, and
  artefact environment;
- a validation reference or review protocol for calibrating thresholds.

Large survey products, generated catalogues, figures, logs, and cluster paths
are intentionally excluded from the repository.
