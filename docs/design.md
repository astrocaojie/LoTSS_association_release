# Design

## 1. Data Inputs

Required inputs:

- LoTSS cutouts stored in H5. The H5 path is always supplied by command line
  with `--h5-path`; the code does not assume a fixed location.
- PyBDSF Gaussian catalog, defaulting to
  `data/example/pybdsf_gaussians.fits`.
- YAML configuration, defaulting to `configs/default.yaml`.

Optional inputs:

- H5-stored rms and mean/background maps.
- H5-stored WCS/header metadata.
- H5-stored cutout identifiers and center coordinates.
- H5-stored Gaussian pixel coordinates, when sky-coordinate matching is not
  possible.

## 2. H5 Structure Detection Strategy

`scripts/inspect_h5.py` recursively prints groups, datasets, shapes, dtypes,
and attributes. The IO module searches for common dataset names:

- image: `image`, `data`, `cutout`, `radio`, `img`, `map`
- rms: `rms`, `rms_map`, `noise`
- mean/background: `mean`, `mean_map`, `background`
- metadata: `source_id`, `object_id`, `cutout_id`, `ra`, `dec`, `header`, `wcs`

The H5 structure is not hard-coded. If automatic detection is ambiguous or
wrong, the user can set explicit keys under `h5:` in the YAML config.

The current reader supports the common cases where images are stored as a single
2-D dataset or as an N x H x W dataset. It also supports metadata arrays indexed
by cutout. More exotic layouts should still be inspectable, but may need
explicit config keys or dedicated adapters.

## 3. Gaussian Catalog Reading Strategy

The Gaussian catalog reader uses `astropy.table.Table.read` and accepts FITS
catalogs. It normalizes common field aliases but keeps the original columns.
Expected columns include:

- identifiers: `Source_id`, `Isl_id`, `Gaussian_id`
- sky position: `RA`, `DEC`, optional errors
- fluxes: `Total_flux`, `Peak_flux`
- sizes and angle: `Maj`, `Min`, `PA`, `DC_Maj`, `DC_Min`, `DC_PA`
- code/status: `S_Code`

Missing fields produce warnings. The pipeline should continue when possible,
for example by using row index as a Gaussian identifier or by disabling ellipse
drawing when size columns are absent.

## 4. Cutout and Gaussian Matching Strategy

Three modes are supported:

1. **Sky-coordinate mode**: if a cutout has WCS/header information and the
   Gaussian catalog has RA/DEC, convert Gaussian world coordinates to cutout
   pixels with `astropy.wcs.WCS`, then keep components inside the image bounds.
2. **Pixel-coordinate mode**: if Gaussian pixel positions are already available
   in H5 or an external table, use those directly.
3. **Fallback mode**: if no WCS or pixel coordinate information exists, run
   segmentation-only processing and record that Gaussian matching was skipped.

The release supports sky-coordinate matching and a lightweight pixel-coordinate
path for tables with x/y columns. External metadata catalogs can be added
without changing the downstream graph interface.

## 5. S/N Map Construction Strategy

For each image:

```text
S = (I - mean) / rms
```

Mean priority:

1. H5 mean/background map.
2. `median(image)`.
3. zero.

RMS priority:

1. H5 rms map.
2. robust MAD rms, `1.4826 * median(abs(image - median(image)))`.
3. standard deviation.

Default config:

- `mean_mode: median`
- `rms_mode: mad`
- optional Gaussian smoothing before segmentation with
  `gaussian_smooth_sigma_pix: 1.0`.

The code guards against non-finite pixels and non-positive rms values.

## 6. Multi-Threshold Segmentation Strategy

For every configured threshold:

1. Build `mask = S/N > threshold`.
2. Remove small objects smaller than `min_mask_area_pix`.
3. Optionally apply binary opening and/or closing.
4. Label connected components using configured connectivity.
5. Store masks and label maps.

The `.npz` output for a cutout contains:

- `snr_map`
- `thresholds`
- `masks`
- `labels_by_threshold`

For each Gaussian component, the pipeline records which label contains its
center at every threshold. Pairwise connectivity is then a table lookup: two
components are connected at a threshold if their labels are equal and non-zero.

## 7. Graph Merge Strategy

Each Gaussian component inside a cutout becomes a graph node. Candidate pairs
are generated with `scipy.spatial.cKDTree` so the pipeline does not compare all
pairs in crowded fields.

Candidate-pair limits:

- `max_pair_distance_arcsec`
- `max_pair_distance_beam`

For every candidate pair, the graph module computes:

- shared PyBDSF island flag,
- threshold connectivity flags,
- bridge S/N statistics along the center-to-center line,
- beam-normalized distance,
- approximate Gaussian ellipse overlap,
- position-angle alignment,
- flux-ratio score,
- pixel support between components,
- valley and negative-bowl penalties,
- compact-pair, too-far, and PA-misalignment penalties.

The weighted score is:

```text
merge_score =
    w_same_island * same_pybdsf_island
  + w_conn_3sigma * connected_at_3sigma
  + w_conn_2p5sigma * connected_at_2p5sigma
  + w_conn_2sigma * connected_at_2sigma
  + w_bridge * bridge_snr_score
  + w_overlap * gaussian_overlap
  + w_close * closeness_score
  + w_pa * PA_alignment_score
  + w_pixel_support * pixel_support_score
  - w_valley * valley_penalty
  - w_compact * compact_pair_penalty
  - w_far * too_far_penalty
```

Pairs with `merge_score > merge_threshold` become graph edges. Final source
candidates are connected components of this graph.

## 8. Merged Source Measurement Strategy

For each connected component:

- sum Gaussian total flux,
- measure pixel flux inside 2 sigma and 2.5 sigma support near the component,
- find peak flux,
- compute bounding box,
- estimate largest angular size from pairwise component/pixel distances,
- compute flux-weighted centroid in pixel coordinates,
- convert centroid to RA/DEC when WCS is available,
- estimate second-moment major/minor axes and position angle,
- flag multi-peak, bent, and double-lobe candidates,
- compute a confidence score from edge evidence and source measurements.

The release is intentionally conservative. Measurements are useful for ranking
and inspection, not for final science-quality photometry.

## 9. Output Catalog Fields

`merged_sources` includes:

- `cutout_id`
- `merged_source_id`
- `n_components`
- `gaussian_ids`
- `island_ids`
- `ra`, `dec`
- `centroid_x`, `centroid_y`
- `total_flux_gaussian`
- `total_flux_pixel_2sigma`
- `total_flux_pixel_2p5sigma`
- `peak_flux`
- `LAS_arcsec`
- `PA`
- `merge_confidence`
- `flags`
- `debug_info`

`edges` includes:

- `cutout_id`
- `gaussian_id_1`, `gaussian_id_2`
- `distance_arcsec`
- `same_pybdsf_island`
- `connected_at_3sigma`
- `connected_at_2p5sigma`
- `connected_at_2sigma`
- `bridge_snr_mean`
- `merge_score`
- `merge_decision`
- `positive_evidence`
- `negative_evidence`

`components` preserves per-cutout Gaussian component rows, pixel coordinates,
and segmentation labels.

## 10. Visualization Plan

`scripts/visualize_results.py` and `lofar_det_vsex.visualize` generate one PNG
per cutout. The figure overlays:

- original radio image,
- S/N map,
- 2, 2.5, and 3 sigma contours,
- Gaussian component centers,
- Gaussian ellipses when size columns exist,
- accepted graph edges,
- final merged source bounding boxes and ids.

Figures are written to `outputs/figures/{cutout_id}_vsex.png`.

## 11. Smoke Test Plan

1. Inspect an example H5:

```bash
python scripts/inspect_h5.py --h5-path /path/to/cutouts.h5
```

2. Print Gaussian catalog columns:

```bash
python scripts/print_gaus_catalog_columns.py \
  --gaus-catalog data/example/pybdsf_gaussians.fits
```

3. Run a small pipeline sample:

```bash
python scripts/run_pipeline.py \
  --h5-path /path/to/cutouts.h5 \
  --gaus-catalog data/example/pybdsf_gaussians.fits \
  --config configs/default.yaml \
  --output-dir outputs \
  --limit 5 \
  --make-figures \
  --debug
```

If no H5 example path is available, the code and catalog-inspection script can
still be checked, and the README must state that the user needs to provide
`--h5-path`.
