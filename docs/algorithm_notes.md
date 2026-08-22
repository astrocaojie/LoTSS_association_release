# Algorithm Notes

This project builds a rule-based extended-source association pipeline for LoTSS
cutouts.
The central design choice is that PyBDSF Gaussian components are treated as
candidate parts, not as final physical radio sources. Low-threshold
SExtractor-style segmentation supplies evidence about diffuse radio continuity,
and a graph merge step turns component-level evidence into merged extended
source candidates.

## 1. PyBDSF Detection Summary

PyBDSF operates on a radio image and normally produces several related object
levels:

- **image**: the input radio image, together with estimated background and rms
  maps.
- **island**: an 8-connected region above an island threshold, usually defined
  relative to the local rms.
- **Gaussian component**: one fitted Gaussian component inside an island.
- **source**: one or more Gaussian components grouped by PyBDSF into a cataloged
  source.

The typical PyBDSF workflow is:

1. Estimate local background and local rms, often using configurable boxes.
2. Detect significant pixels above a peak threshold such as `thresh_pix`.
3. Grow islands down to an island threshold such as `thresh_isl`.
4. Use connected-component labeling, normally 8-connectivity, to define islands.
5. Fit one or more Gaussian components inside each island.
6. Group fitted Gaussians into source entries.
7. Export source catalogs, Gaussian catalogs, and optional diagnostic products
   such as residual, model, rms, mean, and island images.

Important PyBDSF parameter ideas for this project:

- `thresh_pix`: high-significance peak threshold. It affects which peaks are
  considered real enough for fitting.
- `thresh_isl`: lower island threshold. It controls how far emission is grown
  around peaks.
- `rms_box`: scale for estimating local rms and mean/background.
- `rms_map` and `mean_map`: products that can be reused when building an S/N map.
- `atrous_do`: wavelet-based option that can help recover extended emission in
  some cases.
- residual/model/island outputs: useful diagnostics for under-fitting,
  over-splitting, and confusing extended structures.
- Gaussian catalog columns: positions, fluxes, sizes, position angles, island
  identifiers, source identifiers, and quality/status fields may all be useful,
  but not every catalog has the same exact column set.

## 2. SExtractor Detection Summary

SExtractor was designed around image segmentation and measurement. Its core
ideas are:

1. Estimate a background map.
2. Optionally filter the image before detection.
3. Threshold the filtered image.
4. Label connected components as candidate detections.
5. Deblend overlapping detections using a multi-threshold tree.
6. Clean spurious detections.
7. Measure fluxes, shapes, centroids, and flags.

The multi-threshold deblending idea is especially relevant:

- High thresholds identify bright peaks.
- Low thresholds define the larger emission footprint.
- A tree of nested threshold regions separates overlapping sources when a
  parent low-threshold region contains several high-threshold children.

The segmentation map is a first-class output. It is not merely a list of source
positions. It defines the pixels assigned to each source and allows later
measurement of flux, shape, position angle, and size from a mask.

## 3. PyBDSF vs SExtractor

| Aspect | PyBDSF | SExtractor |
| --- | --- | --- |
| Primary domain | Radio interferometric images | Optical/IR images, broadly useful for images |
| Detection unit | Islands above local rms thresholds | Thresholded connected components |
| Internal model | Gaussian fitting inside islands | Pixel segmentation and deblending tree |
| Main output emphasis | Gaussian/source catalogs | Segmentation maps and measurements |
| Strength | Compact and moderately resolved radio components | Explicit pixel masks and multi-threshold structure |
| Weakness for this project | Real extended sources may be split into several islands or sources | Standard deblending tends to split peaks rather than merge them |
| Useful borrowed idea | Gaussian components as robust peak/part descriptors | Low-threshold masks as extended-structure evidence |

## 4. Why PyBDSF Can Split Extended Sources

PyBDSF is excellent at finding statistically significant radio emission and
describing it with Gaussians, but several effects can split a single physical
extended source:

- Low-surface-brightness emission can be weakened by local background/rms
  estimation, especially near bright structure or noise gradients.
- A real radio galaxy can fall into multiple islands if bridge emission drops
  below `thresh_isl`.
- Gaussian grouping usually works inside detected islands; separated islands are
  not always merged into one physical source.
- Bent jets, FRII double lobes, head-tail sources, and fragmented diffuse
  emission may not form a single contiguous high-significance island.
- Gaussian components are descriptive parts of emission, not a direct physical
  source ontology.

## 5. Why Low-Threshold Segmentation Helps Merge Extended Sources

Low-threshold segmentation recovers evidence that is often invisible in a
component-only catalog:

- diffuse bridges between bright peaks,
- common low-S/N envelopes around multiple Gaussians,
- pixel support along the line connecting components,
- valleys that are shallow enough to still support a single extended structure,
- source-scale masks for measuring LAS, flux, shape, and position angle.

For extended radio galaxies, the deciding evidence is often not that two bright
peaks are close, but that the radio image contains plausible low-threshold
continuity between them.

## 6. Reverse-Deblend Strategy

SExtractor usually starts with a broad low-threshold island and then uses
multiple higher thresholds to split that island into separate sources. This
project uses the idea in reverse:

1. Use PyBDSF Gaussians and/or high-S/N peaks as robust component anchors.
2. Build low-threshold segmentation maps at 3, 2.5, and 2 sigma, plus other
   configured thresholds.
3. Ask whether multiple bright components share low-threshold connected
   structure.
4. Build a graph where components are nodes and low-threshold continuity,
   geometry, and catalog context are edge evidence.
5. Use graph connected components as merged extended source candidates.

In short: high thresholds identify parts; low thresholds provide merge evidence.

## 7. Rule-Based Merge Features

Positive evidence:

- same PyBDSF island,
- connected at 3 sigma,
- connected at 2.5 sigma,
- connected at 2 sigma,
- mean/min/max S/N along a bridge line,
- beam-normalized component distance,
- approximate Gaussian ellipse overlap,
- position-angle alignment,
- non-extreme flux ratio,
- pixel support between component centers.

Negative evidence:

- deep valley between components,
- negative bowl along the bridge,
- compact pair penalty,
- excessive separation,
- position-angle misalignment.

The release computes a weighted `merge_score` and links two components when
`merge_score > merge_threshold`. Final merged sources are graph connected
components.

## 8. Known Limitations

- Fully disconnected FRII lobes cannot be solved by radio bridge evidence
  alone. The parent-linking stage records candidate large-scale associations
  for visual review and validation.
- Bright-source artifacts and negative bowls can mislead low-threshold
  segmentation.
- H5 files may not contain WCS. In that case sky-coordinate matching is not
  possible without an external metadata catalog or stored Gaussian pixel
  coordinates.
- PyBDSF Gaussian catalog columns are not completely standardized across
  exports. The code must warn on missing fields and degrade gracefully.
- A rule-based merge score is interpretable, but the numerical thresholds are
  survey-specific. They should be validated and frozen before large-scale use.
