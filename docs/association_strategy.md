# Beam-Aware Radio Component Association Strategy

## Purpose

This stage performs PyBDSF Gaussian component association.

A PyBDSF Gaussian component is treated as a local mathematical description of
radio emission, not as a final physical source by itself. The pipeline builds
pairwise evidence between Gaussian components, then groups associated
components into radio structure groups.

Core objects:

- `Gaussian component`: a PyBDSF-fitted local emission part
- `edge`: evidence that two Gaussian components may belong to the same radio
  emission structure
- `association group`: a set of related Gaussian components

No statistical classifier is trained in this stage; the output is produced by
explicit evidence rules and diagnostic thresholds.

## Why SExtractor Cannot Be Copied Directly

SExtractor was primarily designed for optical and infrared imaging, where
thresholded connected pixels and deblending trees are often a natural first
description of sources.

Radio interferometric images are different:

- the image is convolved by the synthesized beam
- nearby pixels have correlated noise
- sidelobes, negative bowls, and calibration artifacts can mimic diffuse
  structure
- a low-threshold connected mask can join unrelated emission through correlated
  noise

Therefore, 2 sigma connectivity cannot be interpreted as physical connectivity.
In this pipeline, 2 sigma connected labels are weak flags only. They are not a
strong association decision.

## Why Segmentation Still Helps

Multi-threshold segmentation remains useful as morphology evidence:

- it can reveal low-surface-brightness emission around multiple components
- it can support possible bridge or diffuse emission
- 3 sigma and 2.5 sigma connectivity are more credible than 2 sigma
- it provides masks for diagnostic measurements and visual review

The rule is conservative: segmentation can support an association, but low-S/N
connectivity is not used as a sufficient condition by itself.

## Evidence System

The pair score combines positive association evidence and artifact penalties.

Positive evidence:

- beam-aware distance
- Gaussian ellipse overlap or small beam-normalized gap
- component PA alignment
- line-to-PA alignment
- 3 sigma connected segmentation
- 2.5 sigma connected segmentation
- beam-width bridge support
- ridge continuity along the component-to-component path
- flux continuity
- flow alignment

Weak or diagnostic evidence:

- 2 sigma connectivity
- only-2sigma connected labels

Negative evidence:

- deep S/N valley between components
- only-2sigma penalty
- negative bowl penalty
- sidelobe or artifact risk
- excessive beam-normalized distance
- large low-threshold mask swallow risk

The conservative config sets:

```yaml
weights_association:
  conn_2sigma: 0.0
```

This means 2 sigma connectivity does not add positive score by default.

## Association Score

The score is:

```text
association_score =
    w_closeness       * closeness_score
  + w_overlap         * ellipse_overlap_score
  + w_pa_alignment    * pa_alignment_score
  + w_conn_3sigma     * connected_at_3sigma
  + w_conn_2p5sigma   * connected_at_2p5sigma
  + w_conn_2sigma     * connected_at_2sigma
  + w_bridge          * bridge_score
  + w_ridge           * ridge_continuity_score
  + w_flux_continuity * flux_continuity_score
  + w_flow_alignment  * flow_alignment_score
  - w_valley          * deep_valley_penalty
  - w_only_2sigma     * only_2sigma_penalty
  - w_negative_bowl   * negative_bowl_penalty
  - w_sidelobe        * sidelobe_risk_penalty
  - w_too_far         * too_far_penalty
  - w_large_mask      * large_mask_swallow_penalty
```

If an edge is connected only at 2 sigma and lacks independent 2.5 sigma, 3
sigma, bridge, ridge, or overlap support, its score is capped by
`association.max_only_2sigma_score`.

Long-distance associations must have multiple supporting signals such as bridge,
ridge, and alignment. Negative bowls and sidelobe risk lower the group quality.

## Graph Strategy

Edges are classified as:

- `strong`: `association_score >= threshold_strong`
- `weak`: `threshold_weak <= association_score < threshold_strong`
- `rejected`: below threshold or blocked by rejection logic

The clustering policy is:

1. Build connected components from strong edges only.
2. Treat those as core association groups.
3. Use weak edges only as attachments.
4. A weak edge may attach a singleton component to an existing core group.
5. A weak edge may not merge two existing core groups.
6. A weak edge may not form a long chain of singleton attachments.

Each edge stores:

- `edge_type`
- `association_decision`
- `rejection_reason`

## Output Catalogs

Recommended outputs:

- `radio_association_groups.csv`
- `radio_association_groups.parquet`
- `radio_association_edges.csv`
- `radio_association_edges.parquet`
- `radio_association_components.csv`
- `radio_association_components.parquet`

Legacy merged-source catalogs are still written for compatibility, but new
analysis should use `radio_association_groups`.

Group catalog fields include:

- `association_group_id`
- `association_type`
- `association_quality`
- `artifact_risk_flags`
- `LAS_arcsec`
- `LAS_beam`
- `association_score_mean`
- `n_strong_edges`
- `n_weak_edges`
- `n_only_2sigma_edges`

## Association Types

The pipeline does not use source-class labels as association types.

Supported types:

- `compact_multi_gaussian`: several Gaussians inside a small beam-scale
  footprint
- `continuous_extended`: credible 3 sigma or 2.5 sigma connectivity, bridge, or
  ridge support
- `diffuse_extended`: larger low-surface-brightness group with weaker S/N and
  artifact caveats
- `linear_or_tail_like`: high axis ratio or ridge-like component layout
- `complex_association`: many associated Gaussians
- `weak_association`: low-score or weakly supported group
- `artifact_risk`: group dominated by negative bowl, sidelobe, or large-mask
  risk

## Association Quality

Quality is categorical, not a probability:

- `high`: strong edges dominate, score is high, only-2sigma evidence is rare,
  and at least one strong morphology signal is present
- `medium`: geometry and segmentation/bridge/ridge evidence are reasonably
  consistent without severe counter-evidence
- `low`: score is low or evidence is mostly distance/geometry
- `suspicious`: many only-2sigma edges, large mask/crowding concerns, or high
  score dispersion
- `artifact_risk`: negative bowl, sidelobe risk, or large-mask swallow risk is
  prominent

## Visualization Semantics

Overview titles use:

```text
cutout_id | gauss=... assoc_edges=... groups=... multi=... max_group=... only2=...
```

Zoom titles use:

```text
cutout_id group_id | n=... LAS=... beam=... quality=... type=...
```

Zoom panels also show the mean association score, strong/weak edge counts,
only-2sigma edge count, and artifact flags.
