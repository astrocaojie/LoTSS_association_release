# Parent-linking Large-Scale Parent Candidate Association

Parent-linking keeps the target as radio component association. It does not train a model
and does not solve large-scale misses by increasing the weight of 2 sigma
connectivity.

## Scope

local remains the main catalog:

```text
Gaussian components -> local association group
```

Parent-linking is only a supplemental candidate stage for large separated radio-lobe
systems:

```text
local groups -> large-scale parent association candidates
```

The output candidates do not replace `radio_association_groups.csv` and do not
rewrite the local groups. Parent candidate scoring never directly connects
individual Gaussian components.

## Why Candidate-Only

local already handles compact multi-Gaussian sources, lobe-internal Gaussians, and
continuous ridges or tails. The missing case is a large source whose separated
lobes have no reliable 2.5 sigma or 3 sigma bridge. Using stronger 2 sigma
connectivity would also connect unrelated emission in crowded or diffuse
regions, so Parent-linking keeps local association conservative and handles only
large-scale separated candidates.

## Local Sanity

`lofar_det_vsex/local_sanity.py` remains diagnostic. It checks whether each local
local group looks like a continuous local radio structure. Important fields
include:

- number of Gaussians and LAS in beams;
- weak and only-2sigma edge fractions;
- ridge gap fraction;
- saddle-to-peak ratio;
- multi-peak separation;
- weak-chain and large-mask flags;
- local overmerge risk.

Suspicious local overmerges can be split when the weak internal edge structure
is clear. If the split is unstable, the original local group is kept and marked
for visual review.

## Parent Candidate Gates

`lofar_det_vsex/parent_association.py` evaluates only pairs of Stage 1.5 local
groups. A pair is rejected before scoring unless it is genuinely large scale:

- `group_distance_beam >= 10.0`, or
- `group_distance_arcsec >= 60.0`.

A pair of two compact singletons is rejected with
`two_compact_singletons_not_parent_candidates` when both groups satisfy:

- `n_gaussians == 1`;
- `LAS_beam < 2.0`;
- `mask_area_beam < 1.5`.

At least one side must be resolved or lobe-like through:

- `LAS_beam >= 3.0`;
- `n_gaussians >= 2`;
- `mask_area_beam >= 2.0`.

## Conservative Geometry

Candidate scoring uses large-scale geometry and midpoint evidence:

- axis alignment;
- facing score;
- midpoint symmetry;
- flux and size ratio;
- lobe-like local morphology;
- compact/core candidate near the midpoint;
- weak bridge as low-weight support only.

If there is no core or host evidence, the pair must satisfy at least two strong
geometry conditions:

- high axis alignment;
- high facing score;
- high midpoint symmetry;
- both groups lobe-like.

Otherwise it is rejected with
`insufficient_large_scale_geometry_support`.

## Quality Levels

The score thresholds are intentionally conservative:

- `high`: score >= 4.0;
- `medium`: score >= 3.2;
- `low`: score >= 2.5, debug only;
- `suspicious`: blocked or conflicting large-scale evidence.

Only high and medium candidates are written to
`large_scale_parent_candidates.csv` by default. Low and rejected rows remain in
`large_scale_parent_edges.parquet` for diagnostics.

## Outputs

Parent-linking writes local sanity diagnostics:

- `local_association_groups.csv`;
- `local_association_edges.parquet`;
- `local_association_components.parquet`.

Main parent-candidate outputs:

- `large_scale_parent_candidates.csv`;
- `large_scale_parent_edges.parquet`.

Diagnostics:

- `parent_link_diagnostics.csv`;
- `local_sanity_diagnostics.csv`;
- `parent_association_diagnostics.csv`;
- `needs_visual_check.csv`.

Figures:

- `figures/overview/`: local/Parent-linking local groups plus high/medium parent candidates;
- `figures/local_zoom/`: local sanity diagnostics;
- `figures/parent_zoom/`: large-scale parent candidates only.

Every Parent-linking parent candidate should be reviewed by a human before being treated
as a parent radio source association.
