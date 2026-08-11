# Parent Association Purity Design

## Decision

The main parent-association validation should use route B only where the
external catalogue exposes a real association/grouping identifier for radio
components.

Current schema probe:

- LGZ / LoTSS DR1: usable for route B. The DR1 association table has
  `Source_Name`, `LGZ_Assoc`, `LGZ_Assoc_Qual`, and the DR1 component table
  maps `Source_Name -> Component_Name`.
- RGZ DR1: not usable for strict route B with the downloaded tables. The tables
  have `RGZID`, `ZooniverseID`, positions, and `N_comp`, but no list of radio
  component IDs or radio component coordinates per source association.

Therefore the primary paper metric should be LGZ/LoTSS DR1 pair precision. RGZ,
ROGUE, FR, and GRG catalogues can still support geometry-based route A or broad
coverage/internal-consistency route C.

## Route B Metric

Inputs:

- production parent-linking parent memberships:
  `crossmatch_parent_external_support/cache/parent_all_membership.parquet`
  and `parent_host_membership.parquet`.
- Existing local-to-external mapping:
  `crossmatch_parent_external_support/cache/local_to_external_source_mapping.parquet`.
- Existing external component evidence:
  `crossmatch_parent_external_support/cache/external_component_evidence.parquet`.

Procedure:

1. Keep only `external_catalogue == "LoTSS DR1 component/association"`.
2. Keep the best local-to-LGZ source mapping
   (`rank_within_local_catalogue == 1`) for each production parent-linking local group.
3. For each parent, inspect the two member local groups.
4. A parent is broadly testable when both local groups have LGZ source mappings.
5. A parent is in the strict main ground-truth subset when the mapped LGZ
   source IDs are multi-component (`external_source_n_components >= 2`).
6. Supported means both local groups map to the same LGZ `Source_Name`.
7. Conflict means the two local groups map to different LGZ `Source_Name`
   values.
8. Pair precision is `supported / testable`.

The strict multi-component subset is the recommended headline result. The broad
all-LGZ-mapped number is useful as a diagnostic, but it is harsher than the
available ground truth because most DR1 component sources are single-component.

## Prototype Results

Using the existing validation cache:

- LGZ best local mappings: 47,935.
- LGZ source IDs in component evidence: 318,520.
- LGZ source IDs with two or more components: 3,774.

Results:

| parent set | metric mode | N parent | testable | supported | conflict | f testable | pair precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parent_all | all_lgz_mapped | 122,261 | 234 | 85 | 149 | 0.001914 | 0.363248 |
| parent_all | multicomponent_lgz_ground_truth | 122,261 | 88 | 85 | 3 | 0.000720 | 0.965909 |
| parent_host | all_lgz_mapped | 80,212 | 99 | 49 | 50 | 0.001234 | 0.494949 |
| parent_host | multicomponent_lgz_ground_truth | 80,212 | 49 | 49 | 0 | 0.000611 | 1.000000 |

## Caveats

- The testable fraction is small because this is DR1-overlap ground truth being
  applied to the full DR3 parent catalogue.
- The broad `all_lgz_mapped` conflict count includes many cases where both DR3
  lobes land on different single-component DR1 sources. Those rows are not a
  clean pair-association false-positive set.
- For RGZ/ROGUE/FR/GRG, a route-A geometry reconstruction is still useful, but
  it is a derived validation sample rather than free association ground truth.
