"""Angular-distance-only baseline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    graph_clusters_from_edges,
    group_summary_from_membership,
    membership_from_clusters,
    parameter_label,
    write_method_outputs,
)


def run_distance_threshold(
    components: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    tau_beam: float,
    config: dict[str, Any] | None = None,
    image_shape: tuple[int, int] | None = None,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one distance-only connected-component threshold."""

    t0 = perf_counter()
    config = config or {}
    method = "distance_only"
    parameter_id = f"tau_{parameter_label(tau_beam)}"
    if candidate_pairs.empty:
        accepted: list[tuple[int, int]] = []
    else:
        mask = pd.to_numeric(candidate_pairs["distance_beam"], errors="coerce") <= float(tau_beam)
        accepted = [
            (int(row.component_index_1), int(row.component_index_2))
            for row in candidate_pairs.loc[mask, ["component_index_1", "component_index_2"]].itertuples(index=False)
        ]
    clusters = graph_clusters_from_edges(components["component_index"].astype(int).tolist(), accepted)
    membership = membership_from_clusters(clusters, components, method, parameter_id, f"distance_{parameter_label(tau_beam)}")
    groups = group_summary_from_membership(
        membership,
        components,
        method,
        parameter_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )
    sizes = groups["n_components"].to_numpy(int) if not groups.empty else np.asarray([], dtype=int)
    stats = {
        "method": method,
        "parameter_id": parameter_id,
        "threshold_beam": float(tau_beam),
        "n_groups": int(len(groups)),
        "n_singletons": int(np.count_nonzero(sizes == 1)),
        "n_multi_groups": int(np.count_nonzero(sizes >= 2)),
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "n_groups_gt_10": int(np.count_nonzero(sizes > 10)),
        "n_groups_gt_20": int(np.count_nonzero(sizes > 20)),
        "n_groups_gt_50": int(np.count_nonzero(sizes > 50)),
        "n_edges_accepted": int(len(accepted)),
        "runtime_seconds": perf_counter() - t0,
    }
    if write_outputs:
        stem = f"groups_distance_tau_{parameter_label(tau_beam)}"
        write_method_outputs(config.get("output_dir", "baseline_demo/outputs"), stem, groups, membership)
    return groups, membership, stats


def run_distance_baseline(
    components: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    config: dict[str, Any],
    image_shape: tuple[int, int] | None = None,
    target_multi_groups: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Run configured distance-only threshold scan and optional matched threshold."""

    cfg = config.get("distance_baseline", {}) or {}
    thresholds = [float(value) for value in cfg.get("thresholds_beam", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])]
    groups_by_param: dict[str, pd.DataFrame] = {}
    membership_by_param: dict[str, pd.DataFrame] = {}
    stats: list[dict[str, Any]] = []
    for tau in thresholds:
        groups, membership, row = run_distance_threshold(components, candidate_pairs, tau, config, image_shape=image_shape)
        groups_by_param[row["parameter_id"]] = groups
        membership_by_param[row["parameter_id"]] = membership
        stats.append(row)

    if bool(cfg.get("match_full_multi_group_count", True)) and target_multi_groups is not None and thresholds:
        grid_min = float(min(thresholds))
        grid_max = float(max(thresholds))
        step = float(cfg.get("matched_threshold_step", 0.1))
        if step <= 0:
            step = 0.1
        grid = np.round(np.arange(grid_min, grid_max + 0.5 * step, step), 6)
        best: tuple[float, int, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None = None
        for tau in grid:
            groups, membership, row = run_distance_threshold(
                components,
                candidate_pairs,
                float(tau),
                config,
                image_shape=image_shape,
                write_outputs=False,
            )
            diff = abs(int(row["n_multi_groups"]) - int(target_multi_groups))
            if best is None or (diff, abs(tau - np.median(thresholds))) < (best[1], abs(best[0] - np.median(thresholds))):
                best = (float(tau), diff, groups, membership, row)
        if best is not None:
            tau, _diff, groups, membership, row = best
            row = dict(row)
            row["parameter_id"] = f"matched_tau_{parameter_label(tau)}"
            groups = groups.copy()
            membership = membership.copy()
            groups["parameter_id"] = row["parameter_id"]
            membership["parameter_id"] = row["parameter_id"]
            groups_by_param[row["parameter_id"]] = groups
            membership_by_param[row["parameter_id"]] = membership
            stats.append(row)
            write_method_outputs(
                config.get("output_dir", "baseline_demo/outputs"),
                f"groups_distance_matched_tau_{parameter_label(tau)}",
                groups,
                membership,
            )

    return groups_by_param, membership_by_param, pd.DataFrame(stats)
