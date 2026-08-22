"""PyBDSF island-membership baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import group_summary_from_membership, membership_from_clusters, write_method_outputs


def run_pybdsf_island_baseline(
    components: pd.DataFrame,
    config: dict[str, Any],
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Group Gaussian components by PyBDSF island id."""

    method = "pybdsf_island"
    parameter_id = "native"
    if "island_id" not in components and "_island_id" not in components:
        stats = {"method": method, "available": False, "reason": "missing_island_id"}
        groups = pd.DataFrame()
        membership = pd.DataFrame()
        return groups, membership, stats

    island_col = "island_id" if "island_id" in components else "_island_id"
    valid = components[island_col].notna() & ~components[island_col].astype(str).isin(["", "-1", "nan", "None"])
    work = components.loc[valid].copy()
    clusters: list[list[int]] = []
    for _island, rows in work.groupby(island_col, sort=True):
        clusters.append(sorted(rows["component_index"].astype(int).tolist()))
    missing = components.loc[~components["component_index"].astype(int).isin([node for c in clusters for node in c])]
    for _, row in missing.iterrows():
        clusters.append([int(row["component_index"])])
    clusters.sort(key=lambda values: (values[0] if values else -1))
    membership = membership_from_clusters(clusters, components, method, parameter_id, "pybdsf_island")
    groups = group_summary_from_membership(
        membership,
        components,
        method,
        parameter_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )
    stats = {
        "method": method,
        "available": True,
        "n_groups": int(len(groups)),
        "n_multi_groups": int((groups["n_components"] >= 2).sum()) if not groups.empty else 0,
        "n_singletons": int((groups["n_components"] == 1).sum()) if not groups.empty else 0,
    }
    write_method_outputs(config.get("output_dir", "baseline_demo/outputs"), "groups_pybdsf_island", groups, membership)
    return groups, membership, stats
