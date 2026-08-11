"""Adapters for the existing constrained Layer-1 association."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from lofar_det_vsex.association import run_component_association
from lofar_det_vsex.utils import write_dataframe

from .common import (
    group_summary_from_membership,
    membership_from_clusters,
    write_method_outputs,
)


def run_full_layer1_method(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    config: dict[str, Any],
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Any]:
    """Run the repository's current constrained Layer-1 method."""

    t0 = perf_counter()
    result = run_component_association(cutout, segmentation, components, config)
    method = "full_layer1"
    parameter_id = "current_config"
    membership = membership_from_clusters(result.clusters, components, method, parameter_id, "full_layer1")
    groups = group_summary_from_membership(
        membership,
        components,
        method,
        parameter_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )
    edges = result.edges.copy()
    out_dir = Path(config.get("output_dir", "baseline_demo/outputs"))
    write_method_outputs(out_dir, "groups_full_layer1", groups, membership)
    write_dataframe(edges, out_dir / "edges_full_layer1.parquet")
    edges.to_csv(out_dir / "edges_full_layer1.csv", index=False)
    stats = {
        "method": method,
        "parameter_id": parameter_id,
        "n_groups": int(len(groups)),
        "n_multi_groups": int((groups["n_components"] >= 2).sum()) if not groups.empty else 0,
        "n_singletons": int((groups["n_components"] == 1).sum()) if not groups.empty else 0,
        "n_candidate_pairs": int(len(edges)),
        "n_strong_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not edges.empty else 0,
        "n_weak_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not edges.empty else 0,
        "n_rejected_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "rejected").sum()) if not edges.empty else 0,
        "n_accepted_weak_edges": int(((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak") & edges.get("association_decision", pd.Series(dtype=bool)).astype(bool)).sum()) if not edges.empty else 0,
        "runtime_seconds": perf_counter() - t0,
    }
    return groups, membership, edges, stats, result
