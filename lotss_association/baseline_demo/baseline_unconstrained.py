"""Unconstrained graph clustering baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import pandas as pd

from lotss_association.utils import write_dataframe

from .common import group_summary_from_membership, membership_from_clusters, write_method_outputs


def unconstrained_clusters_from_edges(components: pd.DataFrame, edges: pd.DataFrame) -> list[list[int]]:
    """Cluster by ordinary connected components over all strong and accepted weak edges."""

    graph = nx.Graph()
    graph.add_nodes_from(components["component_index"].astype(int).tolist())
    if not edges.empty:
        edge_type = edges.get("edge_type", pd.Series(dtype=str)).astype(str)
        use = edge_type.isin(["strong", "weak"])
        for _, row in edges.loc[use].iterrows():
            graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type=str(row.get("edge_type", "")))
    clusters = [sorted(int(node) for node in cluster) for cluster in nx.connected_components(graph)]
    clusters.sort(key=lambda values: (values[0] if values else -1))
    return clusters


def _strong_core_map(components: pd.DataFrame, edges: pd.DataFrame) -> dict[int, int]:
    graph = nx.Graph()
    graph.add_nodes_from(components["component_index"].astype(int).tolist())
    if not edges.empty:
        for _, row in edges[edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong"].iterrows():
            graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]))
    out: dict[int, int] = {}
    for idx, cluster in enumerate(nx.connected_components(graph)):
        for node in cluster:
            out[int(node)] = idx
    return out


def weak_chain_case_table(
    unconstrained_membership: pd.DataFrame,
    full_membership: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Rank unconstrained groups that span multiple full constrained groups."""

    if unconstrained_membership.empty or full_membership.empty:
        return pd.DataFrame()
    full_by_component = dict(zip(full_membership["component_id"].astype(str), full_membership["predicted_group_id"].astype(str)))
    records: list[dict[str, Any]] = []
    for group_id, rows in unconstrained_membership.groupby("predicted_group_id"):
        component_ids = set(rows["component_id"].astype(str))
        full_groups = {full_by_component[cid] for cid in component_ids if cid in full_by_component}
        if len(full_groups) < 2:
            continue
        weak_edges = 0
        strong_edges = 0
        if not edges.empty:
            for _, edge in edges.iterrows():
                c1 = str(edge.get("gaussian_id_1", edge.get("component_index_1")))
                c2 = str(edge.get("gaussian_id_2", edge.get("component_index_2")))
                if c1 in component_ids and c2 in component_ids:
                    if str(edge.get("edge_type")) == "weak":
                        weak_edges += 1
                    if str(edge.get("edge_type")) == "strong":
                        strong_edges += 1
        records.append(
            {
                "case_type": "weak_chain",
                "unconstrained_group_id": str(group_id),
                "n_components": int(len(component_ids)),
                "n_full_groups": int(len(full_groups)),
                "full_group_ids": ",".join(sorted(full_groups)),
                "n_internal_strong_edges": int(strong_edges),
                "n_internal_accepted_weak_edges": int(weak_edges),
                "ranking_score": float(10 * len(full_groups) + weak_edges + min(len(component_ids), 20) / 20.0),
            }
        )
    return pd.DataFrame(records).sort_values("ranking_score", ascending=False) if records else pd.DataFrame()


def run_unconstrained_graph_baseline(
    components: pd.DataFrame,
    full_edges: pd.DataFrame,
    full_membership: pd.DataFrame | None,
    config: dict[str, Any],
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run ordinary connected components over strong plus accepted weak edges."""

    t0 = perf_counter()
    edges = full_edges.copy()
    clusters = unconstrained_clusters_from_edges(components, edges)
    method = "unconstrained_graph"
    parameter_id = "strong_plus_accepted_weak"
    membership = membership_from_clusters(clusters, components, method, parameter_id, "unconstrained_graph")
    groups = group_summary_from_membership(
        membership,
        components,
        method,
        parameter_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )
    if not edges.empty:
        edges["accepted"] = edges.get("edge_type", pd.Series(dtype=str)).astype(str).isin(["strong", "weak"])

    case_rankings = weak_chain_case_table(membership, full_membership if full_membership is not None else pd.DataFrame(), edges)
    core_by_node = _strong_core_map(components, edges)
    core_counts: defaultdict[str, set[int]] = defaultdict(set)
    for _, row in membership.iterrows():
        try:
            node = int(components.loc[components["component_id"].astype(str) == str(row["component_id"]), "component_index"].iloc[0])
            core_counts[str(row["predicted_group_id"])].add(core_by_node.get(node, node))
        except Exception:
            pass

    out_dir = Path(config.get("output_dir", "baseline_demo/outputs"))
    write_method_outputs(out_dir, "groups_unconstrained_graph", groups, membership)
    write_dataframe(edges, out_dir / "edges_unconstrained.parquet")
    edges.to_csv(out_dir / "edges_unconstrained.csv", index=False)
    if not case_rankings.empty:
        case_rankings.to_csv(out_dir / "case_rankings_weak_chain.csv", index=False)
    stats = {
        "method": method,
        "parameter_id": parameter_id,
        "n_groups": int(len(groups)),
        "n_multi_groups": int((groups["n_components"] >= 2).sum()) if not groups.empty else 0,
        "n_singletons": int((groups["n_components"] == 1).sum()) if not groups.empty else 0,
        "n_strong_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not edges.empty else 0,
        "n_accepted_weak_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not edges.empty else 0,
        "n_rejected_edges": int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "rejected").sum()) if not edges.empty else 0,
        "n_groups_with_multiple_strong_cores": int(sum(len(values) >= 2 for values in core_counts.values())),
        "n_weak_chain_cases": int(len(case_rankings)),
        "runtime_seconds": perf_counter() - t0,
    }
    return groups, membership, edges, case_rankings, stats
