from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from lotss_association.association import cluster_association_groups, compute_association_score
from lotss_association.baseline_demo.reporting import edge_table_hash


def chain_components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component_index": [0, 1, 2, 3, 4],
            "component_id": ["A", "B", "C", "D", "E"],
            "x": [0, 1, 2, 3, 4],
            "y": [0, 0, 0, 0, 0],
        }
    )


def chain_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component_index_1": [0, 1, 2, 3],
            "component_index_2": [1, 2, 3, 4],
            "edge_type": ["strong", "weak", "weak", "strong"],
            "association_decision": [True, False, False, True],
            "association_score": [4.0, 2.5, 2.4, 4.0],
            "rejection_reason": ["", "pending_weak_attachment", "pending_weak_attachment", ""],
        }
    )


def graph_for_chain(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    components = chain_components()
    for _, row in components.iterrows():
        graph.add_node(int(row["component_index"]), **row.to_dict())
    for _, row in edges[edges["edge_type"] == "strong"].iterrows():
        graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type="strong")
    graph.graph["association_edges"] = edges.copy()
    graph.graph["components"] = components
    return graph


class AblationSwitchesTest(unittest.TestCase):
    def test_no_ridge_removes_ridge_score_without_erasing_feature(self) -> None:
        features = {
            "closeness_score": 0.5,
            "ellipse_overlap_score": 0.0,
            "pa_alignment_score": 0.0,
            "connected_at_3sigma": False,
            "connected_at_2p5sigma": False,
            "connected_at_2sigma": False,
            "bridge_score": 0.0,
            "ridge_continuity_score": 1.0,
            "flux_continuity_score": 0.0,
            "flow_alignment_score": 0.0,
            "deep_valley_penalty": 0.0,
            "only_2sigma_penalty": 0.0,
            "negative_bowl_penalty": 0.0,
            "sidelobe_risk_penalty": 0.0,
            "too_far_penalty": 0.0,
            "large_mask_swallow_penalty": 0.0,
        }
        full = compute_association_score(features, {})
        no_ridge = compute_association_score(features, {"ablation": {"use_ridge_continuity": False}})
        self.assertGreater(full, no_ridge)
        self.assertEqual(features["ridge_continuity_score"], 1.0)

    def test_no_artifact_penalties_removes_negative_terms(self) -> None:
        features = {
            "closeness_score": 0.5,
            "ellipse_overlap_score": 0.0,
            "pa_alignment_score": 0.0,
            "connected_at_3sigma": False,
            "connected_at_2p5sigma": False,
            "connected_at_2sigma": False,
            "bridge_score": 0.0,
            "ridge_continuity_score": 0.0,
            "flux_continuity_score": 0.0,
            "flow_alignment_score": 0.0,
            "deep_valley_penalty": 1.0,
            "only_2sigma_penalty": 1.0,
            "negative_bowl_penalty": 1.0,
            "sidelobe_risk_penalty": 1.0,
            "too_far_penalty": 1.0,
            "large_mask_swallow_penalty": 1.0,
        }
        full = compute_association_score(features, {})
        no_artifact = compute_association_score(features, {"ablation": {"use_artifact_penalties_layer1": False}})
        self.assertGreater(no_artifact, full)

    def test_no_anti_chaining_changes_clusters_but_not_edge_table(self) -> None:
        edges = chain_edges()
        full_clusters, full_edges, _ = cluster_association_groups(graph_for_chain(edges), {})
        no_clusters, no_edges, _ = cluster_association_groups(
            graph_for_chain(edges),
            {"ablation": {"use_weak_edge_anti_chaining": False}},
        )
        self.assertEqual(edge_table_hash(full_edges), edge_table_hash(no_edges))
        self.assertEqual(sorted(full_clusters), [[0, 1, 2], [3, 4]])
        self.assertEqual(no_clusters, [[0, 1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
