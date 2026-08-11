from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from lofar_det_vsex.association import cluster_association_groups
from lofar_det_vsex.baseline_demo.baseline_unconstrained import unconstrained_clusters_from_edges


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
        }
    )


class GraphClusteringTest(unittest.TestCase):
    def test_unconstrained_merges_weak_chain(self) -> None:
        clusters = unconstrained_clusters_from_edges(chain_components(), chain_edges())
        self.assertEqual(clusters, [[0, 1, 2, 3, 4]])

    def test_constrained_blocks_weak_chain_between_cores(self) -> None:
        graph = nx.Graph()
        components = chain_components()
        edges = chain_edges()
        for _, row in components.iterrows():
            graph.add_node(int(row["component_index"]), **row.to_dict())
        for _, row in edges[edges["edge_type"] == "strong"].iterrows():
            graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type="strong")
        graph.graph["association_edges"] = edges
        graph.graph["components"] = components
        clusters, updated_edges, _final_graph = cluster_association_groups(graph, {})
        self.assertEqual(sorted(clusters), [[0, 1, 2], [3, 4]])
        reasons = set(updated_edges["rejection_reason"].astype(str))
        self.assertIn("weak_edge_would_form_chain", reasons)


if __name__ == "__main__":
    unittest.main()
