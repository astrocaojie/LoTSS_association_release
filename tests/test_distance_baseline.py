from __future__ import annotations

import unittest

import pandas as pd

from lofar_det_vsex.baseline_demo.baseline_distance import run_distance_threshold
from lofar_det_vsex.baseline_demo.neighbour_search import find_candidate_pairs


class DistanceBaselineTest(unittest.TestCase):
    def test_threshold_edges_and_components(self) -> None:
        components = pd.DataFrame(
            {
                "component_index": [0, 1, 2, 3, 4],
                "component_id": ["A", "B", "C", "D", "E"],
                "x": [0.0, 5.0, 20.0, 100.0, 105.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0],
                "pixel_scale_arcsec": [1.0] * 5,
                "ra": [0.0] * 5,
                "dec": [0.0] * 5,
                "total_flux": [1.0] * 5,
                "peak_snr": [10.0] * 5,
            }
        )
        config = {"beam": {"major_arcsec": 5.0, "minor_arcsec": 5.0}, "neighbour_search": {"max_distance_beam": 5.0}}
        candidates = find_candidate_pairs(components, config).pairs
        groups, _membership, stats = run_distance_threshold(components, candidates, 1.1, config, write_outputs=False)
        sizes = sorted(groups["n_components"].tolist())
        self.assertEqual(sizes, [1, 2, 2])
        self.assertEqual(stats["n_edges_accepted"], 2)

        groups_small, _membership_small, stats_small = run_distance_threshold(components, candidates, 0.9, config, write_outputs=False)
        self.assertEqual(sorted(groups_small["n_components"].tolist()), [1, 1, 1, 1, 1])
        self.assertEqual(stats_small["n_edges_accepted"], 0)


if __name__ == "__main__":
    unittest.main()
