from __future__ import annotations

import unittest

import pandas as pd

from lofar_det_vsex.validation.bbox_support import bbox_contains, match_predictions_to_dr1_components
from lofar_det_vsex.validation.footprint import add_bbox_centre, build_dr1_sky_footprint, filter_predictions_in_footprint
from lofar_det_vsex.validation.metrics import compute_support_rates, wilson_interval


class Dr1ValidationMetricsTest(unittest.TestCase):
    def test_bbox_contains_handles_ra_wrap(self) -> None:
        mask = bbox_contains([359.8, 0.1, 10.0], [1.0, 1.0, 1.0], 359.5, 0.5, 0.0, 2.0)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_bbox_centre_handles_ra_wrap(self) -> None:
        frame = pd.DataFrame({"bbox_ra_min": [359.5], "bbox_ra_max": [0.5], "bbox_dec_min": [0.0], "bbox_dec_max": [2.0]})
        out = add_bbox_centre(frame)
        self.assertAlmostEqual(float(out["ra_centre"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(out["dec_centre"].iloc[0]), 1.0)

    def test_match_predictions_to_components(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_id": ["p0", "p1"],
                "sample": ["parent_high", "parent_high"],
                "bbox_ra_min": [359.5, 10.0],
                "bbox_ra_max": [0.5, 11.0],
                "bbox_dec_min": [0.0, 0.0],
                "bbox_dec_max": [2.0, 1.0],
                "total_flux_jy": [1.0, 1.0],
            }
        )
        dr1 = pd.DataFrame(
            {
                "ra": [0.1, 12.0],
                "dec": [1.0, 0.5],
                "valid_position": [True, True],
                "dr1_component_id": ["c0", "c1"],
                "dr1_source_id": ["s0", "s1"],
            }
        )
        matched = match_predictions_to_dr1_components(predictions, dr1, grid_size_deg=1.0)
        self.assertEqual(matched["supported_by_dr1"].tolist(), [True, False])
        self.assertEqual(matched["matched_dr1_component_ids"].tolist(), ["c0", ""])

    def test_footprint_and_support_rates(self) -> None:
        dr1 = pd.DataFrame({"ra": [10.1], "dec": [0.1], "valid_position": [True]})
        footprint = build_dr1_sky_footprint(dr1, grid_size_deg=1.0)
        predictions = pd.DataFrame(
            {
                "source_id": ["p0", "p1"],
                "sample": ["local_multigaussian_extended", "local_multigaussian_extended"],
                "bbox_ra_min": [10.0, 20.0],
                "bbox_ra_max": [10.2, 20.2],
                "bbox_dec_min": [0.0, 0.0],
                "bbox_dec_max": [0.2, 0.2],
                "total_flux_jy": [0.1, 0.1],
                "supported_by_dr1": [True, True],
            }
        )
        filtered = filter_predictions_in_footprint(predictions, footprint)
        self.assertEqual(filtered["in_dr1_footprint"].tolist(), [True, False])
        rates = compute_support_rates(filtered, [0.05])
        self.assertEqual(int(rates["n_in_dr1_footprint"].iloc[0]), 1)
        self.assertEqual(int(rates["n_supported_by_dr1_component"].iloc[0]), 1)

    def test_wilson_interval_bounds(self) -> None:
        low, high = wilson_interval(5, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
