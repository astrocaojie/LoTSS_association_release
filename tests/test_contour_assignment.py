from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lofar_det_vsex.baseline_demo.baseline_contour import assign_labels_at_components, build_contour_labels


class ContourAssignmentTest(unittest.TestCase):
    def test_same_and_different_regions(self) -> None:
        snr = np.zeros((6, 6), dtype=float)
        snr[1, 1:3] = 4.0
        snr[4, 4] = 5.0
        _mask, labels, n_labels = build_contour_labels(snr, sigma_threshold=3.0, pixel_connectivity=8)
        components = pd.DataFrame({"x": [1, 2, 4], "y": [1, 1, 4]})
        assigned = assign_labels_at_components(components, labels)
        self.assertEqual(n_labels, 2)
        self.assertEqual(assigned[0], assigned[1])
        self.assertNotEqual(assigned[0], assigned[2])

    def test_four_vs_eight_connectivity(self) -> None:
        snr = np.zeros((4, 4), dtype=float)
        snr[1, 1] = 4.0
        snr[2, 2] = 4.0
        _mask4, _labels4, n4 = build_contour_labels(snr, pixel_connectivity=4)
        _mask8, _labels8, n8 = build_contour_labels(snr, pixel_connectivity=8)
        self.assertEqual(n4, 2)
        self.assertEqual(n8, 1)

    def test_nan_boundary_and_tolerance(self) -> None:
        snr = np.zeros((5, 5), dtype=float)
        snr[2, 2] = 4.0
        snr[0, 0] = np.nan
        _mask, labels, _n = build_contour_labels(snr)
        components = pd.DataFrame({"x": [1, -1, 10], "y": [2, 0, 10]})
        assigned_none = assign_labels_at_components(components, labels, tolerance_pixels=0)
        assigned_tol = assign_labels_at_components(components, labels, tolerance_pixels=1)
        self.assertEqual(int(assigned_none[0]), 0)
        self.assertGreater(int(assigned_tol[0]), 0)
        self.assertEqual(int(assigned_tol[1]), 0)
        self.assertEqual(int(assigned_tol[2]), 0)


if __name__ == "__main__":
    unittest.main()
