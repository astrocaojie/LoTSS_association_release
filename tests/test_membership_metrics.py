from __future__ import annotations

import unittest

import numpy as np

from lotss_association.baseline_demo.comparison_metrics import bcubed_metrics, overmerge_split_rates, pairwise_prf


class MembershipMetricsTest(unittest.TestCase):
    def test_pairwise_metrics(self) -> None:
        true = np.asarray(["T1", "T1", "T2", "T2"])
        pred = np.asarray(["P1", "P1", "P1", "P2"])
        metrics = pairwise_prf(true, pred)
        self.assertAlmostEqual(metrics["pairwise_precision"], 1 / 3)
        self.assertAlmostEqual(metrics["pairwise_recall"], 1 / 2)
        self.assertAlmostEqual(metrics["pairwise_f1"], 0.4)

    def test_bcubed_and_merge_split_rates(self) -> None:
        true = np.asarray(["T1", "T1", "T2", "T2"])
        pred = np.asarray(["P1", "P1", "P1", "P2"])
        b3 = bcubed_metrics(true, pred)
        rates = overmerge_split_rates(true, pred)
        self.assertGreater(b3["bcubed_precision"], 0.0)
        self.assertGreater(b3["bcubed_recall"], 0.0)
        self.assertAlmostEqual(rates["overmerge_rate"], 0.5)
        self.assertAlmostEqual(rates["split_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
