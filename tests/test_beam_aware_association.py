from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lotss_association.association import (
    _beam_model_on_grid,
    _candidate_pairs,
    compute_residual_bridge_features,
    unresolved_pair_veto,
)
from lotss_association.beam import (
    angle_delta_180,
    beam_covariance,
    direction_unit,
    elliptical_beam_distance,
    projected_beam_fwhm,
    sky_pa_to_pixel_angle,
)
from lotss_association.morphology import classify_gaussian_component, effective_pa_weight
from lotss_association.parent_links import classify_parent_acceptance, resolve_parent_conflicts


def meerklass_config() -> dict:
    return {
        "beam": {
            "major_arcsec": 25.64,
            "minor_arcsec": 7.65,
            "pa_deg": -89.25,
            "normalization": "directional_ellipse",
        },
        "association": {
            "enable_residual_bridge": True,
            "enable_unresolved_pair_veto": True,
            "min_bridge_width_beam": 0.8,
            "min_bridge_length_beam": 1.0,
            "residual_bridge_threshold_snr": 2.5,
            "residual_bridge_min_length_fraction": 0.25,
            "residual_bridge_min_area_beams": 0.10,
        },
    }


class BeamGeometryTest(unittest.TestCase):
    def test_beam_covariance_projected_width_and_distance(self) -> None:
        cov = beam_covariance(25.64, 7.65, -89.25)
        beam_angle = sky_pa_to_pixel_angle(-89.25)
        self.assertLess(angle_delta_180(beam_angle, 179.25), 1e-6)
        self.assertAlmostEqual(projected_beam_fwhm(beam_angle, cov), 25.64, places=6)
        self.assertAlmostEqual(projected_beam_fwhm(beam_angle + 90.0, cov), 7.65, places=6)
        delta = direction_unit(beam_angle) * 25.64
        self.assertAlmostEqual(elliptical_beam_distance(delta, cov), 1.0, places=6)

    def test_pa_axis_sign_conversion(self) -> None:
        self.assertAlmostEqual(sky_pa_to_pixel_angle(45.0), 45.0)
        self.assertAlmostEqual(sky_pa_to_pixel_angle(45.0, ra_axis_sign=-1.0), 135.0)

    def test_candidate_pairs_filter_by_elliptical_distance_and_absolute_cap(self) -> None:
        config = meerklass_config()
        config["max_pair_distance_arcsec"] = 240.0
        beam_angle = sky_pa_to_pixel_angle(-89.25)
        major_delta = direction_unit(beam_angle) * 120.0
        minor_delta = direction_unit(beam_angle + 90.0) * 120.0
        components = pd.DataFrame(
            [
                {"x": 300.0, "y": 300.0, "pixel_scale_arcsec": 1.0},
                {"x": 300.0 + major_delta[0], "y": 300.0 + major_delta[1], "pixel_scale_arcsec": 1.0},
                {"x": 300.0 + minor_delta[0], "y": 300.0 + minor_delta[1], "pixel_scale_arcsec": 1.0},
            ]
        )
        self.assertEqual(_candidate_pairs(components, config), [(0, 1)])

        capped = meerklass_config()
        capped["max_pair_distance_arcsec"] = 90.0
        self.assertEqual(_candidate_pairs(components.iloc[:2].copy(), capped), [])


class MorphologyClassificationTest(unittest.TestCase):
    def test_unresolved_beam_like_pa_weight_is_zero(self) -> None:
        row = pd.Series({"_maj": 25.64, "_min": 7.65, "_pa": -89.25, "_dc_maj": np.nan, "_dc_min": np.nan})
        result = classify_gaussian_component(row, meerklass_config())
        self.assertEqual(result["morphology_class"], "unresolved")
        self.assertGreater(result["beam_like_score"], 0.70)
        self.assertEqual(effective_pa_weight({**row.to_dict(), **result}, meerklass_config()), 0.0)

    def test_resolved_deconvolved_component(self) -> None:
        row = pd.Series({"_maj": 45.0, "_min": 13.0, "_pa": -80.0, "_dc_maj": 16.0, "_dc_min": 5.0, "_dc_pa": -80.0})
        result = classify_gaussian_component(row, meerklass_config())
        self.assertEqual(result["morphology_class"], "resolved")
        self.assertGreaterEqual(result["resolved_probability"], 0.75)


class UnresolvedVetoAndBridgeTest(unittest.TestCase):
    def test_unresolved_pair_veto_requires_independent_evidence(self) -> None:
        features = {
            "morphology_class_1": "unresolved",
            "morphology_class_2": "unresolved",
            "beam_like_score_1": 0.9,
            "beam_like_score_2": 0.9,
            "connected_at_3sigma": False,
            "connected_at_2p5sigma": False,
            "bridge_score": 0.0,
            "residual_bridge_score": 0.0,
        }
        veto, reason = unresolved_pair_veto(features, meerklass_config())
        self.assertTrue(veto)
        self.assertEqual(reason, "veto_unresolved_pair_no_independent_radio_evidence")
        features["residual_bridge_score"] = 0.8
        veto, _reason = unresolved_pair_veto(features, meerklass_config())
        self.assertFalse(veto)

    def test_residual_bridge_zero_for_two_point_source_null(self) -> None:
        config = {"beam": {"major_arcsec": 10.0, "minor_arcsec": 4.0, "pa_deg": 90.0}, "association": {"residual_bridge_min_length_fraction": 0.20}}
        snr = np.zeros((80, 80), dtype=float)
        yy, xx = np.mgrid[0:80, 0:80]
        c1 = pd.Series({"x": 25.0, "y": 40.0, "pixel_scale_arcsec": 1.0})
        c2 = pd.Series({"x": 55.0, "y": 40.0, "pixel_scale_arcsec": 1.0})
        snr += _beam_model_on_grid(xx, yy, 25.0, 40.0, 12.0, 1.0, config)
        snr += _beam_model_on_grid(xx, yy, 55.0, 40.0, 10.0, 1.0, config)
        features = compute_residual_bridge_features(c1, c2, snr, config)
        self.assertLess(features["residual_bridge_score"], 0.2)

    def test_residual_bridge_accepts_injected_bridge(self) -> None:
        config = {"beam": {"major_arcsec": 10.0, "minor_arcsec": 4.0, "pa_deg": 90.0}, "association": {"residual_bridge_min_length_fraction": 0.20}}
        snr = np.zeros((80, 80), dtype=float)
        yy, xx = np.mgrid[0:80, 0:80]
        c1 = pd.Series({"x": 25.0, "y": 40.0, "pixel_scale_arcsec": 1.0})
        c2 = pd.Series({"x": 55.0, "y": 40.0, "pixel_scale_arcsec": 1.0})
        snr += _beam_model_on_grid(xx, yy, 25.0, 40.0, 12.0, 1.0, config)
        snr += _beam_model_on_grid(xx, yy, 55.0, 40.0, 10.0, 1.0, config)
        bridge_mask = (xx >= 31) & (xx <= 49) & (np.abs(yy - 40) <= 1)
        snr[bridge_mask] += 4.0
        features = compute_residual_bridge_features(c1, c2, snr, config)
        self.assertGreater(features["residual_bridge_score"], 0.55)


def parent_row(**overrides) -> pd.Series:
    base = {
        "double_lobe_geometry_pass": True,
        "endpoint1_is_parent_endpoint_allowed": True,
        "endpoint2_is_parent_endpoint_allowed": True,
        "endpoint1_hard_point_source_veto": False,
        "endpoint2_hard_point_source_veto": False,
        "endpoint1_hard_compact_veto": False,
        "endpoint2_hard_compact_veto": False,
        "endpoint1_noise_artifact_veto": False,
        "endpoint2_noise_artifact_veto": False,
        "endpoint1_veto_final": False,
        "endpoint2_veto_final": False,
        "artifact_environment_score_pair": 0.0,
        "host_evidence": "needs_host_check",
        "host_quality": "none",
        "lobe_peak_host_found": False,
        "same_3sigma_region_as_neighbor": False,
        "same_2p5sigma_region_as_neighbor": False,
        "bridge_snr_support": False,
        "ridge_continuity_score_pair": 0.0,
        "parent_score_final": 3.0,
        "symmetry_score": 0.8,
        "box_gap_beam_robust": 5.0,
    }
    base.update(overrides)
    return pd.Series(base)


class ConservativeParentAcceptanceTest(unittest.TestCase):
    def test_geometry_only_parent_is_visual_candidate(self) -> None:
        result = classify_parent_acceptance(parent_row(), {})
        self.assertEqual(result["parent_acceptance_class"], "geometry_only_visual_candidate")

    def test_midpoint_host_support_is_accepted(self) -> None:
        result = classify_parent_acceptance(parent_row(host_evidence="supports_double_lobe", host_quality="medium"), {})
        self.assertEqual(result["parent_acceptance_class"], "accepted_high_confidence_parent")
        self.assertIn("midpoint_host", result["independent_parent_evidence"])

    def test_lobe_peak_host_rejects_parent(self) -> None:
        result = classify_parent_acceptance(
            parent_row(host_evidence="contradicts_double_lobe", host_quality="none", lobe_peak_host_found=True),
            {},
        )
        self.assertEqual(result["parent_acceptance_class"], "rejected_parent_candidate")
        self.assertEqual(result["parent_acceptance_reason"], "host_at_lobe_peak")

    def test_parent_conflict_resolution_keeps_best_independent_evidence(self) -> None:
        rows = []
        first = parent_row(
            parent_candidate_id="p1",
            local_group_id_1="a",
            local_group_id_2="b",
            host_evidence="supports_double_lobe",
            host_quality="medium",
            bridge_snr_support=True,
            parent_score_final=5.0,
        )
        second = parent_row(
            parent_candidate_id="p2",
            local_group_id_1="a",
            local_group_id_2="c",
            host_evidence="supports_double_lobe",
            host_quality="medium",
            bridge_snr_support=False,
            parent_score_final=4.0,
        )
        for row in [first, second]:
            rec = row.to_dict()
            rec.update(classify_parent_acceptance(row, {}))
            rec["parent_candidate_quality"] = "high"
            rec["needs_visual_check"] = False
            rec["rejection_reason"] = ""
            rows.append(rec)
        resolved = resolve_parent_conflicts(pd.DataFrame(rows), {})
        kept = resolved[resolved["conflict_resolution_status"].astype(str) == "kept"]
        removed = resolved[resolved["conflict_resolution_status"].astype(str) == "removed"]
        self.assertEqual(kept["parent_candidate_id"].tolist(), ["p1"])
        self.assertEqual(removed["parent_candidate_id"].tolist(), ["p2"])


if __name__ == "__main__":
    unittest.main()
