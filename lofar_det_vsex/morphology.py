"""Beam-aware Gaussian morphology classification helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .beam import (
    SIGMA_TO_FWHM,
    angle_delta_180,
    beam_axes_from_config,
    beam_covariance_from_config,
    gaussian_covariance,
    sky_pa_to_pixel_angle,
)
from .utils import safe_float


MORPHOLOGY_CLASSES = {
    "unresolved",
    "marginally_resolved",
    "resolved",
    "multi_gaussian_extended",
    "artifact_like",
    "unknown",
}


def _classification_config(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "beam_like_axis_ratio_tolerance": 0.35,
        "beam_like_size_tolerance": 0.35,
        "beam_like_pa_tolerance_deg": 20.0,
        "beam_like_score_unresolved": 0.70,
        "resolved_deconv_major_fraction": 0.45,
        "resolved_deconv_minor_fraction": 0.35,
        "marginal_deconv_major_fraction": 0.20,
        "intrinsic_resolved_fraction": 0.35,
        "intrinsic_marginal_fraction": 0.15,
        "low_snr_threshold": 5.0,
        "pa_weight_unresolved": 0.0,
        "pa_weight_beam_like": 0.0,
        "pa_weight_marginal": 0.35,
        "pa_weight_resolved": 1.0,
        "pa_weight_unknown": 0.15,
    }
    out = dict(defaults)
    out.update((config or {}).get("beam_aware_classification", {}) or {})
    return out


def _finite_positive(value: Any) -> bool:
    value = safe_float(value, float("nan"))
    return bool(np.isfinite(value) and value > 0)


def _row_value(row: pd.Series, names: list[str], default: float = float("nan")) -> float:
    for name in names:
        if name in row:
            value = safe_float(row.get(name), float("nan"))
            if np.isfinite(value):
                return float(value)
    return float(default)


def _snr_from_row(row: pd.Series) -> float:
    for name in ["peak_snr", "_peak_snr", "Peak_snr", "SNR", "snr", "peak_to_rms"]:
        if name in row:
            value = safe_float(row.get(name), float("nan"))
            if np.isfinite(value):
                return float(value)
    return float("nan")


def _deconv_axes(row: pd.Series) -> tuple[float, float, float]:
    major = _row_value(row, ["_dc_maj", "DC_Maj", "DC_MAJ", "dc_maj"])
    minor = _row_value(row, ["_dc_min", "DC_Min", "DC_MIN", "dc_min"])
    pa = _row_value(row, ["_dc_pa", "DC_PA", "dc_pa"])
    if not (_finite_positive(major) and _finite_positive(minor)):
        return float("nan"), float("nan"), float("nan")
    if minor > major:
        major, minor = minor, major
        pa = pa + 90.0 if np.isfinite(pa) else pa
    return float(major), float(minor), float(pa)


def _observed_axes(row: pd.Series, config: dict[str, Any] | None) -> tuple[float, float, float]:
    beam_major, beam_minor, _beam_pa = beam_axes_from_config(config)
    major = _row_value(row, ["_maj", "Maj", "MAJ", "maj"], beam_major)
    minor = _row_value(row, ["_min", "Min", "MIN", "min"], beam_minor)
    pa = _row_value(row, ["_pa", "PA", "pa"], float("nan"))
    if not _finite_positive(major):
        major = beam_major
    if not _finite_positive(minor):
        minor = beam_minor
    if minor > major:
        major, minor = minor, major
        pa = pa + 90.0 if np.isfinite(pa) else pa
    return float(major), float(minor), float(pa)


def _axis_ratio_score(obs_major: float, obs_minor: float, beam_major: float, beam_minor: float) -> float:
    obs_ratio = obs_major / max(obs_minor, 1e-6)
    beam_ratio = beam_major / max(beam_minor, 1e-6)
    if obs_ratio <= 0 or beam_ratio <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(obs_ratio / beam_ratio)) / 0.35))


def _size_score(obs_major: float, obs_minor: float, beam_major: float, beam_minor: float) -> float:
    major_score = np.exp(-abs(np.log(max(obs_major, 1e-6) / max(beam_major, 1e-6))) / 0.35)
    minor_score = np.exp(-abs(np.log(max(obs_minor, 1e-6) / max(beam_minor, 1e-6))) / 0.35)
    return float(np.sqrt(major_score * minor_score))


def _pa_pixel(pa_deg: float, config: dict[str, Any] | None) -> float:
    beam = (config or {}).get("beam", {}) or {}
    return sky_pa_to_pixel_angle(
        pa_deg,
        ra_axis_sign=float(beam.get("ra_axis_sign", 1.0) or 1.0),
        dec_axis_sign=float(beam.get("dec_axis_sign", 1.0) or 1.0),
    )


def beam_like_score(row: pd.Series, config: dict[str, Any] | None = None) -> float:
    """Return a 0-1 score for whether the observed Gaussian is beam-like."""

    beam_major, beam_minor, beam_pa = beam_axes_from_config(config)
    obs_major, obs_minor, obs_pa = _observed_axes(row, config)
    cfg = _classification_config(config)
    ratio_score = _axis_ratio_score(obs_major, obs_minor, beam_major, beam_minor)
    size = _size_score(obs_major, obs_minor, beam_major, beam_minor)
    if np.isfinite(obs_pa):
        pa_score = max(0.0, 1.0 - angle_delta_180(_pa_pixel(obs_pa, config), beam_pa) / max(float(cfg["beam_like_pa_tolerance_deg"]), 1e-6))
    else:
        pa_score = 0.5
    return float(np.clip(0.40 * ratio_score + 0.40 * size + 0.20 * pa_score, 0.0, 1.0))


def intrinsic_axes_from_component(row: pd.Series, config: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """Return intrinsic FWHM axes and image-plane PA when available.

    The preferred source is catalogue deconvolved axes.  If they are missing,
    the fallback subtracts the configured beam covariance from the observed
    covariance and clips negative eigenvalues to zero.
    """

    dc_major, dc_minor, dc_pa = _deconv_axes(row)
    if _finite_positive(dc_major) and _finite_positive(dc_minor):
        return float(dc_major), float(dc_minor), _pa_pixel(dc_pa, config) if np.isfinite(dc_pa) else float("nan")

    obs_major, obs_minor, obs_pa = _observed_axes(row, config)
    if not np.isfinite(obs_pa):
        return float("nan"), float("nan"), float("nan")
    beam = (config or {}).get("beam", {}) or {}
    obs_cov = gaussian_covariance(
        obs_major,
        obs_minor,
        obs_pa,
        ra_axis_sign=float(beam.get("ra_axis_sign", 1.0) or 1.0),
        dec_axis_sign=float(beam.get("dec_axis_sign", 1.0) or 1.0),
    )
    intrinsic = obs_cov - beam_covariance_from_config(config)
    intrinsic = 0.5 * (intrinsic + intrinsic.T)
    vals, vecs = np.linalg.eigh(intrinsic)
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    major = SIGMA_TO_FWHM * np.sqrt(max(vals[0], 0.0))
    minor = SIGMA_TO_FWHM * np.sqrt(max(vals[1], 0.0))
    pa_pixel = float(np.rad2deg(np.arctan2(vec[1], vec[0])) % 180.0) if major > 0 else float("nan")
    return float(major), float(minor), pa_pixel


def classify_gaussian_component(row: pd.Series, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify a single Gaussian using deconvolved and beam-subtracted shape."""

    cfg = _classification_config(config)
    beam_major, beam_minor, _beam_pa = beam_axes_from_config(config)
    obs_major, obs_minor, obs_pa = _observed_axes(row, config)
    if not (_finite_positive(obs_major) and _finite_positive(obs_minor)):
        return {
            "morphology_class": "artifact_like",
            "resolved_probability": 0.0,
            "resolved_significance": 0.0,
            "beam_like_score": 0.0,
            "classification_reason": "invalid_observed_axes",
        }

    snr = _snr_from_row(row)
    dc_major, dc_minor, _dc_pa = _deconv_axes(row)
    bscore = beam_like_score(row, config)
    resolved_significance = 0.0
    reason = ""
    morphology_class = "unknown"

    if np.isfinite(snr) and snr < float(cfg["low_snr_threshold"]):
        morphology_class = "unknown"
        reason = "low_snr_uncertain"
    elif _finite_positive(dc_major):
        major_frac = dc_major / max(beam_major, 1e-6)
        minor_frac = dc_minor / max(beam_minor, 1e-6) if _finite_positive(dc_minor) else 0.0
        # In the absence of catalogue axis errors, use conservative fractions
        # of the restoring beam as a significance proxy.
        resolved_significance = max(
            major_frac / max(float(cfg["marginal_deconv_major_fraction"]), 1e-6),
            minor_frac / max(float(cfg["marginal_deconv_major_fraction"]), 1e-6),
        )
        if (
            major_frac >= float(cfg["resolved_deconv_major_fraction"])
            or minor_frac >= float(cfg["resolved_deconv_minor_fraction"])
        ):
            morphology_class = "resolved"
            reason = "deconvolved_major_significant"
        elif major_frac >= float(cfg["marginal_deconv_major_fraction"]):
            morphology_class = "marginally_resolved"
            reason = "deconvolved_axes_marginal"
        else:
            morphology_class = "unresolved"
            reason = "deconvolved_axes_not_significant"
    else:
        intrinsic_major, intrinsic_minor, _pa = intrinsic_axes_from_component(row, config)
        major_frac = intrinsic_major / max(beam_major, 1e-6) if np.isfinite(intrinsic_major) else 0.0
        minor_frac = intrinsic_minor / max(beam_minor, 1e-6) if np.isfinite(intrinsic_minor) else 0.0
        resolved_significance = max(
            major_frac / max(float(cfg["intrinsic_marginal_fraction"]), 1e-6),
            minor_frac / max(float(cfg["intrinsic_marginal_fraction"]), 1e-6),
        )
        if major_frac >= float(cfg["intrinsic_resolved_fraction"]) or minor_frac >= float(cfg["intrinsic_resolved_fraction"]):
            morphology_class = "resolved"
            reason = "observed_covariance_exceeds_beam"
        elif major_frac >= float(cfg["intrinsic_marginal_fraction"]):
            morphology_class = "marginally_resolved"
            reason = "observed_covariance_marginally_exceeds_beam"
        elif bscore >= float(cfg["beam_like_score_unresolved"]):
            morphology_class = "unresolved"
            reason = "beam_like_within_errors"
        else:
            morphology_class = "unknown"
            reason = "no_reliable_deconvolved_axes"

    resolved_probability = float(np.clip(resolved_significance / 3.0, 0.0, 1.0))
    if morphology_class == "resolved":
        resolved_probability = max(resolved_probability, 0.75)
    elif morphology_class == "marginally_resolved":
        resolved_probability = float(np.clip(resolved_probability, 0.25, 0.75))
    elif morphology_class == "unresolved":
        resolved_probability = min(resolved_probability, 0.25)

    if morphology_class == "unresolved" and bscore >= float(cfg["beam_like_score_unresolved"]):
        reason = "beam_like_within_errors" if not reason else reason

    return {
        "morphology_class": morphology_class,
        "resolved_probability": resolved_probability,
        "resolved_significance": float(resolved_significance),
        "beam_like_score": float(bscore),
        "classification_reason": reason,
        "observed_major_arcsec": float(obs_major),
        "observed_minor_arcsec": float(obs_minor),
        "observed_pa_pixel_deg": _pa_pixel(obs_pa, config) if np.isfinite(obs_pa) else float("nan"),
    }


def add_morphology_columns(components: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a copy of components with beam-aware morphology columns."""

    if components is None or components.empty:
        return components.copy()
    records = [classify_gaussian_component(row, config) for _, row in components.iterrows()]
    out = components.copy()
    for key in [
        "morphology_class",
        "resolved_probability",
        "resolved_significance",
        "beam_like_score",
        "classification_reason",
        "observed_major_arcsec",
        "observed_minor_arcsec",
        "observed_pa_pixel_deg",
    ]:
        out[key] = [rec.get(key) for rec in records]
    return out


def is_unresolved_or_beam_like(component: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> bool:
    """Return True for unresolved or strongly beam-like components."""

    cfg = _classification_config(config)
    klass = str(component.get("morphology_class", "")).strip()
    score = safe_float(component.get("beam_like_score"), float("nan"))
    if not klass:
        rec = classify_gaussian_component(pd.Series(component), config)
        klass = str(rec["morphology_class"])
        score = safe_float(rec["beam_like_score"], 0.0)
    return bool(klass == "unresolved" or score >= float(cfg["beam_like_score_unresolved"]))


def effective_pa_weight(component: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> float:
    """Return the physical PA weight after suppressing beam-like observed PA."""

    cfg = _classification_config(config)
    klass = str(component.get("morphology_class", "")).strip()
    score = safe_float(component.get("beam_like_score"), float("nan"))
    if not klass:
        rec = classify_gaussian_component(pd.Series(component), config)
        klass = str(rec["morphology_class"])
        score = safe_float(rec["beam_like_score"], 0.0)
    if klass == "resolved":
        return float(cfg["pa_weight_resolved"])
    if klass == "marginally_resolved":
        return float(cfg["pa_weight_marginal"])
    if klass == "unresolved":
        return float(cfg["pa_weight_unresolved"])
    if np.isfinite(score) and score >= float(cfg["beam_like_score_unresolved"]):
        return float(cfg["pa_weight_beam_like"])
    return float(cfg["pa_weight_unknown"])


def effective_component_pa_pixel(component: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> float:
    """Return the best available component PA in image-plane coordinates."""

    klass = str(component.get("morphology_class", "")).strip()
    if klass in {"resolved", "marginally_resolved"}:
        dc_pa = _row_value(pd.Series(component), ["_dc_pa", "DC_PA", "dc_pa"])
        if np.isfinite(dc_pa):
            return _pa_pixel(dc_pa, config)
    pa = _row_value(pd.Series(component), ["_pa", "PA", "pa"])
    return _pa_pixel(pa, config) if np.isfinite(pa) else float("nan")
