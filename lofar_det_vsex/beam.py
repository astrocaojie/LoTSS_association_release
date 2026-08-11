"""Elliptical restoring-beam geometry utilities.

All public functions in this module use arcsec for lengths and degrees for
angles unless the name explicitly says otherwise.  Angles are measured in the
image pixel plane from +x toward +y, modulo 180 degrees, after converting sky
position angles when needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
SIGMA_TO_FWHM = 1.0 / FWHM_TO_SIGMA


def normalize_angle_180(angle_deg: float) -> float:
    """Return an undirected angle in degrees in the half-open range [0, 180)."""

    if not np.isfinite(angle_deg):
        return float("nan")
    return float(angle_deg % 180.0)


def angle_delta_180(a_deg: float, b_deg: float) -> float:
    """Smallest separation between two undirected angles in degrees."""

    if not np.isfinite(a_deg) or not np.isfinite(b_deg):
        return 90.0
    return float(abs((a_deg - b_deg + 90.0) % 180.0 - 90.0))


def sky_pa_to_pixel_angle(
    pa_deg: float,
    ra_axis_sign: float = 1.0,
    dec_axis_sign: float = 1.0,
) -> float:
    """Convert sky PA to an image-plane angle.

    Sky PA is measured east of north.  The output angle is measured from +x to
    +y in the pixel plane.  ``ra_axis_sign`` is +1 when increasing pixel x is
    increasing RA/east and -1 when increasing pixel x is decreasing RA/west.
    ``dec_axis_sign`` is +1 when increasing pixel y is north and -1 when it is
    south.  The default preserves the historical project convention where
    ``pixel_angle = 90 - PA`` modulo 180.
    """

    if not np.isfinite(pa_deg):
        return float("nan")
    pa = np.deg2rad(float(pa_deg))
    east = np.sin(pa)
    north = np.cos(pa)
    x = float(ra_axis_sign) * east
    y = float(dec_axis_sign) * north
    return normalize_angle_180(float(np.rad2deg(np.arctan2(y, x))))


def _beam_frame(config: dict[str, Any] | None = None) -> tuple[float, float]:
    beam = (config or {}).get("beam", {}) or {}
    ra_axis_sign = float(beam.get("ra_axis_sign", 1.0) or 1.0)
    dec_axis_sign = float(beam.get("dec_axis_sign", 1.0) or 1.0)
    return ra_axis_sign, dec_axis_sign


def beam_pixel_pa_from_config(config: dict[str, Any] | None = None) -> float:
    """Return restoring-beam major-axis PA in image-plane coordinates."""

    beam = (config or {}).get("beam", {}) or {}
    if "pixel_pa_deg" in beam and beam.get("pixel_pa_deg") is not None:
        return normalize_angle_180(float(beam.get("pixel_pa_deg")))
    pa = float(beam.get("pa_deg", 0.0) or 0.0)
    ra_axis_sign, dec_axis_sign = _beam_frame(config)
    return sky_pa_to_pixel_angle(pa, ra_axis_sign=ra_axis_sign, dec_axis_sign=dec_axis_sign)


def _rotation(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def covariance_from_fwhm(
    major_arcsec: float,
    minor_arcsec: float,
    pixel_pa_deg: float,
) -> np.ndarray:
    """Build a 2D covariance matrix from FWHM axes in image coordinates."""

    major = float(major_arcsec)
    minor = float(minor_arcsec)
    if not np.isfinite(major) or major <= 0:
        raise ValueError(f"major_arcsec must be positive, got {major_arcsec!r}")
    if not np.isfinite(minor) or minor <= 0:
        raise ValueError(f"minor_arcsec must be positive, got {minor_arcsec!r}")
    if minor > major:
        major, minor = minor, major
        pixel_pa_deg = float(pixel_pa_deg) + 90.0
    sigma_major = major * FWHM_TO_SIGMA
    sigma_minor = minor * FWHM_TO_SIGMA
    rot = _rotation(normalize_angle_180(pixel_pa_deg))
    diag = np.diag([sigma_major * sigma_major, sigma_minor * sigma_minor])
    return rot @ diag @ rot.T


def beam_covariance(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    bpa_deg: float,
    *,
    ra_axis_sign: float = 1.0,
    dec_axis_sign: float = 1.0,
    pixel_pa_deg: float | None = None,
) -> np.ndarray:
    """Return the restoring-beam covariance matrix in arcsec squared.

    ``bpa_deg`` is the FITS/sky beam position angle in degrees east of north.
    The optional axis-sign parameters encode the local WCS parity before the
    covariance is expressed in image x/y axes.
    """

    angle = (
        normalize_angle_180(float(pixel_pa_deg))
        if pixel_pa_deg is not None
        else sky_pa_to_pixel_angle(
            float(bpa_deg),
            ra_axis_sign=ra_axis_sign,
            dec_axis_sign=dec_axis_sign,
        )
    )
    return covariance_from_fwhm(float(bmaj_arcsec), float(bmin_arcsec), angle)


def gaussian_covariance(
    major_arcsec: float,
    minor_arcsec: float,
    pa_deg: float,
    *,
    pa_is_pixel: bool = False,
    ra_axis_sign: float = 1.0,
    dec_axis_sign: float = 1.0,
) -> np.ndarray:
    """Return a Gaussian covariance matrix in arcsec squared."""

    angle = normalize_angle_180(pa_deg) if pa_is_pixel else sky_pa_to_pixel_angle(pa_deg, ra_axis_sign, dec_axis_sign)
    return covariance_from_fwhm(float(major_arcsec), float(minor_arcsec), angle)


def beam_covariance_from_config(config: dict[str, Any] | None = None) -> np.ndarray:
    """Return the configured beam covariance in arcsec squared."""

    beam = (config or {}).get("beam", {}) or {}
    major = float(beam.get("major_arcsec", 6.0) or 6.0)
    minor = float(beam.get("minor_arcsec", major) or major)
    pa = float(beam.get("pa_deg", 0.0) or 0.0)
    ra_axis_sign, dec_axis_sign = _beam_frame(config)
    pixel_pa = beam.get("pixel_pa_deg")
    return beam_covariance(
        major,
        minor,
        pa,
        ra_axis_sign=ra_axis_sign,
        dec_axis_sign=dec_axis_sign,
        pixel_pa_deg=None if pixel_pa is None else float(pixel_pa),
    )


def beam_axes_from_config(config: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """Return ``(major_arcsec, minor_arcsec, pixel_pa_deg)`` from config."""

    beam = (config or {}).get("beam", {}) or {}
    major = float(beam.get("major_arcsec", 6.0) or 6.0)
    minor = float(beam.get("minor_arcsec", major) or major)
    angle = beam_pixel_pa_from_config(config)
    if minor > major:
        major, minor = minor, major
        angle = normalize_angle_180(angle + 90.0)
    return float(major), float(minor), float(angle)


def direction_unit(angle_deg: float) -> np.ndarray:
    """Return a unit vector for an image-plane angle."""

    theta = np.deg2rad(float(angle_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=float)


def direction_angle_from_delta(dx: float, dy: float) -> float:
    """Return the image-plane direction angle of a pixel or arcsec delta."""

    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx) < 1e-12 and abs(dy) < 1e-12):
        return float("nan")
    return normalize_angle_180(float(np.rad2deg(np.arctan2(dy, dx))))


def projected_beam_fwhm(direction_angle: float, beam_cov: np.ndarray) -> float:
    """Return beam FWHM in arcsec projected along an image-plane direction."""

    if not np.isfinite(direction_angle):
        vals = np.linalg.eigvalsh(np.asarray(beam_cov, dtype=float))
        return float(SIGMA_TO_FWHM * np.sqrt(max(float(np.nanmean(vals)), 1e-12)))
    unit = direction_unit(direction_angle)
    sigma2 = float(unit.T @ np.asarray(beam_cov, dtype=float) @ unit)
    return float(SIGMA_TO_FWHM * np.sqrt(max(sigma2, 1e-12)))


def elliptical_beam_distance(delta_xy_arcsec: np.ndarray | tuple[float, float], beam_cov: np.ndarray) -> float:
    """Return separation in direction-dependent beam FWHM units.

    For a displacement along the beam major axis this equals
    ``distance_arcsec / BMAJ``; along the minor axis it equals
    ``distance_arcsec / BMIN``.
    """

    delta = np.asarray(delta_xy_arcsec, dtype=float).reshape(2)
    if not np.isfinite(delta).all():
        return float("nan")
    if np.hypot(delta[0], delta[1]) <= 0:
        return 0.0
    inv = np.linalg.pinv(np.asarray(beam_cov, dtype=float))
    sigma_units = float(np.sqrt(max(float(delta.T @ inv @ delta), 0.0)))
    return float(sigma_units / SIGMA_TO_FWHM)


def projected_beam_distance_from_delta(
    dx_pix: float,
    dy_pix: float,
    pixel_scale_arcsec: float,
    config: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Return ``(distance_beam, projected_fwhm_arcsec, direction_angle_deg)``."""

    dx_arcsec = float(dx_pix) * float(pixel_scale_arcsec)
    dy_arcsec = float(dy_pix) * float(pixel_scale_arcsec)
    direction = direction_angle_from_delta(dx_pix, dy_pix)
    cov = beam_covariance_from_config(config)
    projected = projected_beam_fwhm(direction, cov)
    distance = elliptical_beam_distance((dx_arcsec, dy_arcsec), cov)
    return float(distance), float(projected), float(direction)


def beam_axis_components(
    dx_pix: float,
    dy_pix: float,
    pixel_scale_arcsec: float,
    config: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Return parallel/perpendicular separations relative to beam major axis.

    The first two values are absolute arcsec separations parallel and
    perpendicular to the configured beam major axis.  The third is the pair
    angle relative to the beam major axis in degrees.
    """

    _major, _minor, beam_angle = beam_axes_from_config(config)
    delta = np.asarray([float(dx_pix) * float(pixel_scale_arcsec), float(dy_pix) * float(pixel_scale_arcsec)])
    major_unit = direction_unit(beam_angle)
    minor_unit = direction_unit(beam_angle + 90.0)
    pair_angle = direction_angle_from_delta(dx_pix, dy_pix)
    return (
        float(abs(delta @ major_unit)),
        float(abs(delta @ minor_unit)),
        angle_delta_180(pair_angle, beam_angle) if np.isfinite(pair_angle) else float("nan"),
    )


def beam_area_arcsec2(
    bmaj_arcsec: float | None = None,
    bmin_arcsec: float | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    """Return Gaussian beam area in square arcsec."""

    if config is not None:
        beam = (config or {}).get("beam", {}) or {}
        bmaj_arcsec = float(beam.get("major_arcsec", 6.0) or 6.0)
        bmin_arcsec = float(beam.get("minor_arcsec", bmaj_arcsec) or bmaj_arcsec)
    major = float(6.0 if bmaj_arcsec is None else bmaj_arcsec)
    minor = float(major if bmin_arcsec is None else bmin_arcsec)
    return float(np.pi * major * minor / (4.0 * np.log(2.0)))
