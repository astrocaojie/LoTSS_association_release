"""Cutout-to-Gaussian matching helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .catalog import select_catalog_region
from .segmentation import labels_at_points
from .utils import infer_pixel_scale_arcsec


def match_gaussians_to_cutout(
    gaussians: pd.DataFrame,
    cutout: Any,
    segmentation: Any | None = None,
    pixel_scale_arcsec: float | None = None,
    preselect_margin_arcsec: float = 120.0,
) -> tuple[pd.DataFrame, str]:
    """Find Gaussian components that fall inside a cutout.

    Returns a normalized component DataFrame and the matching mode name.
    """

    height, width = cutout.image.shape
    pixel_scale = pixel_scale_arcsec or infer_pixel_scale_arcsec(cutout.wcs)
    radius_arcsec = 0.5 * np.hypot(width, height) * pixel_scale + preselect_margin_arcsec

    if cutout.wcs is not None and "_ra" in gaussians and "_dec" in gaussians:
        candidates = select_catalog_region(gaussians, cutout.ra, cutout.dec, radius_arcsec)
        if len(candidates) == 0:
            out = _empty_components(cutout.cutout_id)
            out.attrs["n_outside_after_projection"] = 0
            return out, "sky"
        x, y = cutout.wcs.celestial.world_to_pixel_values(
            candidates["_ra"].to_numpy(float),
            candidates["_dec"].to_numpy(float),
        )
        out = candidates.copy()
        out["x"] = x
        out["y"] = y
        keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        n_outside = int((~keep).sum())
        out = out.loc[keep].copy()
        out.attrs["n_outside_after_projection"] = n_outside
        mode = "sky"
    elif "_x" in gaussians and "_y" in gaussians and gaussians["_x"].notna().any():
        out = gaussians.copy()
        out["x"] = out["_x"].to_numpy(float)
        out["y"] = out["_y"].to_numpy(float)
        keep = (
            np.isfinite(out["x"].to_numpy(float))
            & np.isfinite(out["y"].to_numpy(float))
            & (out["x"].to_numpy(float) >= 0)
            & (out["x"].to_numpy(float) < width)
            & (out["y"].to_numpy(float) >= 0)
            & (out["y"].to_numpy(float) < height)
        )
        out = out.loc[keep].copy()
        out.attrs["n_outside_after_projection"] = int((~keep).sum())
        mode = "pixel"
    else:
        out = _empty_components(cutout.cutout_id)
        out.attrs["n_outside_after_projection"] = 0
        return out, "fallback_no_wcs"

    if len(out) == 0:
        out.attrs["n_outside_after_projection"] = int(out.attrs.get("n_outside_after_projection", 0))
        return _empty_components(cutout.cutout_id), mode

    n_outside = int(out.attrs.get("n_outside_after_projection", 0))
    out = out.reset_index(drop=True)
    out.attrs["n_outside_after_projection"] = n_outside
    out["cutout_id"] = cutout.cutout_id
    out["cutout_index"] = cutout.index
    out["component_index"] = np.arange(len(out), dtype=int)
    out["pixel_scale_arcsec"] = pixel_scale

    if segmentation is not None:
        labels = labels_at_points(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            out["x"].to_numpy(float),
            out["y"].to_numpy(float),
        )
        for key, values in labels.items():
            out[key] = values

    return out, mode


def _empty_components(cutout_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cutout_id",
            "cutout_index",
            "component_index",
            "_source_id",
            "_island_id",
            "_gaussian_id",
            "_ra",
            "_dec",
            "_total_flux",
            "_peak_flux",
            "_maj",
            "_min",
            "_pa",
            "x",
            "y",
            "pixel_scale_arcsec",
        ]
    ).assign(cutout_id=cutout_id)
