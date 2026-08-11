"""FITS tile and PyBDSF Gaussian loading for baseline demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from lofar_det_vsex.catalog import normalized_gaussian_dataframe, read_gaussian_catalog
from lofar_det_vsex.segmentation import build_snr_map, labels_at_points, segment_snr_map
from lofar_det_vsex.utils import get_logger, robust_mad_rms, safe_float, write_dataframe


@dataclass
class TileData:
    """Container for one tile/demo region."""

    cutout: Any
    image: np.ndarray
    rms: np.ndarray | float | None
    wcs: WCS | None
    components: pd.DataFrame
    segmentation: Any
    config: dict[str, Any]
    quality_stats: dict[str, Any]


def _select_image_hdu(hdul: fits.HDUList, hdu: int | str | None = None) -> Any:
    if hdu is not None:
        return hdul[hdu]
    for item in hdul:
        if getattr(item, "data", None) is not None:
            arr = np.asarray(item.data)
            if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
                return item
    raise ValueError("No numeric image HDU found")


def _two_d_plane(data: np.ndarray) -> np.ndarray:
    """Return a 2-D Stokes/frequency plane from a FITS image array."""

    out = np.asarray(data)
    while out.ndim > 2:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"Expected 2-D image plane, got shape {np.asarray(data).shape}")
    return out


def _read_fits_image(path: str | Path, hdu: int | str | None = None) -> tuple[np.ndarray, fits.Header]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing FITS image: {path}")
    with fits.open(path, memmap=True) as hdul:
        selected = _select_image_hdu(hdul, hdu)
        data = np.asarray(_two_d_plane(selected.data), dtype=np.float32)
        header = selected.header.copy()
    return data, header


def _read_fits_header_shape(path: str | Path, hdu: int | str | None = None) -> tuple[fits.Header, tuple[int, int], int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing FITS image: {path}")
    with fits.open(path, memmap=True) as hdul:
        selected = _select_image_hdu(hdul, hdu)
        shape = _two_d_plane(selected.data).shape
        header = selected.header.copy()
        hdu_index = hdul.index_of(selected)
    return header, tuple(int(v) for v in shape), int(hdu_index)


def _read_fits_image_region(
    path: str | Path,
    bounds: tuple[int, int, int, int],
    hdu: int | str | None = None,
) -> tuple[np.ndarray, fits.Header]:
    """Read a 2-D FITS region and shift WCS CRPIX into local coordinates."""

    xmin, xmax, ymin, ymax = bounds
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        selected = _select_image_hdu(hdul, hdu)
        data = _two_d_plane(selected.data)
        region = np.asarray(data[ymin:ymax, xmin:xmax], dtype=np.float32)
        header = selected.header.copy()
    for axis, offset in ((1, xmin), (2, ymin)):
        key = f"CRPIX{axis}"
        if key in header:
            header[key] = float(header[key]) - float(offset)
    return region, header


def _beam_from_header(header: fits.Header, config: dict[str, Any]) -> tuple[float, float, float]:
    beam_cfg = config.get("beam", {}) or {}
    major = safe_float(header.get("BMAJ"), np.nan) * 3600.0
    minor = safe_float(header.get("BMIN"), np.nan) * 3600.0
    pa = safe_float(header.get("BPA"), np.nan)
    if not np.isfinite(major) or major <= 0:
        major = safe_float(beam_cfg.get("major_arcsec"), 6.0)
    if not np.isfinite(minor) or minor <= 0:
        minor = safe_float(beam_cfg.get("minor_arcsec"), major)
    if not np.isfinite(pa):
        pa = safe_float(beam_cfg.get("pa_deg"), 0.0)
    return float(major), float(minor), float(pa)


def _pixel_scale_arcsec(wcs: WCS | None, config: dict[str, Any]) -> float:
    default = safe_float(config.get("pixel_scale_arcsec"), 1.5)
    if wcs is None:
        return default
    try:
        scales = proj_plane_pixel_scales(wcs.celestial)
        value = float(np.mean(np.abs(scales)) * 3600.0)
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return default


def _optional_rms(config: dict[str, Any], image: np.ndarray, catalogue: pd.DataFrame | None = None) -> np.ndarray | float | None:
    rms_path = config.get("rms_map")
    if rms_path:
        path = Path(str(rms_path))
        if path.exists():
            rms, _header = _read_fits_image(path, config.get("rms_hdu"))
            if rms.shape != image.shape:
                raise ValueError(f"RMS map shape {rms.shape} does not match image shape {image.shape}")
            return rms
        raise FileNotFoundError(f"Configured rms_map does not exist: {path}")
    if bool(config.get("use_catalogue_rms_for_snr", True)) and catalogue is not None:
        for col in ["local_rms", "Isl_rms", "isl_rms", "Local_rms", "RMS", "rms"]:
            if col in catalogue:
                values = pd.to_numeric(catalogue[col], errors="coerce")
                finite = values[np.isfinite(values) & (values > 0)]
                if len(finite):
                    return float(finite.median())
    mode = str(config.get("rms_mode", "mad"))
    if mode in {"mad", "std"}:
        return robust_mad_rms(image) if mode == "mad" else float(np.nanstd(image))
    return None


def _optional_rms_region(
    config: dict[str, Any],
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    catalogue: pd.DataFrame | None = None,
) -> np.ndarray | float | None:
    rms_path = config.get("rms_map")
    if rms_path:
        path = Path(str(rms_path))
        if path.exists():
            rms, _header = _read_fits_image_region(path, bounds, config.get("rms_hdu"))
            if rms.shape != image.shape:
                raise ValueError(f"RMS map region shape {rms.shape} does not match image shape {image.shape}")
            return rms
        raise FileNotFoundError(f"Configured rms_map does not exist: {path}")
    return _optional_rms({**config, "rms_map": None}, image, catalogue)


def _region_bounds(config: dict[str, Any], image_shape: tuple[int, int], wcs: WCS | None) -> tuple[int, int, int, int]:
    region = config.get("demo_region", {}) or {}
    mode = str(region.get("mode", "full_tile"))
    enabled = bool(region.get("enabled", False))
    height, width = image_shape
    if not enabled or mode == "full_tile":
        return 0, width, 0, height
    if mode == "pixel_box":
        xmin = int(region.get("xmin", 0))
        xmax = int(region.get("xmax", width))
        ymin = int(region.get("ymin", 0))
        ymax = int(region.get("ymax", height))
    elif mode == "sky_box":
        if wcs is None:
            raise ValueError("demo_region.mode=sky_box requires FITS WCS")
        keys = ["ra_min", "ra_max", "dec_min", "dec_max"]
        if not all(key in region for key in keys):
            raise ValueError(f"sky_box region requires {keys}")
        ras = np.asarray([float(region["ra_min"]), float(region["ra_min"]), float(region["ra_max"]), float(region["ra_max"])])
        decs = np.asarray([float(region["dec_min"]), float(region["dec_max"]), float(region["dec_min"]), float(region["dec_max"])])
        xs, ys = wcs.celestial.world_to_pixel_values(ras, decs)
        xmin = int(np.floor(np.nanmin(xs)))
        xmax = int(np.ceil(np.nanmax(xs)))
        ymin = int(np.floor(np.nanmin(ys)))
        ymax = int(np.ceil(np.nanmax(ys)))
    else:
        raise ValueError(f"Unsupported demo_region.mode: {mode}")
    padding = int(region.get("padding_pixels", 0) or 0)
    xmin = max(0, xmin - padding)
    xmax = min(width, xmax + padding)
    ymin = max(0, ymin - padding)
    ymax = min(height, ymax + padding)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Empty demo region after clipping: {(xmin, xmax, ymin, ymax)}")
    return xmin, xmax, ymin, ymax


def _component_local_rms(
    rows: pd.DataFrame,
    rms: np.ndarray | float | None,
    fallback: float,
) -> pd.Series:
    if isinstance(rms, np.ndarray):
        values = np.full(len(rows), fallback, dtype=float)
        x = np.rint(pd.to_numeric(rows["x"], errors="coerce")).astype("Int64")
        y = np.rint(pd.to_numeric(rows["y"], errors="coerce")).astype("Int64")
        height, width = rms.shape
        valid = x.notna() & y.notna() & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if valid.any():
            values[valid.to_numpy()] = rms[y[valid].astype(int).to_numpy(), x[valid].astype(int).to_numpy()]
        values[~np.isfinite(values) | (values <= 0)] = fallback
        return pd.Series(values, index=rows.index)
    for col in ["Isl_rms", "isl_rms", "Local_rms", "local_rms", "RMS", "rms"]:
        if col in rows:
            values = pd.to_numeric(rows[col], errors="coerce")
            values = values.where(np.isfinite(values) & (values > 0), fallback)
            return values
    if rms is not None and np.isfinite(float(rms)) and float(rms) > 0:
        return pd.Series(float(rms), index=rows.index)
    return pd.Series(fallback, index=rows.index)


def load_tile_demo(config: dict[str, Any]) -> TileData:
    """Load image, catalogue, demo region, segmentation, and clean components."""

    logger = get_logger()
    image_path = config.get("radio_image")
    catalogue_path = config.get("gaussian_catalogue")
    if not image_path:
        raise ValueError("Config must set radio_image for the tile baseline demo")
    if not catalogue_path:
        raise ValueError("Config must set gaussian_catalogue for the tile baseline demo")

    full_header, full_shape, _hdu_index = _read_fits_header_shape(image_path, config.get("radio_hdu"))
    full_wcs = None
    try:
        full_wcs = WCS(full_header).celestial
        if not full_wcs.has_celestial:
            full_wcs = None
    except Exception:
        full_wcs = None

    beam_major, beam_minor, beam_pa = _beam_from_header(full_header, config)
    config = dict(config)
    config["beam"] = {
        **(config.get("beam", {}) or {}),
        "major_arcsec": beam_major,
        "minor_arcsec": beam_minor,
        "pa_deg": beam_pa,
    }
    pixel_scale = _pixel_scale_arcsec(full_wcs, config)

    table = read_gaussian_catalog(catalogue_path)
    raw_rows = len(table)
    gaussians, columns = normalized_gaussian_dataframe(table)

    height, width = full_shape
    if full_wcs is not None and "_ra" in gaussians and "_dec" in gaussians:
        x, y = full_wcs.world_to_pixel_values(gaussians["_ra"].to_numpy(float), gaussians["_dec"].to_numpy(float))
        gaussians["x"] = x
        gaussians["y"] = y
        match_mode = "sky_wcs"
    elif "_x" in gaussians and "_y" in gaussians:
        gaussians["x"] = pd.to_numeric(gaussians["_x"], errors="coerce")
        gaussians["y"] = pd.to_numeric(gaussians["_y"], errors="coerce")
        match_mode = "catalog_pixels"
    else:
        raise ValueError("Catalogue lacks RA/DEC with WCS and lacks pixel x/y columns")

    xmin, xmax, ymin, ymax = _region_bounds(config, full_shape, full_wcs)
    image, header = _read_fits_image_region(image_path, (xmin, xmax, ymin, ymax), config.get("radio_hdu"))
    wcs = None
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            wcs = None
    except Exception:
        wcs = None
    rms = _optional_rms_region(config, image, (xmin, xmax, ymin, ymax), gaussians)
    fallback_rms = robust_mad_rms(image)
    if rms is not None and np.asarray(rms).ndim == 0 and np.isfinite(float(rms)) and float(rms) > 0:
        fallback_rms = float(rms)

    keep_region = (
        np.isfinite(gaussians["x"].to_numpy(float))
        & np.isfinite(gaussians["y"].to_numpy(float))
        & (gaussians["x"].to_numpy(float) >= xmin)
        & (gaussians["x"].to_numpy(float) < xmax)
        & (gaussians["y"].to_numpy(float) >= ymin)
        & (gaussians["y"].to_numpy(float) < ymax)
    )
    within_region = int(np.count_nonzero(keep_region))
    outside_image_mask = (
        ~np.isfinite(gaussians["x"].to_numpy(float))
        | ~np.isfinite(gaussians["y"].to_numpy(float))
        | (gaussians["x"].to_numpy(float) < 0)
        | (gaussians["x"].to_numpy(float) >= width)
        | (gaussians["y"].to_numpy(float) < 0)
        | (gaussians["y"].to_numpy(float) >= height)
    )
    components = gaussians.loc[keep_region].copy()
    components["x_full"] = pd.to_numeric(components["x"], errors="coerce")
    components["y_full"] = pd.to_numeric(components["y"], errors="coerce")
    components["x"] = components["x_full"] - float(xmin)
    components["y"] = components["y_full"] - float(ymin)
    before_dup = len(components)
    components = components.drop_duplicates(subset=["_gaussian_id"], keep="first").copy()
    removed_duplicates = before_dup - len(components)

    finite_geom = (
        np.isfinite(pd.to_numeric(components["x"], errors="coerce"))
        & np.isfinite(pd.to_numeric(components["y"], errors="coerce"))
        & np.isfinite(pd.to_numeric(components["_ra"], errors="coerce"))
        & np.isfinite(pd.to_numeric(components["_dec"], errors="coerce"))
    )
    for col in ["_maj", "_min"]:
        if col in components:
            values = pd.to_numeric(components[col], errors="coerce")
            finite_geom &= values.fillna(beam_major).ge(0)
    invalid_geometry = int((~finite_geom).sum())
    components = components.loc[finite_geom].copy().reset_index(drop=True)
    max_components = config.get("max_components")
    if max_components is not None:
        max_n = int(max_components)
        if max_n > 0 and len(components) > max_n:
            components = components.sample(n=max_n, random_state=int(config.get("random_seed", 42))).sort_index().reset_index(drop=True)

    components["component_index"] = np.arange(len(components), dtype=int)
    components["component_id"] = components["_gaussian_id"].astype(str)
    components["source_id"] = components["_source_id"]
    components["island_id"] = components["_island_id"]
    components["ra"] = pd.to_numeric(components["_ra"], errors="coerce")
    components["dec"] = pd.to_numeric(components["_dec"], errors="coerce")
    components["x_pixel"] = pd.to_numeric(components["x"], errors="coerce")
    components["y_pixel"] = pd.to_numeric(components["y"], errors="coerce")
    components["x_pixel_full"] = pd.to_numeric(components["x_full"], errors="coerce")
    components["y_pixel_full"] = pd.to_numeric(components["y_full"], errors="coerce")
    components["peak_flux"] = pd.to_numeric(components["_peak_flux"], errors="coerce")
    components["total_flux"] = pd.to_numeric(components["_total_flux"], errors="coerce")
    components["major_axis"] = pd.to_numeric(components["_dc_maj"], errors="coerce").where(
        pd.to_numeric(components["_dc_maj"], errors="coerce") > 0,
        pd.to_numeric(components["_maj"], errors="coerce"),
    )
    components["minor_axis"] = pd.to_numeric(components["_dc_min"], errors="coerce").where(
        pd.to_numeric(components["_dc_min"], errors="coerce") > 0,
        pd.to_numeric(components["_min"], errors="coerce"),
    )
    components["position_angle"] = pd.to_numeric(components["_dc_pa"], errors="coerce").where(
        np.isfinite(pd.to_numeric(components["_dc_pa"], errors="coerce")),
        pd.to_numeric(components["_pa"], errors="coerce"),
    )
    components["pixel_scale_arcsec"] = pixel_scale
    components["beam_major"] = beam_major
    components["beam_minor"] = beam_minor
    components["beam_pa"] = beam_pa
    components["local_rms"] = _component_local_rms(components, rms, fallback_rms)
    components["peak_snr"] = components["peak_flux"] / components["local_rms"].replace(0, np.nan)
    components["cutout_id"] = str(config.get("tile_id", Path(str(image_path)).stem))
    components["cutout_index"] = 0

    snr, mean_value, rms_used = build_snr_map(
        image,
        rms=rms,
        mean=None,
        mean_mode=str(config.get("mean_mode", "median")),
        rms_mode=str(config.get("rms_mode", "mad")),
        smooth_before_segmentation=bool(config.get("smooth_before_segmentation", True)),
        gaussian_smooth_sigma_pix=float(config.get("gaussian_smooth_sigma_pix", 1.0)),
    )
    segmentation = segment_snr_map(
        snr,
        thresholds=config.get("snr_thresholds", [5.0, 4.0, 3.0, 2.5, 2.0]),
        min_mask_area_pix=int(config.get("min_mask_area_pix", 20)),
        connectivity=int(config.get("connectivity", 2)),
        binary_opening=bool(config.get("binary_opening", False)),
        binary_closing=bool(config.get("binary_closing", True)),
    )
    labels = labels_at_points(
        segmentation.labels_by_threshold,
        segmentation.thresholds,
        components["x"].to_numpy(float),
        components["y"].to_numpy(float),
    )
    for key, values in labels.items():
        components[key] = values

    clean_columns = [
        "component_id",
        "source_id",
        "island_id",
        "ra",
        "dec",
        "x_pixel",
        "y_pixel",
        "peak_flux",
        "total_flux",
        "peak_snr",
        "major_axis",
        "minor_axis",
        "position_angle",
        "local_rms",
        "beam_major",
        "beam_minor",
        "beam_pa",
        "component_index",
        "cutout_id",
        "pixel_scale_arcsec",
        "x_pixel_full",
        "y_pixel_full",
    ]
    out_dir = Path(config.get("output_dir", "baseline_demo/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(components.reindex(columns=clean_columns), out_dir / "components_clean.parquet")
    components.reindex(columns=clean_columns).to_csv(out_dir / "components_clean.csv", index=False)

    quality_stats = {
        "number_raw_catalogue_rows": int(raw_rows),
        "number_within_tile_demo_region": int(within_region),
        "number_removed_as_duplicates": int(removed_duplicates),
        "number_outside_image": int(np.count_nonzero(outside_image_mask)),
        "number_with_invalid_geometry": int(invalid_geometry),
        "number_retained": int(len(components)),
        "match_mode": match_mode,
        "radio_image": str(image_path),
        "gaussian_catalogue": str(catalogue_path),
        "rms_source": "rms_map" if isinstance(rms, np.ndarray) else ("scalar_or_catalogue" if rms is not None else "estimated"),
        "mean_used": "map" if hasattr(mean_value, "shape") else mean_value,
        "rms_used": "map" if hasattr(rms_used, "shape") else rms_used,
        "beam_major_arcsec": beam_major,
        "beam_minor_arcsec": beam_minor,
        "beam_pa_deg": beam_pa,
        "pixel_scale_arcsec": pixel_scale,
        "demo_region_bounds": [xmin, xmax, ymin, ymax],
        "full_image_shape": [height, width],
        "analysis_image_shape": list(image.shape),
        "catalogue_source_id_column": columns.source_id,
        "catalogue_island_id_column": columns.island_id,
        "catalogue_gaussian_id_column": columns.gaussian_id,
        "major_minor_axis_basis": "deconvolved when positive, otherwise observed; arcsec",
        "flux_unit": "catalogue native units, typically Jy for LoTSS PyBDSF",
        "beam_unit": "arcsec",
    }
    pd.DataFrame([quality_stats]).to_csv(out_dir / "data_quality_stats.csv", index=False)
    logger.info("Retained %d/%d Gaussian components for tile demo", len(components), raw_rows)

    cutout = SimpleNamespace(
        cutout_id=str(config.get("tile_id", Path(str(image_path)).stem)),
        image=image,
        rms=rms,
        mean=None,
        ra=None,
        dec=None,
        header=header,
        wcs=wcs,
        index=0,
        metadata={"image_path": str(image_path), "region_bounds": [xmin, xmax, ymin, ymax]},
    )
    return TileData(
        cutout=cutout,
        image=image,
        rms=rms,
        wcs=wcs,
        components=components,
        segmentation=segmentation,
        config=config,
        quality_stats=quality_stats,
    )


def inspect_tile_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight input inventory without loading all arrays into outputs."""

    image_path = config.get("radio_image")
    catalogue_path = config.get("gaussian_catalogue")
    inventory: dict[str, Any] = {
        "radio_image": image_path,
        "gaussian_catalogue": catalogue_path,
        "rms_map": config.get("rms_map"),
    }
    if image_path and Path(str(image_path)).exists():
        image, header = _read_fits_image(image_path, config.get("radio_hdu"))
        wcs = WCS(header).celestial
        beam = _beam_from_header(header, config)
        inventory.update(
            {
                "radio_image_format": "FITS",
                "radio_image_shape": tuple(image.shape),
                "has_wcs": bool(wcs.has_celestial),
                "beam_major_arcsec": beam[0],
                "beam_minor_arcsec": beam[1],
                "beam_pa_deg": beam[2],
                "pixel_scale_arcsec": _pixel_scale_arcsec(wcs, config),
            }
        )
    if catalogue_path and Path(str(catalogue_path)).exists():
        table = Table.read(catalogue_path)
        _df, columns = normalized_gaussian_dataframe(table[: min(len(table), 5)])
        inventory.update(
            {
                "gaussian_catalogue_format": Path(str(catalogue_path)).suffix,
                "gaussian_catalogue_rows": len(table),
                "has_source_id": columns.source_id is not None,
                "has_island_id": columns.island_id is not None,
                "source_id_column": columns.source_id,
                "island_id_column": columns.island_id,
                "gaussian_id_column": columns.gaussian_id,
            }
        )
    return inventory
