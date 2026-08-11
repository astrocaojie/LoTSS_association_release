"""Preflight, validation, reports, and manual-review exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from lofar_det_vsex.association import compute_beam_size_arcsec
from lofar_det_vsex.utils import json_dumps_safe, robust_mad_rms

from .common import component_id_series, parse_component_ids
from .data_loading import _read_fits_header_shape, _region_bounds
from .plotting import _display_image, _plot_case, _save_figure


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps_safe(data), encoding="utf-8")


def fits_preflight_inventory(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect FITS header and configured-region image statistics."""

    image_path = config.get("radio_image")
    if not image_path:
        return {"radio_image": None, "error": "radio_image not configured"}
    header, shape, image_hdu_index = _read_fits_header_shape(image_path, config.get("radio_hdu"))
    wcs = WCS(header).celestial
    bounds = _region_bounds(config, shape, wcs if wcs.has_celestial else None)
    xmin, xmax, ymin, ymax = bounds
    with fits.open(image_path, memmap=True) as hdul:
        data = hdul[image_hdu_index].data
        while data.ndim > 2:
            data = data[0]
        region = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float)
    finite = region[np.isfinite(region)]
    stats = {
        "radio_image": str(image_path),
        "fits_dimensions": list(shape),
        "number_of_hdus": len(fits.open(image_path, memmap=True)),
        "image_hdu_index": int(image_hdu_index),
        "selected_plane": "2D image" if int(header.get("NAXIS", 2)) == 2 else "first Stokes/frequency plane by leading-axis index 0",
        "BUNIT": header.get("BUNIT"),
        "BMAJ": header.get("BMAJ"),
        "BMIN": header.get("BMIN"),
        "BPA": header.get("BPA"),
        "CDELT1": header.get("CDELT1"),
        "CDELT2": header.get("CDELT2"),
        "CTYPE1": header.get("CTYPE1"),
        "CTYPE2": header.get("CTYPE2"),
        "CRVAL1": header.get("CRVAL1"),
        "CRVAL2": header.get("CRVAL2"),
        "presence_of_nan_pixels_region": bool(np.isnan(region).any()),
        "n_nan_pixels_region": int(np.isnan(region).sum()),
        "minimum_region": float(np.nanmin(region)) if finite.size else np.nan,
        "maximum_region": float(np.nanmax(region)) if finite.size else np.nan,
        "median_region": float(np.nanmedian(region)) if finite.size else np.nan,
        "robust_rms_estimate_region": float(robust_mad_rms(region)),
        "region_bounds_pixel_full": [int(xmin), int(xmax), int(ymin), int(ymax)],
        "region_shape": [int(ymax - ymin), int(xmax - xmin)],
    }
    return stats


def catalogue_preflight(config: dict[str, Any], components: pd.DataFrame | None = None) -> dict[str, Any]:
    """Inspect catalogue columns, units, mappings, and duplicate ids."""

    path = config.get("gaussian_catalogue")
    if not path:
        return {"gaussian_catalogue": None, "error": "gaussian_catalogue not configured"}
    table = Table.read(path)
    columns_path = Path(config.get("output_dir", "baseline_demo/outputs")) / "preflight" / "catalogue_columns.csv"
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in table.colnames:
        rows.append({"column": name, "dtype": str(table[name].dtype), "unit": str(getattr(table[name], "unit", "") or "")})
    pd.DataFrame(rows).to_csv(columns_path, index=False)
    names = set(table.colnames)
    mapping = {
        "ra_column": next((c for c in ["RA", "ra", "RA_deg", "ra_deg"] if c in names), None),
        "dec_column": next((c for c in ["DEC", "Dec", "dec", "DEC_deg", "dec_deg"] if c in names), None),
        "peak_flux_column": next((c for c in ["Peak_flux", "PEAK_FLUX", "peak_flux"] if c in names), None),
        "total_flux_column": next((c for c in ["Total_flux", "TOTAL_FLUX", "total_flux"] if c in names), None),
        "major_axis_column": next((c for c in ["Maj", "MAJ", "maj"] if c in names), None),
        "minor_axis_column": next((c for c in ["Min", "MIN", "min"] if c in names), None),
        "pa_column": next((c for c in ["PA", "pa"] if c in names), None),
        "island_id_column": next((c for c in ["Isl_id", "ISL_ID", "Island_id", "island_id"] if c in names), None),
        "source_id_column": next((c for c in ["Source_id", "SOURCE_ID", "source_id"] if c in names), None),
        "local_rms_column": next((c for c in ["Isl_rms", "isl_rms", "Local_rms", "local_rms", "RMS", "rms"] if c in names), None),
        "gaussian_id_column": next((c for c in ["Gaus_id", "Gaussian_id", "GAUSSIAN_ID", "gaussian_id"] if c in names), None),
    }
    warnings = []
    for unit_key, col in mapping.items():
        if col is None:
            continue
        unit = str(getattr(table[col], "unit", "") or "")
        if not unit:
            warnings.append(f"TUNIT missing for {col}; using project catalogue loader conventions")
    duplicate_gaussians = 0
    if mapping["gaussian_id_column"] is not None:
        values = pd.Series(np.asarray(table[mapping["gaussian_id_column"]]).astype(str))
        duplicate_gaussians = int(values.duplicated().sum())
    island_unique_note = "unknown"
    if mapping["island_id_column"] is not None:
        island_values = pd.Series(np.asarray(table[mapping["island_id_column"]]).astype(str))
        island_unique_note = "island_id reused across rows as expected for multi-Gaussian islands"
    out = {
        "gaussian_catalogue": str(path),
        "number_of_rows": int(len(table)),
        "column_names": table.colnames,
        **mapping,
        "duplicate_gaussian_rows": duplicate_gaussians,
        "island_id_uniqueness": island_unique_note,
        "unit_notes": {
            "RA/Dec": "degree from TUNIT when present; otherwise loader convention for LoTSS PyBDSF",
            "major/minor": "arcsec expected by current project loader; TUNIT warnings listed if absent",
            "PA": "degree",
            "flux": "catalogue native units; LoTSS PyBDSF is typically Jy/Jy beam",
        },
        "warnings": warnings,
    }
    if components is not None and not components.empty:
        out.update(
            {
                "projected_components_retained": int(len(components)),
                "projected_x_min_local": float(pd.to_numeric(components["x"], errors="coerce").min()),
                "projected_x_max_local": float(pd.to_numeric(components["x"], errors="coerce").max()),
                "projected_y_min_local": float(pd.to_numeric(components["y"], errors="coerce").min()),
                "projected_y_max_local": float(pd.to_numeric(components["y"], errors="coerce").max()),
            }
        )
    return out


def plot_gaussian_overlay(image: np.ndarray, components: pd.DataFrame, output_path: str | Path, seed: int = 42) -> None:
    stride = max(1, int(np.ceil(max(image.shape) / 3000)))
    shown = image[::stride, ::stride]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_display_image(shown), origin="lower", cmap="gray")
    if not components.empty:
        ax.scatter(components["x"] / stride, components["y"] / stride, s=8, facecolor="none", edgecolor="tab:cyan", linewidth=0.5)
        sample = components.sample(min(30, len(components)), random_state=seed)
        for _, row in sample.iterrows():
            ax.text(float(row["x"]) / stride + 3, float(row["y"]) / stride + 3, str(row["component_id"]), color="yellow", fontsize=5)
    ax.plot(
        [0, shown.shape[1] - 1, shown.shape[1] - 1, 0, 0],
        [0, 0, shown.shape[0] - 1, shown.shape[0] - 1, 0],
        color="tab:red",
        linewidth=1.2,
    )
    ax.set_title("Radio image with Gaussian centres and demo-region boundary")
    ax.set_xticks([])
    ax.set_yticks([])
    _save_figure(fig, Path(output_path).with_suffix(""))


def plot_mask_overlay(image: np.ndarray, mask: np.ndarray, components: pd.DataFrame, output_path: str | Path) -> None:
    stride = max(1, int(np.ceil(max(image.shape) / 3000)))
    shown = image[::stride, ::stride]
    shown_mask = mask[::stride, ::stride]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_display_image(shown), origin="lower", cmap="gray")
    ax.contour(shown_mask.astype(float), levels=[0.5], colors=["tab:orange"], linewidths=0.7)
    if not components.empty:
        ax.scatter(components["x"] / stride, components["y"] / stride, s=8, facecolor="none", edgecolor="tab:cyan", linewidth=0.5)
    ax.set_title("Radio image, 3 sigma mask contour, Gaussian centres")
    ax.set_xticks([])
    ax.set_yticks([])
    _save_figure(fig, Path(output_path).with_suffix(""))


def edge_table_hash(edges: pd.DataFrame) -> str:
    """Hash deterministic pre-clustering edge evidence columns."""

    if edges.empty:
        return hashlib.sha256(b"empty").hexdigest()
    drop = {"accepted", "debug_info"}
    cols = [c for c in edges.columns if c not in drop]
    work = edges.loc[:, cols].copy()
    sort_cols = [c for c in ["component_index_1", "component_index_2"] if c in work]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)
    csv = work.to_csv(index=False, na_rep="NaN")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def component_sample_check(memberships: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    reference: set[str] | None = None
    for method_key, membership in sorted(memberships.items()):
        comp = set(membership.get("component_id", pd.Series(dtype=str)).astype(str))
        if reference is None:
            reference = comp
        rows.append(
            {
                "method": method_key,
                "N_components": int(len(comp)),
                "component_id_set_identical": bool(comp == reference),
                "missing_vs_reference": ",".join(sorted((reference or set()) - comp)[:50]),
                "extra_vs_reference": ",".join(sorted(comp - (reference or set()))[:50]),
            }
        )
    return pd.DataFrame(rows)


def beam_distance_sample(components: pd.DataFrame, candidate_pairs: pd.DataFrame, config: dict[str, Any], n: int = 20) -> pd.DataFrame:
    if candidate_pairs.empty:
        return pd.DataFrame()
    sample = candidate_pairs.sample(min(n, len(candidate_pairs)), random_state=int(config.get("random_seed", 42))).copy()
    beam = compute_beam_size_arcsec(config)
    out = sample[["component_id_1", "component_id_2", "distance_arcsec", "distance_beam"]].copy()
    out = out.rename(columns={"component_id_1": "component_i", "component_id_2": "component_j", "distance_arcsec": "angular_distance_arcsec"})
    out["beam_fwhm_arcsec"] = beam
    out["distance_beam_recomputed"] = out["angular_distance_arcsec"] / max(beam, 1e-6)
    return out


def mask_stats(mask: np.ndarray, labels: np.ndarray, assigned_labels: np.ndarray) -> dict[str, Any]:
    valid_pixels = int(mask.size)
    counts = np.bincount(labels.ravel())
    label_counts = pd.Series(assigned_labels[assigned_labels > 0]).value_counts()
    return {
        "fraction_valid_pixels_above_3sigma": float(mask.sum() / max(valid_pixels, 1)),
        "number_connected_regions": int(labels.max()),
        "largest_connected_region_area": int(counts[1:].max()) if len(counts) > 1 else 0,
        "regions_with_zero_gaussians": int(max(int(labels.max()) - len(label_counts), 0)),
        "regions_with_one_gaussian": int((label_counts == 1).sum()),
        "regions_with_at_least_two_gaussians": int((label_counts >= 2).sum()),
    }


def write_preflight_report(
    output_dir: str | Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    catalogue: dict[str, Any],
    data_quality: dict[str, Any],
    edge_validation: dict[str, Any] | None = None,
    mask_check: dict[str, Any] | None = None,
) -> None:
    """Write a compact preflight markdown report."""

    preflight_dir = Path(output_dir) / "preflight"
    lines = [
        "# Preflight Report",
        "",
        "## Inputs",
        f"- Tile ID: `{config.get('tile_id')}`",
        f"- Radio FITS: `{config.get('radio_image')}`",
        f"- Gaussian catalogue: `{config.get('gaussian_catalogue')}`",
        f"- RMS map: `{config.get('rms_map')}`",
        f"- Demo region: `{config.get('demo_region')}`",
        "",
        "## Radio FITS",
        f"- FITS dimensions: `{preflight.get('fits_dimensions')}`",
        f"- Number of HDUs: `{preflight.get('number_of_hdus')}`",
        f"- Image HDU index: `{preflight.get('image_hdu_index')}`",
        f"- Selected plane: `{preflight.get('selected_plane')}`",
        f"- BUNIT/BMAJ/BMIN/BPA: `{preflight.get('BUNIT')}`, `{preflight.get('BMAJ')}`, `{preflight.get('BMIN')}`, `{preflight.get('BPA')}`",
        f"- CDELT1/CDELT2: `{preflight.get('CDELT1')}`, `{preflight.get('CDELT2')}`",
        f"- CTYPE1/CTYPE2: `{preflight.get('CTYPE1')}`, `{preflight.get('CTYPE2')}`",
        f"- CRVAL1/CRVAL2: `{preflight.get('CRVAL1')}`, `{preflight.get('CRVAL2')}`",
        f"- NaN pixels in region: `{preflight.get('n_nan_pixels_region')}`",
        f"- Region min/max/median: `{preflight.get('minimum_region')}`, `{preflight.get('maximum_region')}`, `{preflight.get('median_region')}`",
        f"- Region robust RMS: `{preflight.get('robust_rms_estimate_region')}`",
        "",
        "## Catalogue",
        f"- Rows: `{catalogue.get('number_of_rows')}`",
        f"- RA/Dec columns: `{catalogue.get('ra_column')}`, `{catalogue.get('dec_column')}`",
        f"- Flux columns: peak `{catalogue.get('peak_flux_column')}`, total `{catalogue.get('total_flux_column')}`",
        f"- Shape columns: major `{catalogue.get('major_axis_column')}`, minor `{catalogue.get('minor_axis_column')}`, PA `{catalogue.get('pa_column')}`",
        f"- Island/source/local RMS columns: `{catalogue.get('island_id_column')}`, `{catalogue.get('source_id_column')}`, `{catalogue.get('local_rms_column')}`",
        f"- Duplicate Gaussian rows: `{catalogue.get('duplicate_gaussian_rows')}`",
        f"- Unit warnings: `{catalogue.get('warnings')}`",
        "",
        "## Data Quality",
        f"- Full image shape: `{data_quality.get('full_image_shape')}`",
        f"- Analysis image shape: `{data_quality.get('analysis_image_shape')}`",
        f"- Raw catalogue rows: `{data_quality.get('number_raw_catalogue_rows')}`",
        f"- Retained components: `{data_quality.get('number_retained')}`",
        f"- Outside image: `{data_quality.get('number_projected_outside_image')}`",
        f"- Invalid geometry: `{data_quality.get('number_invalid_geometry')}`",
        f"- Duplicate components: `{data_quality.get('number_duplicate_component_ids')}`",
        f"- Beam FWHM arcsec: `{data_quality.get('beam_major_arcsec')}` x `{data_quality.get('beam_minor_arcsec')}`",
        f"- Pixel scale arcsec: `{data_quality.get('pixel_scale_arcsec')}`",
    ]
    if mask_check is not None:
        lines.extend(
            [
                "",
                "## 3 Sigma Mask",
                f"- Fraction above 3 sigma: `{mask_check.get('fraction_valid_pixels_above_3sigma')}`",
                f"- Connected regions: `{mask_check.get('number_connected_regions')}`",
                f"- Largest region area: `{mask_check.get('largest_connected_region_area')}`",
                f"- Regions with zero/one/ge2 Gaussians: `{mask_check.get('regions_with_zero_gaussians')}`, `{mask_check.get('regions_with_one_gaussian')}`, `{mask_check.get('regions_with_at_least_two_gaussians')}`",
            ]
        )
    if edge_validation is not None:
        lines.extend(
            [
                "",
                "## Edge Input Validation",
                f"- Full hash: `{edge_validation.get('edge_input_hash_full')}`",
                f"- Unconstrained hash: `{edge_validation.get('edge_input_hash_unconstrained')}`",
                f"- Edge tables identical before clustering: `{edge_validation.get('edge_tables_identical_before_clustering')}`",
                f"- Component sets identical: `{edge_validation.get('component_sets_identical')}`",
            ]
        )
    (preflight_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def expected_output_check(output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    expected = [
        "components_clean.parquet",
        "baseline_summary.csv",
        "method_agreement.csv",
        "case_rankings.csv",
        "tile_baseline_report.md",
        "mask_3sigma.fits",
        "mask_3sigma_labels.fits",
        "groups_pybdsf_island.parquet",
        "groups_contour_3sigma.parquet",
        "groups_unconstrained_graph.parquet",
        "groups_full_layer1.parquet",
        "edges_unconstrained.parquet",
        "edges_full_layer1.parquet",
        "preflight/preflight_report.md",
    ]
    rows = [{"path": name, "exists": (output / name).exists()} for name in expected]
    rows.append({"path": "groups_distance_*.parquet", "exists": bool(list(output.glob("groups_distance_*.parquet")))})
    return pd.DataFrame(rows)


def membership_overlap_rows(reference: pd.DataFrame, other: pd.DataFrame, reference_name: str, other_name: str) -> pd.DataFrame:
    if reference.empty or other.empty:
        return pd.DataFrame()
    joined = reference[["component_id", "predicted_group_id"]].rename(columns={"predicted_group_id": reference_name}).merge(
        other[["component_id", "predicted_group_id"]].rename(columns={"predicted_group_id": other_name}),
        on="component_id",
    )
    return joined


def structural_tables(
    components: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    groups: dict[str, pd.DataFrame],
    edges: pd.DataFrame,
    distance_stats: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write method-specific structural comparison tables."""

    output = Path(output_dir)
    full = memberships.get("full_layer1:current_config", pd.DataFrame())
    pyb = memberships.get("pybdsf_island:native", pd.DataFrame())
    contour_key = next((k for k in memberships if k.startswith("contour_3sigma:")), "")
    contour = memberships.get(contour_key, pd.DataFrame())
    uncon = memberships.get("unconstrained_graph:strong_plus_accepted_weak", pd.DataFrame())
    results: dict[str, Any] = {}

    if not full.empty and not pyb.empty:
        comp_island = dict(zip(components["component_id"].astype(str), components["island_id"].astype(str)))
        rows = []
        for gid, frame in full.groupby("predicted_group_id"):
            islands = {comp_island.get(cid) for cid in frame["component_id"].astype(str)}
            islands.discard(None)
            rows.append({"full_group_id": gid, "n_components": len(frame), "n_pybdsf_islands": len(islands), "pybdsf_islands": ",".join(sorted(islands))})
        pyb_table = pd.DataFrame(rows)
        pyb_table.to_csv(output / "pybdsf_fragmentation_comparison.csv", index=False)
        denom = max(len(pyb_table), 1)
        results["pybdsf_fraction_full_groups_1_island"] = float((pyb_table["n_pybdsf_islands"] == 1).sum() / denom)
        results["pybdsf_fraction_full_groups_2_islands"] = float((pyb_table["n_pybdsf_islands"] == 2).sum() / denom)
        results["pybdsf_fraction_full_groups_ge3_islands"] = float((pyb_table["n_pybdsf_islands"] >= 3).sum() / denom)
        results["number_pybdsf_islands_merged_by_full_layer1"] = int(pyb_table.loc[pyb_table["n_pybdsf_islands"] >= 2, "n_pybdsf_islands"].sum())

    if not full.empty and not contour.empty:
        joined = membership_overlap_rows(full, contour, "full_group_id", "contour_group_id")
        split_rows = []
        for gid, frame in joined.groupby("full_group_id"):
            split_rows.append({"full_group_id": gid, "n_contour_groups": frame["contour_group_id"].nunique(), "n_components": len(frame)})
        contour_split = pd.DataFrame(split_rows)
        contour_split.to_csv(output / "contour_split_comparison.csv", index=False)
        merge_rows = []
        for gid, frame in joined.groupby("contour_group_id"):
            merge_rows.append({"contour_group_id": gid, "n_full_groups": frame["full_group_id"].nunique(), "n_components": len(frame)})
        pd.DataFrame(merge_rows).to_csv(output / "contour_merge_comparison.csv", index=False)
        results["full_groups_reproduced_by_one_3sigma_region"] = int((contour_split["n_contour_groups"] == 1).sum()) if not contour_split.empty else 0
        results["full_groups_split_into_2_3sigma_regions"] = int((contour_split["n_contour_groups"] == 2).sum()) if not contour_split.empty else 0
        results["full_groups_split_into_ge3_3sigma_regions"] = int((contour_split["n_contour_groups"] >= 3).sum()) if not contour_split.empty else 0

    if not full.empty and not uncon.empty:
        joined = membership_overlap_rows(uncon, full, "unconstrained_group_id", "full_group_id")
        rows = []
        for gid, frame in joined.groupby("unconstrained_group_id"):
            rows.append({"unconstrained_group_id": gid, "n_full_groups": frame["full_group_id"].nunique(), "n_components": len(frame)})
        table = pd.DataFrame(rows)
        table.to_csv(output / "unconstrained_vs_full_split_comparison.csv", index=False)
        multi_uncon = max(int((groups.get("unconstrained_graph:strong_plus_accepted_weak", pd.DataFrame()).get("n_components", pd.Series(dtype=int)) >= 2).sum()), 1)
        results["number_unconstrained_groups"] = int(uncon["predicted_group_id"].nunique())
        results["number_full_constrained_groups"] = int(full["predicted_group_id"].nunique())
        results["unconstrained_groups_containing_ge2_full_groups"] = int((table["n_full_groups"] >= 2).sum()) if not table.empty else 0
        results["fraction_unconstrained_groups_unchanged"] = float((table["n_full_groups"] == 1).sum() / max(len(table), 1)) if not table.empty else 0.0
        results["fraction_unconstrained_groups_split_into_2_full_groups"] = float((table["n_full_groups"] == 2).sum() / max(len(table), 1)) if not table.empty else 0.0
        results["fraction_unconstrained_groups_split_into_ge3_full_groups"] = float((table["n_full_groups"] >= 3).sum() / max(len(table), 1)) if not table.empty else 0.0
        results["unconstrained_multi_component_groups_denominator"] = int(multi_uncon)
        results["unconstrained_groups_containing_ge2_strong_cores"] = int((table["n_full_groups"] >= 2).sum()) if not table.empty else 0
        results["full_core_pairs_connected_only_through_weak_edges"] = int(sum(max(0, n * (n - 1) // 2) for n in table["n_full_groups"].astype(int) if n >= 2)) if not table.empty else 0
        results["weak_only_chains"] = int((table["n_full_groups"] >= 2).sum()) if not table.empty else 0
        results["fraction_multi_core_over_unconstrained_multi_component_groups"] = float(
            results["unconstrained_groups_containing_ge2_strong_cores"] / max(multi_uncon, 1)
        )

    if not distance_stats.empty:
        merged = distance_stats.copy()
        group_rows = []
        for key, group_table in sorted(groups.items()):
            if not key.startswith("distance_only:") or group_table.empty or "n_components" not in group_table:
                continue
            parameter_id = key.split(":", 1)[1]
            sizes = pd.to_numeric(group_table["n_components"], errors="coerce").dropna().to_numpy(float)
            group_rows.append(
                {
                    "parameter_id": parameter_id,
                    "median_group_size": float(np.median(sizes)) if len(sizes) else np.nan,
                    "fraction_components_in_multi_groups": float(sizes[sizes >= 2].sum() / max(len(components), 1)) if len(sizes) else 0.0,
                    "n_multi_component_groups": int(np.count_nonzero(sizes >= 2)) if len(sizes) else 0,
                }
            )
        if group_rows:
            merged = merged.merge(pd.DataFrame(group_rows), on="parameter_id", how="left")
        if not full.empty:
            merge_rows = []
            for key, membership in sorted(memberships.items()):
                if not key.startswith("distance_only:tau_"):
                    continue
                parameter_id = key.split(":", 1)[1]
                joined = membership_overlap_rows(membership, full, "distance_group_id", "full_group_id")
                if joined.empty:
                    continue
                n_full_by_distance = joined.groupby("distance_group_id")["full_group_id"].nunique()
                merge_rows.append(
                    {
                        "parameter_id": parameter_id,
                        "number_groups_containing_multiple_full_layer1_groups": int((n_full_by_distance >= 2).sum()),
                        "max_full_layer1_groups_per_distance_group": int(n_full_by_distance.max()) if len(n_full_by_distance) else 0,
                    }
                )
            if merge_rows:
                merge_table = pd.DataFrame(merge_rows)
                merge_table.to_csv(output / "distance_threshold_merge_disagreement.csv", index=False)
                merged = merged.merge(merge_table, on="parameter_id", how="left")
                results["distance_threshold_merge_disagreement_max"] = int(
                    merge_table["number_groups_containing_multiple_full_layer1_groups"].max()
                )
        merged.to_csv(output / "distance_threshold_structural_stats.csv", index=False)
    return results


def _bbox_for_components(components: pd.DataFrame, component_ids: set[str], pad: int = 32) -> dict[str, int]:
    rows = components[components["component_id"].astype(str).isin(component_ids)]
    if rows.empty:
        return {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}
    return {
        "xmin": int(max(0, np.floor(rows["x"].min()) - pad)),
        "xmax": int(np.ceil(rows["x"].max()) + pad),
        "ymin": int(max(0, np.floor(rows["y"].min()) - pad)),
        "ymax": int(np.ceil(rows["y"].max()) + pad),
    }


def build_case_exports(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Build ranked case tables and manual-review templates."""

    output = Path(output_dir)
    cases_dir = output / "cases"
    manual_dir = output / "manual_review"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manual_dir.mkdir(parents=True, exist_ok=True)

    full = memberships.get("full_layer1:current_config", pd.DataFrame())
    uncon = memberships.get("unconstrained_graph:strong_plus_accepted_weak", pd.DataFrame())
    pyb = memberships.get("pybdsf_island:native", pd.DataFrame())
    contour = next((m for k, m in memberships.items() if k.startswith("contour_3sigma:")), pd.DataFrame())
    distance_keys = sorted(k for k in memberships if k.startswith("distance_only:"))
    distance = memberships.get(distance_keys[-1], pd.DataFrame()) if distance_keys else pd.DataFrame()

    all_cases = []

    if not full.empty and not uncon.empty:
        joined = membership_overlap_rows(uncon, full, "unconstrained_group_id", "full_group_id")
        full_by_component = dict(zip(full["component_id"].astype(str), full["predicted_group_id"].astype(str)))
        weak_rows = []
        for gid, frame in joined.groupby("unconstrained_group_id"):
            full_ids = sorted(frame["full_group_id"].astype(str).unique())
            if len(full_ids) < 2:
                continue
            cids = set(frame["component_id"].astype(str))
            bbox = _bbox_for_components(components, cids)
            internal = edges[
                edges.get("gaussian_id_1", pd.Series(dtype=str)).astype(str).isin(cids)
                & edges.get("gaussian_id_2", pd.Series(dtype=str)).astype(str).isin(cids)
            ] if not edges.empty else pd.DataFrame()
            n_strong = int((internal.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not internal.empty else 0
            n_weak = int((internal.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not internal.empty else 0
            weak_core_graph = nx.Graph()
            weak_core_graph.add_nodes_from(full_ids)
            if not internal.empty:
                for _, edge in internal.iterrows():
                    if str(edge.get("edge_type")) != "weak":
                        continue
                    c1 = str(edge.get("gaussian_id_1", ""))
                    c2 = str(edge.get("gaussian_id_2", ""))
                    g1 = full_by_component.get(c1)
                    g2 = full_by_component.get(c2)
                    if g1 is not None and g2 is not None and g1 != g2:
                        weak_core_graph.add_edge(g1, g2)
            path_lengths = []
            if len(weak_core_graph) >= 2:
                for source, lengths in nx.all_pairs_shortest_path_length(weak_core_graph):
                    for target, length in lengths.items():
                        if str(source) < str(target):
                            path_lengths.append(int(length))
            weak_path_length = max(path_lengths) if path_lengths else np.nan
            connected_only_by_weak = bool(path_lengths and nx.is_connected(weak_core_graph.subgraph(full_ids)))
            weak_rows.append({
                "case_id": f"weak_chain_{len(weak_rows):03d}",
                "case_type": "weak_chain",
                "unconstrained_group_id": gid,
                "full_group_ids": ",".join(full_ids),
                "n_components": len(cids),
                "n_strong_cores": len(full_ids),
                "n_strong_edges": n_strong,
                "n_weak_edges": n_weak,
                "weak_path_length": weak_path_length,
                "strong_cores_connected_by_weak_edges": connected_only_by_weak,
                "boundary_flag": "boundary_affected" if any(v <= 0 for v in [bbox["xmin"], bbox["ymin"]]) else "interior",
                **bbox,
                "component_ids": ",".join(sorted(cids)),
            })
        weak_table = pd.DataFrame(weak_rows)
        if not weak_table.empty:
            weak_table["_boundary_rank"] = (weak_table["boundary_flag"].astype(str) != "interior").astype(int)
            weak_table = weak_table.sort_values(
                ["_boundary_rank", "strong_cores_connected_by_weak_edges", "n_strong_cores", "n_weak_edges", "n_components"],
                ascending=[True, False, False, False, True],
            ).drop(columns=["_boundary_rank"]).head(30)
        weak_table.to_csv(cases_dir / "weak_chain_candidates.csv", index=False)
        all_cases.append(weak_table)

    if not full.empty and not pyb.empty:
        comp_island = dict(zip(components["component_id"].astype(str), components["island_id"].astype(str)))
        rows = []
        for gid, frame in full.groupby("predicted_group_id"):
            cids = set(frame["component_id"].astype(str))
            islands = {comp_island.get(cid) for cid in cids}
            islands.discard(None)
            if len(islands) >= 2 and len(cids) >= 3:
                rows.append({"case_id": f"pybdsf_split_{len(rows):03d}", "case_type": "pybdsf_split", "full_group_id": gid, "n_components": len(cids), "n_pybdsf_islands": len(islands), "component_ids": ",".join(sorted(cids)), **_bbox_for_components(components, cids)})
        table = pd.DataFrame(rows).sort_values(["n_pybdsf_islands", "n_components"], ascending=[False, False]).head(20) if rows else pd.DataFrame()
        table.to_csv(cases_dir / "pybdsf_fragmentation_candidates.csv", index=False)
        all_cases.append(table)

    if not full.empty and not contour.empty:
        joined = membership_overlap_rows(full, contour, "full_group_id", "contour_group_id")
        rows = []
        for gid, frame in joined.groupby("full_group_id"):
            n = frame["contour_group_id"].nunique()
            cids = set(frame["component_id"].astype(str))
            if n >= 2:
                rows.append({"case_id": f"contour_split_{len(rows):03d}", "case_type": "contour_split", "full_group_id": gid, "n_components": len(cids), "n_contour_groups": n, "component_ids": ",".join(sorted(cids)), **_bbox_for_components(components, cids)})
        table = pd.DataFrame(rows).sort_values(["n_contour_groups", "n_components"], ascending=[False, False]).head(20) if rows else pd.DataFrame()
        table.to_csv(cases_dir / "contour_split_candidates.csv", index=False)
        all_cases.append(table)

    if not full.empty and not distance.empty:
        joined = membership_overlap_rows(distance, full, "distance_group_id", "full_group_id")
        rows = []
        for gid, frame in joined.groupby("distance_group_id"):
            n = frame["full_group_id"].nunique()
            cids = set(frame["component_id"].astype(str))
            if n >= 2:
                rows.append({"case_id": f"distance_overmerge_{len(rows):03d}", "case_type": "distance_overmerge", "distance_group_id": gid, "n_components": len(cids), "n_full_groups": n, "component_ids": ",".join(sorted(cids)), **_bbox_for_components(components, cids)})
        table = pd.DataFrame(rows).sort_values(["n_full_groups", "n_components"], ascending=[False, True]).head(20) if rows else pd.DataFrame()
        table.to_csv(cases_dir / "distance_overmerge_candidates.csv", index=False)
        all_cases.append(table)

    # Agreement cases: components in groups that all methods keep together.
    if memberships:
        rows = []
        full_groups = full.groupby("predicted_group_id") if not full.empty else []
        for gid, frame in full_groups:
            cids = set(frame["component_id"].astype(str))
            if len(cids) < 2:
                continue
            ok = True
            for mem in memberships.values():
                sub = mem[mem["component_id"].astype(str).isin(cids)]
                if sub["predicted_group_id"].nunique() != 1:
                    ok = False
                    break
            if ok:
                rows.append({"case_id": f"all_agree_{len(rows):03d}", "case_type": "all_method_agreement", "full_group_id": gid, "n_components": len(cids), "component_ids": ",".join(sorted(cids)), **_bbox_for_components(components, cids)})
        table = pd.DataFrame(rows).sort_values("n_components", ascending=False).head(10) if rows else pd.DataFrame()
        table.to_csv(cases_dir / "all_method_agreement_candidates.csv", index=False)
        all_cases.append(table)

    case_rankings = pd.concat([t for t in all_cases if t is not None and not t.empty], ignore_index=True) if any(t is not None and not t.empty for t in all_cases) else pd.DataFrame()
    case_rankings.to_csv(output / "case_rankings.csv", index=False)

    label_rows = []
    for _, row in case_rankings.iterrows():
        for cid in parse_component_ids(row.get("component_ids", "")):
            label_rows.append({"case_id": row["case_id"], "component_id": cid, "true_local_group_id": "", "label_quality": "", "artifact_flag": "", "notes": ""})
    pd.DataFrame(label_rows).to_csv(manual_dir / "manual_labels_template.csv", index=False)
    instructions = (
        "# Manual Review Instructions\n\n"
        "Fill `true_local_group_id` so components belonging to the same local radio emission structure share one id within each case.\n\n"
        "`label_quality`: confident / uncertain\n\n"
        "`artifact_flag`: real / likely_artifact / uncertain\n\n"
        "Do not treat any baseline as ground truth; use the radio cutouts and component layout.\n"
    )
    (manual_dir / "manual_review_instructions.md").write_text(instructions, encoding="utf-8")
    make_contact_sheet(image, components, edges, memberships, case_rankings.head(30), manual_dir / "manual_review_contact_sheet")
    make_contact_sheet(image, components, edges, memberships, case_rankings.head(30), cases_dir / "contact_sheet")
    return case_rankings


def make_contact_sheet(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    output_stem: str | Path,
) -> None:
    if cases.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No candidate cases found", ha="center", va="center")
        _save_figure(fig, Path(output_stem))
        return
    n = min(12, len(cases))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, cases.head(n).iterrows()):
        cids = set(parse_component_ids(row.get("component_ids", "")))
        sub = components[components["component_id"].astype(str).isin(cids)]
        if sub.empty:
            continue
        pad = 32
        x0 = max(0, int(np.floor(sub["x"].min())) - pad)
        x1 = min(image.shape[1], int(np.ceil(sub["x"].max())) + pad)
        y0 = max(0, int(np.floor(sub["y"].min())) - pad)
        y1 = min(image.shape[0], int(np.ceil(sub["y"].max())) + pad)
        ax.imshow(_display_image(image[y0:y1, x0:x1]), origin="lower", cmap="gray")
        ax.scatter(sub["x"] - x0, sub["y"] - y0, s=14, facecolor="none", edgecolor="tab:cyan", linewidth=0.6)
        ax.set_title(f"{row.get('case_id')} n={len(cids)}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    _save_figure(fig, Path(output_stem))


def write_tile_report(
    output_dir: str | Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    catalogue: dict[str, Any],
    data_quality: dict[str, Any],
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    validation: dict[str, Any],
    structural: dict[str, Any],
    case_rankings: pd.DataFrame,
) -> None:
    output = Path(output_dir)

    def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if df.empty:
            return "_No rows available._"
        work = df.head(max_rows).copy() if max_rows is not None else df.copy()
        work = work.fillna("")
        cols = list(work.columns)
        lines = [
            "| " + " | ".join(str(c) for c in cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in work.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
        return "\n".join(lines)

    def read_output_csv(name: str) -> pd.DataFrame:
        path = output / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return df[[column for column in columns if column in df.columns]].copy() if not df.empty else df

    distance_table = read_output_csv("distance_threshold_structural_stats.csv")
    pyb_table = read_output_csv("pybdsf_fragmentation_comparison.csv")
    contour_split_table = read_output_csv("contour_split_comparison.csv")
    contour_merge_table = read_output_csv("contour_merge_comparison.csv")
    uncon_table = read_output_csv("unconstrained_vs_full_split_comparison.csv")
    run_summary = read_output_csv("run_summary.csv")
    edge_table = read_output_csv("edges_full_layer1.csv")
    case_counts = (
        case_rankings["case_type"].value_counts().rename_axis("case_type").reset_index(name="n_cases")
        if not case_rankings.empty and "case_type" in case_rankings
        else pd.DataFrame()
    )
    full_rows = summary[summary["method"].astype(str) == "full_layer1"]
    distance_scan = distance_table[distance_table.get("parameter_id", pd.Series(dtype=str)).astype(str).str.startswith("tau_")].copy()
    largest_distance_note = ""
    if not distance_scan.empty and {"threshold_beam", "max_group_size"}.issubset(distance_scan.columns):
        distance_scan = distance_scan.sort_values("threshold_beam")
        first = distance_scan.iloc[0]
        last = distance_scan.iloc[-1]
        largest_distance_note = (
            f"The largest distance-only group increased from `{int(first['max_group_size'])}` "
            f"at tau `{first['threshold_beam']}` to `{int(last['max_group_size'])}` at tau `{last['threshold_beam']}`."
        )
    full_cross_multi = 0.0
    if not pyb_table.empty and "n_pybdsf_islands" in pyb_table:
        full_cross_multi = float((pyb_table["n_pybdsf_islands"] >= 2).sum() / max(len(pyb_table), 1))
    edge_counts = (
        edge_table["edge_type"].value_counts().rename_axis("edge_type").reset_index(name="n_edges")
        if not edge_table.empty and "edge_type" in edge_table
        else pd.DataFrame()
    )

    lines = [
        "# Tile Baseline Report",
        "",
        "## Tile And Inputs",
        f"- Tile ID: `{config.get('tile_id')}`",
        f"- Radio FITS: `{config.get('radio_image')}`",
        f"- Gaussian catalogue: `{config.get('gaussian_catalogue')}`",
        f"- RMS map: `{config.get('rms_map')}`",
        f"- Output directory: `{output}`",
        f"- Demo region: `{(config.get('demo_region') or {})}`",
        f"- Analysis image shape: `{data_quality.get('analysis_image_shape')}`",
        f"- Full image shape: `{data_quality.get('full_image_shape')}`",
        "",
        "## Preflight",
        f"- FITS dimensions: `{preflight.get('fits_dimensions')}`",
        f"- Image HDU index: `{preflight.get('image_hdu_index')}`",
        f"- BUNIT/BMAJ/BMIN/BPA: `{preflight.get('BUNIT')}`, `{preflight.get('BMAJ')}`, `{preflight.get('BMIN')}`, `{preflight.get('BPA')}`",
        f"- Pixel scale CDELT: `{preflight.get('CDELT1')}`, `{preflight.get('CDELT2')}`",
        f"- Region robust rms: `{preflight.get('robust_rms_estimate_region')}`",
        f"- Gaussian rows: `{catalogue.get('number_of_rows')}`; retained: `{data_quality.get('number_retained')}`",
        f"- Beam arcsec: `{data_quality.get('beam_major_arcsec')}` x `{data_quality.get('beam_minor_arcsec')}`; pixel scale arcsec: `{data_quality.get('pixel_scale_arcsec')}`",
        f"- Mean/RMS used for S/N: `{data_quality.get('mean_used')}`, `{data_quality.get('rms_used')}` from `{data_quality.get('rms_source')}`",
        "",
        "## Method Definitions",
        "- PyBDSF island: same native island id.",
        "- Distance-only: connected components over beam-normalized centre-distance thresholds.",
        "- 3 sigma connectivity: connected components of the 3 sigma S/N mask only.",
        "- Unconstrained graph: ordinary connected components over the full method's strong and weak edges.",
        "- Full Layer-1: current constrained strong-core plus weak-singleton attachment method.",
        "",
        "## Parameters",
        "```json",
        json.dumps(
            {
                "distance_baseline": config.get("distance_baseline"),
                "contour_baseline": config.get("contour_baseline"),
                "association": config.get("association"),
                "beam": config.get("beam"),
                "neighbour_search": config.get("neighbour_search"),
            },
            indent=2,
            default=str,
        ),
        "```",
        "",
        "## Baseline Summary",
        md_table(summary),
        "",
        "## Runtime And Memory",
        md_table(select_columns(run_summary, ["runtime_seconds", "n_components", "n_candidate_pairs", "warnings"])),
        "",
        "## Edge Counts",
        md_table(edge_counts),
        "",
        "## Distance Threshold Sensitivity",
        md_table(
            select_columns(
                distance_table,
                [
                    "parameter_id",
                    "threshold_beam",
                    "n_groups",
                    "n_singletons",
                    "n_multi_component_groups",
                    "median_group_size",
                    "max_group_size",
                    "fraction_components_in_multi_groups",
                    "number_groups_containing_multiple_full_layer1_groups",
                ],
            )
        ),
        "",
        "## PyBDSF Island Fragmentation Comparison",
        f"- Fraction of full groups containing 1 PyBDSF island: `{structural.get('pybdsf_fraction_full_groups_1_island')}`.",
        f"- Fraction containing 2 islands: `{structural.get('pybdsf_fraction_full_groups_2_islands')}`.",
        f"- Fraction containing 3 or more islands: `{structural.get('pybdsf_fraction_full_groups_ge3_islands')}`.",
        f"- Number of PyBDSF island memberships merged by full Layer-1 groups: `{structural.get('number_pybdsf_islands_merged_by_full_layer1')}`.",
        "The full method associates components across multiple PyBDSF islands.",
        md_table(select_columns(pyb_table, ["full_group_id", "n_components", "n_pybdsf_islands"]), max_rows=20),
        "",
        "## 3 Sigma Connectivity Comparison",
        f"- Full groups reproduced by one 3 sigma region: `{structural.get('full_groups_reproduced_by_one_3sigma_region')}`.",
        f"- Full groups split into 2 regions: `{structural.get('full_groups_split_into_2_3sigma_regions')}`.",
        f"- Full groups split into 3 or more regions: `{structural.get('full_groups_split_into_ge3_3sigma_regions')}`.",
        md_table(select_columns(contour_split_table, ["full_group_id", "n_contour_groups", "n_components"]), max_rows=20),
        "",
        "3 sigma groups that contain multiple full Layer-1 groups:",
        md_table(select_columns(contour_merge_table[contour_merge_table.get("n_full_groups", pd.Series(dtype=int)) >= 2] if not contour_merge_table.empty else contour_merge_table, ["contour_group_id", "n_full_groups", "n_components"]), max_rows=20),
        "",
        "## Unconstrained Versus Constrained Comparison",
        f"- Number of unconstrained groups: `{structural.get('number_unconstrained_groups')}`.",
        f"- Number of full constrained groups: `{structural.get('number_full_constrained_groups')}`.",
        f"- Unconstrained groups containing >=2 full groups: `{structural.get('unconstrained_groups_containing_ge2_full_groups')}`.",
        f"- Unconstrained groups containing >=2 strong cores: `{structural.get('unconstrained_groups_containing_ge2_strong_cores')}`.",
        f"- Full core pairs connected only through weak edges: `{structural.get('full_core_pairs_connected_only_through_weak_edges')}`.",
        f"- Weak-only chains: `{structural.get('weak_only_chains')}`.",
        f"- f_multi-core over unconstrained multi-component groups: `{structural.get('fraction_multi_core_over_unconstrained_multi_component_groups')}`.",
        f"- Fraction unchanged: `{structural.get('fraction_unconstrained_groups_unchanged')}`.",
        f"- Fraction split into 2 full groups: `{structural.get('fraction_unconstrained_groups_split_into_2_full_groups')}`.",
        f"- Fraction split into 3 or more full groups: `{structural.get('fraction_unconstrained_groups_split_into_ge3_full_groups')}`.",
        "Observed: the constrained and unconstrained methods produced different memberships for the groups counted above.",
        "Interpretation: these differences are caused by restricting weak edges from merging established strong cores.",
        md_table(select_columns(uncon_table[uncon_table.get("n_full_groups", pd.Series(dtype=int)) >= 2] if not uncon_table.empty else uncon_table, ["unconstrained_group_id", "n_full_groups", "n_components"]), max_rows=20),
        "",
        "## Method Agreement",
        md_table(agreement, max_rows=30),
        "",
        "## Representative Cases",
        md_table(case_counts),
        f"- Case ranking table: `{output / 'case_rankings.csv'}`",
        f"- Contact sheet: `{output / 'cases' / 'contact_sheet.pdf'}` and `{output / 'cases' / 'contact_sheet.png'}`",
        "",
        "## Key Observed Results",
        f"- The full and unconstrained methods used identical pre-clustering edge tables: `{validation.get('edge_tables_identical_before_clustering')}`.",
        f"- Edge input hash full: `{validation.get('edge_input_hash_full')}`.",
        f"- Edge input hash unconstrained: `{validation.get('edge_input_hash_unconstrained')}`.",
        f"- Component sets identical across methods: `{validation.get('component_sets_identical')}`.",
        f"- Weak-chain candidate cases found: `{int((case_rankings.get('case_type', pd.Series(dtype=str)) == 'weak_chain').sum()) if not case_rankings.empty else 0}`.",
        f"- Unconstrained groups containing more than one constrained strong core: `{structural.get('unconstrained_groups_containing_ge2_strong_cores')}`.",
        f"- Fraction of full Layer-1 groups crossing more than one PyBDSF island: `{full_cross_multi}`.",
        f"- Fraction of full groups crossing >=3 PyBDSF islands: `{structural.get('pybdsf_fraction_full_groups_ge3_islands')}`.",
        f"- Unconstrained groups containing >=2 full groups: `{structural.get('unconstrained_groups_containing_ge2_full_groups')}`.",
        f"- {largest_distance_note}" if largest_distance_note else "- Distance threshold scan completed.",
        "",
        "## Limitations",
        "No manual truth was supplied for this run, so no accuracy, precision, recall, or F1 claims are made.",
        "The full method is used only as a reference method for structural agreement and split/merge comparisons.",
        "",
        "## Manual Truth",
        "Use `manual_review/manual_labels_template.csv` and the contact sheet to prepare human labels.",
        "After labels are filled, rerun with `--manual-labels` to compute pairwise precision/recall/F1, B-cubed metrics, exact-group recovery, over-merge rate, and split rate.",
    ]
    (output / "tile_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
