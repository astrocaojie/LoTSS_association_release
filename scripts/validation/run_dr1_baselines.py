#!/usr/bin/env python
"""Generate and evaluate DR1 component-reference baseline catalogues.

The baselines here are deliberately simple source-construction rules over the
existing PyBDSF Gaussian catalogues.  They emit the same prediction catalogue
schema used by the full-method DR1 bbox-containment evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.wcs import WCS

from lofar_det_vsex.utils import load_yaml
from lofar_det_vsex.validation.bbox_support import match_predictions_to_dr1_components
from lofar_det_vsex.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue
from lofar_det_vsex.validation.footprint import build_dr1_sky_footprint, filter_predictions_in_footprint
from lofar_det_vsex.validation.metrics import compute_support_rates
from scripts.lotss_dr3_common import read_manifest, read_table_any
from scripts.validation.dr1_ablation_core6_evaluate import _add_bbox_area, _area_diagnostics, _corrected, _shifted_random_rates


DEFAULT_MANIFEST = Path(os.environ.get("LOTSS_ASSOC_MANIFEST", PROJECT_ROOT / "outputs" / "lotss_dr3_full" / "manifests" / "lotss_dr3_fits_manifest.csv"))
CORE_BASELINES = ["B0", "B1", "B2", "B3", "B4"]
PENDING_BASELINES = ["B5", "B6", "B7"]
REQUIRED_COLUMNS = {"source_id", "sample", "bbox_ra_min", "bbox_ra_max", "bbox_dec_min", "bbox_dec_max", "total_flux_jy"}


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(v): int(v) for v in values}
        self.rank = {int(v): 0 for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def labels(self) -> dict[int, int]:
        roots: dict[int, int] = {}
        labels: dict[int, int] = {}
        for value in sorted(self.parent):
            root = self.find(value)
            if root not in roots:
                roots[root] = len(roots)
            labels[value] = roots[root]
        return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dr1_validation/baseline_variants.yaml"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dr1-catalogue", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--baseline", default=None, help="Optional baseline ID/name to run.")
    parser.add_argument("--max-fields", type=int, default=None, help="Maximum DR1-overlapping fields to process.")
    parser.add_argument("--beam-arcsec", type=float, default=6.0)
    parser.add_argument("--distance-threshold-beam", type=float, default=3.0)
    parser.add_argument("--unconstrained-threshold-beam", type=float, default=9.0)
    parser.add_argument("--dr1-margin-deg", type=float, default=0.75)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _select_baselines(config: dict[str, Any], selected: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline_id, payload in (config.get("baselines") or {}).items():
        name = str(payload.get("name", baseline_id))
        if selected and selected not in {baseline_id, name}:
            continue
        rows.append(
            {
                "baseline_id": baseline_id,
                "variant": name,
                "bbox_policy": str(payload.get("bbox_policy", "")),
                "implemented": baseline_id in CORE_BASELINES,
            }
        )
    return rows


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    return None


def _numeric(frame: pd.DataFrame, name: str | None, default: float = np.nan) -> pd.Series:
    if name is None or name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _normalise_components(raw: pd.DataFrame, field_id: str, image_path: str, dr1_bounds: dict[str, float]) -> pd.DataFrame:
    columns = list(raw.columns)
    ra_col = _find_col(columns, ["RA", "ra", "RA_deg"])
    dec_col = _find_col(columns, ["DEC", "Dec", "dec", "DEJ2000"])
    if ra_col is None or dec_col is None:
        return pd.DataFrame()
    total_col = _find_col(columns, ["Total_flux", "total_flux"])
    peak_col = _find_col(columns, ["Peak_flux", "peak_flux"])
    rms_col = _find_col(columns, ["Isl_rms", "local_rms", "Wave_Isl_rms", "Resid_Isl_rms"])
    maj_col = _find_col(columns, ["Maj", "DC_Maj", "major_axis"])
    min_col = _find_col(columns, ["Min", "DC_Min", "minor_axis"])
    pa_col = _find_col(columns, ["PA", "DC_PA", "pa"])
    gaus_col = _find_col(columns, ["Gaus_id", "gaus_id", "component_id"])
    isl_col = _find_col(columns, ["Isl_id", "island_id"])
    src_col = _find_col(columns, ["Source_id", "source_id"])
    scode_col = _find_col(columns, ["S_Code", "scode", "morphology_class"])

    out = pd.DataFrame(index=raw.index)
    out["field_id"] = field_id
    out["image_path"] = image_path
    out["component_row"] = np.arange(len(raw), dtype=np.int64)
    out["gaussian_id"] = raw[gaus_col].astype(str).to_numpy() if gaus_col else out["component_row"].astype(str)
    out["component_id"] = field_id + ":" + out["gaussian_id"].astype(str)
    out["ra"] = _numeric(raw, ra_col) % 360.0
    out["dec"] = _numeric(raw, dec_col)
    out["total_flux_jy"] = _numeric(raw, total_col, 0.0).fillna(0.0).clip(lower=0.0)
    out["peak_flux_jy"] = _numeric(raw, peak_col, 0.0).fillna(0.0).clip(lower=0.0)
    out["local_rms_jy"] = _numeric(raw, rms_col, np.nan)
    out["major_deg"] = _numeric(raw, maj_col, args_fallback_major_deg())
    out["minor_deg"] = _numeric(raw, min_col, args_fallback_minor_deg())
    out["pa_deg"] = _numeric(raw, pa_col, 0.0).fillna(0.0)
    out["island_id"] = raw[isl_col].astype(str).to_numpy() if isl_col else out["gaussian_id"].astype(str)
    out["pybdsf_source_id"] = raw[src_col].astype(str).to_numpy() if src_col else out["island_id"].astype(str)
    out["s_code"] = raw[scode_col].astype(str).to_numpy() if scode_col else ""
    out["peak_snr"] = out["peak_flux_jy"] / out["local_rms_jy"].replace(0.0, np.nan)
    valid = out["ra"].between(dr1_bounds["ra_min"], dr1_bounds["ra_max"], inclusive="both") & out["dec"].between(
        dr1_bounds["dec_min"], dr1_bounds["dec_max"], inclusive="both"
    )
    out = out.loc[valid & out["ra"].notna() & out["dec"].notna()].copy()
    if out.empty:
        return out
    out["component_index"] = np.arange(len(out), dtype=np.int64)
    out["major_deg"] = out["major_deg"].fillna(args_fallback_major_deg()).clip(lower=args_fallback_major_deg())
    out["minor_deg"] = out["minor_deg"].fillna(args_fallback_minor_deg()).clip(lower=args_fallback_minor_deg())
    return out


def args_fallback_major_deg() -> float:
    return 6.0 / 3600.0


def args_fallback_minor_deg() -> float:
    return 5.0 / 3600.0


def _contour_scale(components: pd.DataFrame, sigma: float) -> np.ndarray:
    peak = pd.to_numeric(components["peak_flux_jy"], errors="coerce").to_numpy(float)
    rms = pd.to_numeric(components["local_rms_jy"], errors="coerce").replace(0.0, np.nan).to_numpy(float)
    threshold = float(sigma) * rms
    ratio = np.divide(peak, threshold, out=np.ones_like(peak), where=np.isfinite(threshold) & (threshold > 0))
    ratio = np.clip(ratio, 1.01, 1.0e6)
    return np.sqrt(np.log(ratio) / np.log(2.0))


def _component_bboxes(components: pd.DataFrame, scale: np.ndarray | float) -> pd.DataFrame:
    out = components.copy()
    scale_arr = np.full(len(out), float(scale), dtype=float) if np.isscalar(scale) else np.asarray(scale, dtype=float)
    semi_major = 0.5 * pd.to_numeric(out["major_deg"], errors="coerce").to_numpy(float) * scale_arr
    semi_minor = 0.5 * pd.to_numeric(out["minor_deg"], errors="coerce").to_numpy(float) * scale_arr
    pa = np.deg2rad(pd.to_numeric(out["pa_deg"], errors="coerce").fillna(0.0).to_numpy(float))
    ra = pd.to_numeric(out["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(out["dec"], errors="coerce").to_numpy(float)
    x_half = np.sqrt((semi_major * np.sin(pa)) ** 2 + (semi_minor * np.cos(pa)) ** 2)
    y_half = np.sqrt((semi_major * np.cos(pa)) ** 2 + (semi_minor * np.sin(pa)) ** 2)
    cos_dec = np.clip(np.cos(np.deg2rad(dec)), 0.05, 1.0)
    ra_half = x_half / cos_dec
    out["component_bbox_ra_min"] = (ra - ra_half) % 360.0
    out["component_bbox_ra_max"] = (ra + ra_half) % 360.0
    out["component_bbox_dec_min"] = np.clip(dec - y_half, -90.0, 90.0)
    out["component_bbox_dec_max"] = np.clip(dec + y_half, -90.0, 90.0)
    out["contour_radius_deg"] = np.maximum(semi_major, semi_minor)
    return out


def _coords_deg(components: pd.DataFrame) -> np.ndarray:
    dec0 = float(pd.to_numeric(components["dec"], errors="coerce").median())
    cos_dec = max(math.cos(math.radians(dec0)), 0.05)
    ra = pd.to_numeric(components["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(components["dec"], errors="coerce").to_numpy(float)
    return np.column_stack([ra * cos_dec, dec])


def _labels_from_key(components: pd.DataFrame, key_col: str) -> pd.Series:
    keys = components[key_col].astype(str).fillna("")
    labels = keys.factorize(sort=True)[0]
    return pd.Series(labels, index=components.index, dtype=int)


def _labels_from_distance(components: pd.DataFrame, threshold_deg: float) -> pd.Series:
    values = components["component_index"].astype(int).tolist()
    uf = UnionFind(values)
    if len(values) >= 2:
        tree = cKDTree(_coords_deg(components))
        for left, right in tree.query_pairs(float(threshold_deg), output_type="set"):
            uf.union(int(left), int(right))
    labels = uf.labels()
    return components["component_index"].astype(int).map(labels).astype(int)


def _labels_from_contours(components: pd.DataFrame, sigma: float, *, max_search_deg: float = 0.25) -> pd.Series:
    work = _component_bboxes(components, _contour_scale(components, sigma))
    values = work["component_index"].astype(int).tolist()
    uf = UnionFind(values)
    if len(values) >= 2:
        coords = _coords_deg(work)
        radius = pd.to_numeric(work["contour_radius_deg"], errors="coerce").fillna(0.0).to_numpy(float)
        max_radius = float(np.nanmax(radius)) if len(radius) else 0.0
        tree = cKDTree(coords)
        for i, rad in enumerate(radius):
            query_radius = min(float(rad + max_radius), float(max_search_deg))
            if query_radius <= 0:
                continue
            for j in tree.query_ball_point(coords[i], query_radius):
                if j <= i:
                    continue
                if np.linalg.norm(coords[i] - coords[j]) <= radius[i] + radius[j]:
                    uf.union(int(i), int(j))
    labels = uf.labels()
    return work["component_index"].astype(int).map(labels).astype(int)


def _labels_unconstrained(components: pd.DataFrame, beam_arcsec: float, threshold_beam: float) -> pd.Series:
    work = _component_bboxes(components, _contour_scale(components, 2.0))
    values = work["component_index"].astype(int).tolist()
    uf = UnionFind(values)
    if len(values) >= 2:
        coords = _coords_deg(work)
        radius = pd.to_numeric(work["contour_radius_deg"], errors="coerce").fillna(0.0).to_numpy(float)
        distance_floor = float(beam_arcsec) * float(threshold_beam) / 3600.0
        max_radius = min(max(float(np.nanmax(radius)) if len(radius) else 0.0, distance_floor), 0.30)
        tree = cKDTree(coords)
        for i, rad in enumerate(radius):
            query_radius = min(float(rad + max_radius + distance_floor), 0.30)
            for j in tree.query_ball_point(coords[i], query_radius):
                if j <= i:
                    continue
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if dist <= max(radius[i] + radius[j], distance_floor):
                    uf.union(int(i), int(j))
    labels = uf.labels()
    return work["component_index"].astype(int).map(labels).astype(int)


def _group_predictions(
    components: pd.DataFrame,
    labels: pd.Series,
    baseline_id: str,
    variant: str,
    bbox_policy: str,
    bbox_definition: str,
    scale: np.ndarray | float,
) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame()
    work = _component_bboxes(components, scale)
    work["_group_label"] = labels.to_numpy()
    rows: list[dict[str, Any]] = []
    for label, group in work.groupby("_group_label", sort=True):
        flux = pd.to_numeric(group["total_flux_jy"], errors="coerce").fillna(0.0)
        weight = flux.where(flux > 0, 1.0)
        total_weight = float(weight.sum()) if float(weight.sum()) > 0 else float(len(group))
        ra_centre = float((pd.to_numeric(group["ra"], errors="coerce") * weight).sum() / total_weight)
        dec_centre = float((pd.to_numeric(group["dec"], errors="coerce") * weight).sum() / total_weight)
        rows.append(
            {
                "source_id": f"{baseline_id}:{group['field_id'].iloc[0]}:{int(label)}",
                "sample": "baseline_all",
                "ra_centre": ra_centre % 360.0,
                "dec_centre": dec_centre,
                "bbox_ra_min": float(pd.to_numeric(group["component_bbox_ra_min"], errors="coerce").min()) % 360.0,
                "bbox_ra_max": float(pd.to_numeric(group["component_bbox_ra_max"], errors="coerce").max()) % 360.0,
                "bbox_dec_min": float(pd.to_numeric(group["component_bbox_dec_min"], errors="coerce").min()),
                "bbox_dec_max": float(pd.to_numeric(group["component_bbox_dec_max"], errors="coerce").max()),
                "total_flux_jy": float(flux.sum()),
                "peak_flux_jy": float(pd.to_numeric(group["peak_flux_jy"], errors="coerce").max()),
                "local_rms_jy": float(pd.to_numeric(group["local_rms_jy"], errors="coerce").median()),
                "peak_snr": float(pd.to_numeric(group["peak_snr"], errors="coerce").max()),
                "n_gaussians": int(len(group)),
                "field_id": str(group["field_id"].iloc[0]),
                "baseline_id": baseline_id,
                "variant": variant,
                "bbox_policy": bbox_policy,
                "bbox_definition": bbox_definition,
                "no_dr1_optical_ids_used": True,
            }
        )
    return pd.DataFrame(rows)


def _run_baseline_for_field(
    components: pd.DataFrame,
    baseline: dict[str, Any],
    beam_arcsec: float,
    distance_threshold_beam: float,
    unconstrained_threshold_beam: float,
) -> pd.DataFrame:
    baseline_id = str(baseline["baseline_id"])
    variant = str(baseline["variant"])
    bbox_policy = str(baseline["bbox_policy"])
    if baseline_id == "B0":
        labels = _labels_from_key(components, "island_id")
        return _group_predictions(components, labels, baseline_id, variant, bbox_policy, "union of PyBDSF Gaussian FWHM bboxes grouped by native Isl_id; no padding", 1.0)
    if baseline_id == "B1":
        labels = _labels_from_contours(components, 3.0)
        return _group_predictions(
            components,
            labels,
            baseline_id,
            variant,
            bbox_policy,
            "union of Gaussian-model 3sigma contour bboxes connected by contour contact; no padding",
            _contour_scale(components, 3.0),
        )
    if baseline_id == "B2":
        labels = _labels_from_contours(components, 2.0)
        return _group_predictions(
            components,
            labels,
            baseline_id,
            variant,
            bbox_policy,
            "union of Gaussian-model 2sigma contour bboxes connected by contour contact; no padding",
            _contour_scale(components, 2.0),
        )
    if baseline_id == "B3":
        labels = _labels_from_distance(components, float(beam_arcsec) * float(distance_threshold_beam) / 3600.0)
        return _group_predictions(
            components,
            labels,
            baseline_id,
            variant,
            bbox_policy,
            f"union of Gaussian FWHM bboxes connected only by distance <= {distance_threshold_beam:g} beams; no padding",
            1.0,
        )
    if baseline_id == "B4":
        labels = _labels_unconstrained(components, beam_arcsec, unconstrained_threshold_beam)
        return _group_predictions(
            components,
            labels,
            baseline_id,
            variant,
            bbox_policy,
            f"union of 2sigma Gaussian contour bboxes with unconstrained graph connectivity and {unconstrained_threshold_beam:g}-beam distance floor; no padding",
            _contour_scale(components, 2.0),
        )
    return pd.DataFrame()


def _row_value(row: Any, names: list[str]) -> str:
    for name in names:
        value = getattr(row, name, "")
        if str(value).strip():
            return str(value)
    return ""


def _field_header_overlaps_dr1(row: Any, dr1_bounds: dict[str, float]) -> bool:
    path = Path(_row_value(row, ["scratch_fits_path", "fits_path", "h5_path"]))
    if not path.exists() or path.suffix.lower() in {".h5", ".hdf5"}:
        return True
    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
        nx = int(header.get("NAXIS1", 0))
        ny = int(header.get("NAXIS2", 0))
        if nx <= 0 or ny <= 0:
            return True
        wcs = WCS(header, naxis=2)
        pixels = np.asarray(
            [
                [1.0, 1.0],
                [float(nx), 1.0],
                [1.0, float(ny)],
                [float(nx), float(ny)],
                [0.5 * float(nx), 0.5 * float(ny)],
            ]
        )
        world = wcs.all_pix2world(pixels, 1)
        ra = pd.to_numeric(pd.Series(world[:, 0]), errors="coerce") % 360.0
        dec = pd.to_numeric(pd.Series(world[:, 1]), errors="coerce")
        if ra.notna().sum() < 2 or dec.notna().sum() < 2:
            return True
        ra_min = float(ra.min())
        ra_max = float(ra.max())
        dec_min = float(dec.min())
        dec_max = float(dec.max())
        return not (
            ra_max < dr1_bounds["ra_min"]
            or ra_min > dr1_bounds["ra_max"]
            or dec_max < dr1_bounds["dec_min"]
            or dec_min > dr1_bounds["dec_max"]
        )
    except Exception:
        return True


def _filter_manifest_by_dr1_header(manifest: pd.DataFrame, dr1_bounds: dict[str, float]) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    keep = []
    for row in manifest.itertuples(index=False):
        keep.append(_field_header_overlaps_dr1(row, dr1_bounds))
    out = manifest.loc[pd.Series(keep, index=manifest.index)].copy()
    return out if not out.empty else manifest


def _build_catalogues(
    manifest: pd.DataFrame,
    baselines: list[dict[str, Any]],
    dr1_bounds: dict[str, float],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    implemented = [row for row in baselines if row["implemented"]]
    rows_by_baseline: dict[str, list[pd.DataFrame]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    processed_fields = 0
    scanned_fields = 0
    candidate_manifest = _filter_manifest_by_dr1_header(manifest, dr1_bounds)
    for row in candidate_manifest.itertuples(index=False):
        scanned_fields += 1
        field_id = str(getattr(row, "file_id", "") or getattr(row, "field_name", ""))
        path = Path(str(getattr(row, "pybdsf_catalog_path", "")))
        image_path = str(getattr(row, "scratch_fits_path", "") or getattr(row, "fits_path", "") or getattr(row, "h5_path", ""))
        if not field_id or not path.exists():
            failures.append({"field_id": field_id, "baseline": "all", "status": "missing_pybdsf_catalogue", "reason": str(path)})
            continue
        try:
            components = _normalise_components(read_table_any(path), field_id, image_path, dr1_bounds)
        except Exception as exc:
            failures.append({"field_id": field_id, "baseline": "all", "status": "read_failed", "reason": f"{exc.__class__.__name__}: {exc}"})
            continue
        if components.empty:
            continue
        processed_fields += 1
        for baseline in implemented:
            try:
                pred = _run_baseline_for_field(components, baseline, args.beam_arcsec, args.distance_threshold_beam, args.unconstrained_threshold_beam)
                if not pred.empty:
                    rows_by_baseline[str(baseline["baseline_id"])].append(pred)
            except Exception as exc:
                failures.append(
                    {
                        "field_id": field_id,
                        "baseline": str(baseline["baseline_id"]),
                        "status": "baseline_failed",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
        if args.max_fields is not None and processed_fields >= int(args.max_fields):
            break
    paths: dict[str, Path] = {}
    catalogue_dir = output_dir / "baseline_catalogues"
    catalogue_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    for baseline in implemented:
        baseline_id = str(baseline["baseline_id"])
        variant = str(baseline["variant"])
        frame = pd.concat(rows_by_baseline.get(baseline_id, []), ignore_index=True) if rows_by_baseline.get(baseline_id) else pd.DataFrame()
        path = catalogue_dir / f"{baseline_id}_{variant}_predictions.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        paths[baseline_id] = path
        inventory_rows.append(
            {
                **baseline,
                "prediction_catalogue": str(path),
                "n_prediction_rows": int(len(frame)),
                "n_singletons": int((pd.to_numeric(frame.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0) <= 1).sum()) if not frame.empty else 0,
                "n_multi_component_groups": int((pd.to_numeric(frame.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 2).sum()) if not frame.empty else 0,
                "max_group_size": int(pd.to_numeric(frame.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").max()) if not frame.empty else 0,
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    manifest_stats = pd.DataFrame(
        [
            {
                "candidate_fields_after_header_filter": int(len(candidate_manifest)),
                "scanned_fields": int(scanned_fields),
                "processed_dr1_overlapping_fields": int(processed_fields),
                "mode": args.mode,
                "max_fields": args.max_fields,
            }
        ]
    )
    failed = pd.DataFrame(failures, columns=["field_id", "baseline", "status", "reason"])
    return paths, pd.concat([inventory, manifest_stats], axis=1), failed


def _read_prediction_catalogue(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing evaluator columns: {missing}")
    return frame


def _diagnostics(predictions: pd.DataFrame, baseline_id: str, variant: str, bbox_policy: str) -> dict[str, Any]:
    n_gauss = pd.to_numeric(predictions.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "baseline_id": baseline_id,
        "variant": variant,
        "bbox_policy": bbox_policy,
        "n_sources": int(len(predictions)),
        "singleton_count": int((n_gauss <= 1).sum()),
        "multi_component_group_count": int((n_gauss >= 2).sum()),
        "max_group_size": int(n_gauss.max()) if len(n_gauss) else 0,
        "median_group_size": float(n_gauss.median()) if len(n_gauss) else np.nan,
    }


def _evaluate_catalogues(
    baselines: list[dict[str, Any]],
    paths: dict[str, Path],
    dr1: pd.DataFrame,
    footprint: Any,
    flux_cuts: list[float],
    shifts: list[list[float]],
    grid_size: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_rows: list[pd.DataFrame] = []
    random_rows: list[pd.DataFrame] = []
    corrected_rows: list[pd.DataFrame] = []
    area_rows: list[pd.DataFrame] = []
    area_bin_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    read_checks: list[dict[str, Any]] = []
    for baseline in baselines:
        if not baseline["implemented"]:
            continue
        baseline_id = str(baseline["baseline_id"])
        variant = str(baseline["variant"])
        path = paths.get(baseline_id)
        if path is None or not path.exists():
            read_checks.append({"baseline_id": baseline_id, "variant": variant, "path": str(path or ""), "evaluator_readable": False, "reason": "missing"})
            continue
        try:
            predictions = _read_prediction_catalogue(path)
            read_checks.append({"baseline_id": baseline_id, "variant": variant, "path": str(path), "evaluator_readable": True, "reason": ""})
        except Exception as exc:
            read_checks.append({"baseline_id": baseline_id, "variant": variant, "path": str(path), "evaluator_readable": False, "reason": f"{exc.__class__.__name__}: {exc}"})
            continue
        predictions = _add_bbox_area(filter_predictions_in_footprint(predictions, footprint))
        predictions = match_predictions_to_dr1_components(predictions, dr1, grid_size_deg=grid_size)
        predictions.to_csv(output_dir / "baseline_catalogues" / f"{baseline_id}_{variant}_predictions_with_dr1_support.csv.gz", index=False, compression="gzip")
        observed_rows.append(compute_support_rates(predictions, flux_cuts, variant=variant))
        random_rows.append(_shifted_random_rates(predictions, dr1, shifts, flux_cuts, grid_size, variant))
        area, area_bins = _area_diagnostics(predictions, flux_cuts, variant)
        if not area.empty:
            area_rows.append(area)
        if not area_bins.empty:
            area_bin_rows.append(area_bins)
        diagnostics.append(_diagnostics(predictions, baseline_id, variant, str(baseline["bbox_policy"])))
    observed = pd.concat(observed_rows, ignore_index=True) if observed_rows else pd.DataFrame()
    random_rates = pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame()
    corrected = _corrected(observed, random_rates) if not observed.empty else pd.DataFrame()
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = corrected.merge(diagnostic_frame[["variant", "singleton_count", "multi_component_group_count", "max_group_size"]], on="variant", how="left") if not corrected.empty else pd.DataFrame()
    support_cols = [
        "variant",
        "sample",
        "flux_cut_jy",
        "n_in_dr1_footprint",
        "n_supported_by_dr1_component",
        "observed_support_rate",
        "binomial_or_wilson_95ci_low",
        "binomial_or_wilson_95ci_high",
        "random_support_rate_mean",
        "random_support_rate_std",
        "chance_corrected_support_rate",
        "singleton_count",
        "multi_component_group_count",
        "max_group_size",
    ]
    for col in support_cols:
        if col not in summary:
            summary[col] = np.nan
    summary = summary[support_cols]
    summary.to_csv(output_dir / "baseline_support_summary.csv", index=False)
    summary.to_latex(output_dir / "baseline_support_summary.tex", index=False, float_format="%.4f")
    corrected.to_csv(output_dir / "baseline_random_shift_corrected_support.csv", index=False)
    observed.to_csv(output_dir / "baseline_observed_support.csv", index=False)
    random_rates.to_csv(output_dir / "baseline_random_shift_support_by_shift.csv", index=False)
    (pd.concat(area_rows, ignore_index=True) if area_rows else pd.DataFrame()).to_csv(output_dir / "baseline_bbox_area_diagnostics.csv", index=False)
    (pd.concat(area_bin_rows, ignore_index=True) if area_bin_rows else pd.DataFrame()).to_csv(output_dir / "baseline_bbox_area_bin_support.csv", index=False)
    diagnostic_frame.to_csv(output_dir / "baseline_diagnostics_summary.csv", index=False)
    pd.DataFrame(read_checks).to_csv(output_dir / "baseline_evaluator_read_check.csv", index=False)
    return summary, pd.DataFrame(read_checks)


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir or config.get("outputs", {}).get("root", PROJECT_ROOT / "outputs/dr1_validation/baselines"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    baselines = _select_baselines(config, args.baseline)
    if not baselines:
        raise SystemExit(f"No baselines selected by {args.baseline!r}")
    if args.mode == "smoke" and args.max_fields is None:
        args.max_fields = 3

    ref = config.get("reference", {}) or {}
    dr1_path = args.dr1_catalogue or ref.get("dr1_catalogue", str(DEFAULT_DR1_COMPONENT_CSV))
    flux_cuts = [float(x) for x in ref.get("flux_cuts_jy", [0.0, 0.05, 0.1])]
    shifts = ref.get("random_shifts_ra_dec_deg", [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    grid_size = float(ref.get("footprint_grid_size_deg", 0.25))
    dr1, dr1_summary = load_dr1_component_catalogue(dr1_path, report_path=output_dir / "dr1_component_catalogue_report.txt")
    valid = dr1.loc[dr1["valid_position"].astype(bool)]
    dr1_bounds = {
        "ra_min": float(valid["ra"].min()) - float(args.dr1_margin_deg),
        "ra_max": float(valid["ra"].max()) + float(args.dr1_margin_deg),
        "dec_min": float(valid["dec"].min()) - float(args.dr1_margin_deg),
        "dec_max": float(valid["dec"].max()) + float(args.dr1_margin_deg),
    }
    footprint = build_dr1_sky_footprint(dr1, grid_size_deg=grid_size)
    manifest = read_manifest(args.manifest)
    if manifest.empty:
        raise SystemExit(f"Manifest is missing or empty: {args.manifest}")

    skipped = pd.DataFrame(
        [
            {
                "baseline_id": row["baseline_id"],
                "variant": row["variant"],
                "status": "skipped_or_pending",
                "reason": "not implemented in tonight guarded baseline batch; B0-B4 are the required runnable core",
            }
            for row in baselines
            if not row["implemented"]
        ]
    )
    skipped.to_csv(output_dir / "baseline_skipped_methods.csv", index=False)

    paths, inventory, failures = _build_catalogues(manifest, baselines, dr1_bounds, output_dir, args)
    inventory.to_csv(output_dir / "baseline_variant_inventory.csv", index=False)
    failures.to_csv(output_dir / "baseline_failed_methods.csv", index=False)
    summary, read_checks = _evaluate_catalogues(baselines, paths, dr1, footprint, flux_cuts, shifts, grid_size, output_dir)

    implemented_core = [row for row in baselines if row["baseline_id"] in CORE_BASELINES]
    readable_core = set(read_checks.loc[read_checks.get("evaluator_readable", pd.Series(dtype=bool)).astype(bool), "baseline_id"].astype(str)) if not read_checks.empty else set()
    core_ok = all(row["baseline_id"] in readable_core for row in implemented_core) and not summary.empty
    metadata = {
        "mode": args.mode,
        "config": str(args.config),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "runtime_seconds": round(perf_counter() - t0, 3),
        "reference": ref,
        "dr1_summary": dr1_summary,
        "dr1_component_only": True,
        "no_dr1_optical_ids_used": True,
        "bbox_padding_policy": "No artificial bbox padding; bboxes are unions of Gaussian FWHM or Gaussian contour extents according to baseline metadata.",
        "chance_correction": "F_corr = (F_obs - F_rand) / (1 - F_rand)",
        "implemented_methods": [row["baseline_id"] for row in baselines if row["implemented"]],
        "skipped_methods": [row["baseline_id"] for row in baselines if not row["implemented"]],
        "core_B0_B4_evaluator_readable": bool(core_ok),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote baseline DR1 validation outputs to {output_dir}")
    if args.mode == "full" and not core_ok:
        raise SystemExit("B0-B4 did not all generate evaluator-readable prediction catalogues.")


if __name__ == "__main__":
    main()
