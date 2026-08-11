#!/usr/bin/env python
"""Full LoTSS DR3 production wrapper for the stable association pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lofar_det_vsex.catalog import normalized_gaussian_dataframe, read_gaussian_catalog
from lofar_det_vsex.host_query import HOST_QUERY_LOG_COLUMNS, HostQueryClient
from lofar_det_vsex.io import H5CutoutReader
from lofar_det_vsex.io import Cutout
from lofar_det_vsex.parent_links import (
    SOURCE_MORPH_TABLE_COLUMNS,
    PARENT_CANDIDATE_COLUMNS,
    PARENT_DIAGNOSTIC_COLUMNS,
    PARENT_EDGE_DEBUG_COLUMNS,
    PARENT_HOST_CANDIDATE_COLUMNS,
    run_parent_links,
    parent_link_config,
)
from lofar_det_vsex.segmentation import load_segmentation
from lofar_det_vsex.utils import load_yaml, setup_logging, write_dataframe
from lofar_det_vsex.visualize import plot_parent_link_cutout_all, plot_parent_link_parent_zoom
from scripts.run_pipeline import combine_partials, ensure_output_tree as ensure_local_output_tree, process_cutout
from lotss_dr3_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_H5_ROOT,
    DEFAULT_ORIGINAL_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FINAL_MERGED_COLUMNS,
    HOST_COLUMNS,
    PROCESSING_VERSION,
    bool_text,
    copy_repro_files,
    default_pybdsf_catalog_path,
    ensure_output_dirs,
    field_id_from_path,
    find_existing_pybdsf_catalog,
    h5_map,
    list_fits,
    list_h5,
    pybdsf_catalog_sane,
    read_manifest,
    scratch_path_for,
    write_csv_gz,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-data-root", type=Path, default=DEFAULT_ORIGINAL_DATA_ROOT)
    parser.add_argument("--h5-root", type=Path, default=DEFAULT_H5_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_lotss_conservative.yaml")
    parser.add_argument("--manifest", type=Path, default=None, help="Use an existing LoTSS manifest instead of rebuilding it.")
    parser.add_argument("--skip-data-benchmark", action="store_true", help="Skip FITS/H5 benchmark and use --input-format.")
    parser.add_argument("--input-format", choices=["fits", "h5"], default="fits")
    parser.add_argument("--release-tag", default="release", help="Short label recorded in run metadata and reports.")
    parser.add_argument("--use-existing-pybdsf", action="store_true")
    parser.add_argument("--run-missing-pybdsf", action="store_true")
    parser.add_argument("--query-wise-host", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Smoke limit: fields and cutouts per field.")
    parser.add_argument("--debug-sample-figures", type=int, default=0)
    parser.add_argument("--save-all-parent-zoom", action="store_true")
    parser.add_argument("--max-host-queries-per-field", type=int, default=1000)
    parser.add_argument("--pybdsf-frequency-hz", type=float, default=144000000.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _concat(frames: list[pd.DataFrame], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns or [])


def _build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    dirs = ensure_output_dirs(args.output_root)
    search_roots = [dirs.pybdsf_raw, dirs.pybdsf_processed]
    h5_by_field = h5_map([args.h5_root])
    rows = []
    for fits_path in list_fits(args.original_data_root):
        field_id = field_id_from_path(fits_path)
        scratch_path = scratch_path_for(fits_path, args.original_data_root, args.data_root)
        h5_path = h5_by_field.get(field_id)
        catalog = find_existing_pybdsf_catalog(field_id, search_roots, raw_root=dirs.pybdsf_raw)
        if catalog is None:
            catalog = default_pybdsf_catalog_path(dirs.pybdsf_raw, field_id)
            ok = False
        else:
            ok, _, _ = pybdsf_catalog_sane(catalog)
        rows.append(
            {
                "file_id": field_id,
                "fits_path": str(fits_path),
                "scratch_fits_path": str(scratch_path),
                "h5_path": str(h5_path) if h5_path else "",
                "field_name": field_id,
                "has_existing_pybdsf_catalog": str(bool(ok)),
                "pybdsf_catalog_path": str(catalog),
                "needs_pybdsf": str(not ok),
                "status": "ready" if ok else "needs_pybdsf",
            }
        )
    frame = pd.DataFrame(rows)
    write_manifest(frame, dirs.manifests / "lotss_dr3_fits_manifest.csv")
    return frame


def _benchmark_data_format(args: argparse.Namespace, manifest: pd.DataFrame) -> dict[str, Any]:
    dirs = ensure_output_dirs(args.output_root)
    rows = manifest[(manifest["h5_path"].astype(str) != "")].head(3).copy()
    records = []
    for _, row in rows.iterrows():
        field_id = str(row["file_id"])
        fits_path = Path(str(row.get("scratch_fits_path") or row.get("fits_path")))
        if not fits_path.exists():
            fits_path = Path(str(row.get("fits_path")))
        h5_path = Path(str(row["h5_path"]))
        fit_sec = np.nan
        h5_sec = np.nan
        fit_ok = False
        h5_ok = False
        try:
            from astropy.io import fits

            t0 = time.perf_counter()
            with fits.open(fits_path, memmap=True) as handle:
                _ = handle[0].header
                data = handle[0].data
                y0 = max(0, data.shape[-2] // 2 - 1024)
                x0 = max(0, data.shape[-1] // 2 - 1024)
                cut = np.asarray(data[y0 : y0 + 2048, x0 : x0 + 2048])
                fit_ok = cut.size > 0 and np.isfinite(cut).any()
            fit_sec = time.perf_counter() - t0
        except Exception:
            fit_ok = False
        try:
            t0 = time.perf_counter()
            reader = H5CutoutReader(h5_path)
            cutout = reader.read(0)
            h5_ok = cutout.image.size > 0 and cutout.wcs is not None
            h5_sec = time.perf_counter() - t0
        except Exception:
            h5_ok = False
        records.append(
            {
                "field_id": field_id,
                "fits_path": str(fits_path),
                "h5_path": str(h5_path),
                "fits_read_sec": fit_sec,
                "h5_read_sec": h5_sec,
                "fits_ok": fit_ok,
                "h5_ok": h5_ok,
            }
        )
    bench = pd.DataFrame(records)
    fits_count = len(list_fits(args.original_data_root))
    h5_count = len(list_h5(args.h5_root))
    h5_complete = h5_count >= fits_count and fits_count > 0
    h5_stable = bool(len(bench) and bench["h5_ok"].astype(bool).all())
    fits_stable = bool(len(bench) and bench["fits_ok"].astype(bool).all())
    h5_median = float(pd.to_numeric(bench.get("h5_read_sec", pd.Series(dtype=float)), errors="coerce").median()) if len(bench) else np.nan
    fits_median = float(pd.to_numeric(bench.get("fits_read_sec", pd.Series(dtype=float)), errors="coerce").median()) if len(bench) else np.nan
    fits_faster = bool(fits_stable and h5_stable and np.isfinite(fits_median) and np.isfinite(h5_median) and fits_median <= h5_median)
    selected = "fits" if fits_stable and (fits_faster or not (h5_complete and h5_stable)) else "h5"
    if selected == "fits" and fits_faster:
        reason = "FITS memmap cutout reads were faster in the 3-field benchmark; PyBDSF also consumes FITS directly."
    elif selected == "fits":
        reason = "H5 is incomplete or unstable; using scratch FITS for production parent-linking input."
    else:
        reason = "H5 is complete, stable, and faster in the benchmark; PyBDSF still uses FITS."
    lines = [
        "# LoTSS DR3 Data Format Benchmark",
        "",
        f"- fits_count: {fits_count}",
        f"- h5_count: {h5_count}",
        f"- h5_complete: {h5_complete}",
        f"- h5_reader_stable: {h5_stable}",
        f"- fits_reader_stable: {fits_stable}",
        f"- median_fits_2048_cutout_read_sec: {fits_median:.4f}" if np.isfinite(fits_median) else "- median_fits_2048_cutout_read_sec: nan",
        f"- median_h5_first_cutout_read_sec: {h5_median:.4f}" if np.isfinite(h5_median) else "- median_h5_first_cutout_read_sec: nan",
        f"- selected_association_input_format: {selected}",
        f"- reason: {reason}",
        "",
        "## Samples",
        "",
        bench.to_markdown(index=False) if not bench.empty else "No paired H5/FITS samples found.",
        "",
    ]
    (dirs.manifests / "data_format_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "fits_count": fits_count,
        "h5_count": h5_count,
        "selected": selected,
        "reason": reason,
        "h5_complete": h5_complete,
        "h5_stable": h5_stable,
        "fits_stable": fits_stable,
    }


class FitsTileCutoutReader:
    """Read the same 2048 tile grid directly from a full LoTSS FITS image."""

    # 按固定窗口把整张 FITS tile 切成重叠 cutout，保证边界附近的大尺度结构仍可被后续 rescue 逻辑检查。
    def __init__(self, fits_path: str | Path, cutout_size: int = 2048, overlap: int = 205):
        from astropy.io import fits
        from astropy.wcs import WCS

        self.fits_path = Path(fits_path)
        self.cutout_size = int(cutout_size)
        self.overlap = int(overlap)
        self.stride = self.cutout_size - self.overlap
        self.handle = fits.open(self.fits_path, memmap=True)
        self.header = self.handle[0].header.copy()
        self.data = self.handle[0].data
        if self.data.ndim > 2:
            self.data = np.squeeze(self.data)
        self.height, self.width = self.data.shape[-2], self.data.shape[-1]
        self.nx = int(np.ceil(max(self.width - self.cutout_size, 0) / self.stride)) + 1
        self.ny = int(np.ceil(max(self.height - self.cutout_size, 0) / self.stride)) + 1
        self.wcs_full = WCS(self.header).celestial

    def __len__(self) -> int:
        return int(self.nx * self.ny)

    def iter_indices(self, start_index: int = 0, end_index: int | None = None, limit: int | None = None) -> list[int]:
        end = len(self) if end_index is None else min(int(end_index), len(self))
        indices = list(range(max(0, int(start_index)), end))
        if limit is not None:
            indices = indices[: int(limit)]
        return indices

    def read(self, index: int) -> Cutout:
        from astropy.wcs import WCS

        gy, gx = divmod(int(index), self.nx)
        x0 = gx * self.stride
        y0 = gy * self.stride
        x1 = min(x0 + self.cutout_size, self.width)
        y1 = min(y0 + self.cutout_size, self.height)
        image = np.full((self.cutout_size, self.cutout_size), np.nan, dtype=np.float32)
        crop = np.asarray(self.data[y0:y1, x0:x1], dtype=np.float32)
        image[: crop.shape[0], : crop.shape[1]] = crop
        header = self.header.copy()
        header["NAXIS"] = 2
        header["NAXIS1"] = self.cutout_size
        header["NAXIS2"] = self.cutout_size
        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) - float(x0)
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) - float(y0)
        wcs = WCS(header).celestial
        center_x = x0 + 0.5 * (self.cutout_size - 1)
        center_y = y0 + 0.5 * (self.cutout_size - 1)
        try:
            ra, dec = self.wcs_full.pixel_to_world_values(center_x, center_y)
        except Exception:
            ra, dec = None, None
        return Cutout(
            cutout_id=f"cutout_{index:06d}",
            image=image,
            rms=None,
            mean=None,
            ra=float(ra) if ra is not None else None,
            dec=float(dec) if dec is not None else None,
            header=header,
            wcs=wcs,
            index=index,
            metadata={
                "fits_path": str(self.fits_path),
                "layout": "fits_tile_grid",
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "cutout_size": self.cutout_size,
                "overlap": self.overlap,
                "stride": self.stride,
            },
        )

    def close(self) -> None:
        self.handle.close()


def _run_stage(command: list[str]) -> None:
    proc = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"Stage failed with code {proc.returncode}: {' '.join(command)}")


def _read_status(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame(columns=["file_id", "status", "message"])


def _write_field_status(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.nan, np.nan, np.nan, np.nan
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        vals = value[:4]
    else:
        text = str(value).strip()
        if not text:
            return np.nan, np.nan, np.nan, np.nan
        try:
            parsed = json.loads(text)
            vals = parsed[:4] if isinstance(parsed, list) else []
        except Exception:
            vals = [part.strip() for part in text.replace(";", ",").split(",")[:4]]
    try:
        return tuple(float(v) for v in vals[:4])  # type: ignore[return-value]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _split_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _host_query_status(edge: pd.Series) -> str:
    status = str(edge.get("query_status", "") or edge.get("host_status", "") or "")
    if "cache_hit" in status:
        return "cached"
    if "failed" in status:
        return "failed"
    if str(edge.get("host_status", "")) == "host_found":
        return "queried"
    if str(edge.get("host_status", "")) == "no_plausible_host":
        return "no_match"
    return status or "unknown"


def _local_group_rows(field_id: str, image_path: str, input_format: str, groups: pd.DataFrame, created_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in groups.iterrows():
        xmin, ymin, xmax, ymax = _parse_bbox(group.get("bounding_box", ""))
        rows.append(
            {
                "merged_source_id": f"local:{field_id}:{group.get('association_group_id')}",
                "field_id": field_id,
                "image_path": image_path,
                "input_format_used": input_format,
                "ra": group.get("ra", np.nan),
                "dec": group.get("dec", np.nan),
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
                "union_bbox": group.get("bounding_box", ""),
                "parent_union_box": "",
                "n_gaussians": group.get("n_gaussians", np.nan),
                "member_gaussian_ids": group.get("gaussian_ids", ""),
                "member_local_group_ids": group.get("association_group_id", ""),
                "association_type": group.get("association_type", "local_group"),
                "association_quality": group.get("association_quality", ""),
                "parent_candidate_quality": "",
                "parent_score": np.nan,
                "is_parent_candidate": False,
                "is_local_group_only": True,
                "LAS_arcsec": group.get("LAS_arcsec", np.nan),
                "LAS_beam": group.get("LAS_beam", np.nan),
                "area_3sigma_beam": np.nan,
                "major_axis_beam": np.nan,
                "minor_axis_beam": np.nan,
                "axis_ratio": group.get("axis_ratio", np.nan),
                "peak_snr": np.nan,
                "total_flux": group.get("total_flux_gaussian", np.nan),
                "flux_ratio": np.nan,
                "size_ratio": np.nan,
                "box_gap_beam_robust": np.nan,
                "center_distance_beam": np.nan,
                "axis_alignment_score": np.nan,
                "facing_score": np.nan,
                "symmetry_score": np.nan,
                "host_catalog": "",
                "host_ra": np.nan,
                "host_dec": np.nan,
                "host_sep_arcsec": np.nan,
                "host_score": np.nan,
                "host_quality": "",
                "W1": np.nan,
                "W2": np.nan,
                "W1_minus_W2": np.nan,
                "midpoint_host_found": False,
                "lobe1_peak_host_found": False,
                "lobe2_peak_host_found": False,
                "lobe_peak_host_contradiction": False,
                "pybdsf_status": "done",
                "association_status": "done",
                "host_query_status": "not_applicable",
                "host_cache_key": "",
                "processing_version": PROCESSING_VERSION,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)


def _parent_rows(
    field_id: str,
    image_path: str,
    input_format: str,
    groups: pd.DataFrame,
    candidates: pd.DataFrame,
    edges: pd.DataFrame,
    created_at: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=FINAL_MERGED_COLUMNS)
    group_lookup = {str(row.get("association_group_id")): row for _, row in groups.iterrows()}
    edge_lookup = {str(row.get("parent_candidate_id")): row for _, row in edges.iterrows()} if not edges.empty else {}
    rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        gid1 = str(cand.get("local_group_id_1", ""))
        gid2 = str(cand.get("local_group_id_2", ""))
        g1 = group_lookup.get(gid1, pd.Series(dtype=object))
        g2 = group_lookup.get(gid2, pd.Series(dtype=object))
        edge = edge_lookup.get(str(cand.get("parent_candidate_id", "")), cand)
        gauss = sorted(set(_split_ids(g1.get("gaussian_ids", "")) + _split_ids(g2.get("gaussian_ids", ""))))
        xmin = cand.get("parent_bbox_xmin", np.nan)
        xmax = cand.get("parent_bbox_xmax", np.nan)
        ymin = cand.get("parent_bbox_ymin", np.nan)
        ymax = cand.get("parent_bbox_ymax", np.nan)
        contradiction = str(cand.get("host_evidence", "")) == "contradicts_double_lobe" or str(cand.get("rejection_reason", "")) == "lobe_peak_host_contradiction"
        rows.append(
            {
                "merged_source_id": f"parent:{field_id}:{cand.get('parent_candidate_id')}",
                "field_id": field_id,
                "image_path": image_path,
                "input_format_used": input_format,
                "ra": edge.get("midpoint_ra", np.nan),
                "dec": edge.get("midpoint_dec", np.nan),
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
                "union_bbox": f"{xmin},{ymin},{xmax},{ymax}",
                "parent_union_box": f"{xmin},{ymin},{xmax},{ymax}",
                "n_gaussians": len(gauss),
                "member_gaussian_ids": ",".join(gauss),
                "member_local_group_ids": ",".join([gid for gid in [gid1, gid2] if gid]),
                "association_type": cand.get("parent_candidate_type", "parent_candidate"),
                "association_quality": cand.get("parent_candidate_quality", ""),
                "parent_candidate_quality": cand.get("parent_candidate_quality", ""),
                "parent_score": cand.get("parent_score_final", cand.get("lobe_pair_score", np.nan)),
                "is_parent_candidate": True,
                "is_local_group_only": False,
                "LAS_arcsec": cand.get("parent_LAS_arcsec", np.nan),
                "LAS_beam": cand.get("parent_LAS_beam", np.nan),
                "area_3sigma_beam": cand.get("parent_union_area_beam", np.nan),
                "major_axis_beam": np.nan,
                "minor_axis_beam": np.nan,
                "axis_ratio": np.nan,
                "peak_snr": np.nan,
                "total_flux": np.nansum([pd.to_numeric(g1.get("total_flux_gaussian", np.nan), errors="coerce"), pd.to_numeric(g2.get("total_flux_gaussian", np.nan), errors="coerce")]),
                "flux_ratio": cand.get("flux_ratio", np.nan),
                "size_ratio": cand.get("size_ratio", np.nan),
                "box_gap_beam_robust": cand.get("box_gap_beam_robust", np.nan),
                "center_distance_beam": cand.get("center_distance_beam", np.nan),
                "axis_alignment_score": cand.get("axis_alignment_score", np.nan),
                "facing_score": cand.get("facing_score", np.nan),
                "symmetry_score": cand.get("symmetry_score", np.nan),
                "host_catalog": cand.get("best_host_catalog", ""),
                "host_ra": cand.get("best_host_ra", np.nan),
                "host_dec": cand.get("best_host_dec", np.nan),
                "host_sep_arcsec": cand.get("best_host_sep_midpoint_arcsec", np.nan),
                "host_score": cand.get("best_host_score", np.nan),
                "host_quality": cand.get("host_quality", ""),
                "W1": cand.get("best_host_W1", np.nan),
                "W2": cand.get("best_host_W2", np.nan),
                "W1_minus_W2": cand.get("best_host_W1_W2", np.nan),
                "midpoint_host_found": str(cand.get("host_status", "")) == "host_found",
                "lobe1_peak_host_found": bool(cand.get("lobe1_peak_host_found", False)),
                "lobe2_peak_host_found": bool(cand.get("lobe2_peak_host_found", False)),
                "lobe_peak_host_contradiction": bool(contradiction),
                "pybdsf_status": "done",
                "association_status": "done",
                "host_query_status": _host_query_status(edge),
                "host_cache_key": str(edge.get("query_id", "")),
                "processing_version": PROCESSING_VERSION,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)


def _field_partials_dir(dirs: Any) -> Path:
    path = dirs.association_catalogs / "partials"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_field_outputs(dirs: Any, field_id: str, outputs: dict[str, pd.DataFrame]) -> None:
    partials = _field_partials_dir(dirs)
    for stem, frame in outputs.items():
        parquet_path = partials / f"{field_id}_{stem}.parquet"
        csv_path = partials / f"{field_id}_{stem}.csv"
        write_dataframe(frame, parquet_path)
        frame.to_csv(csv_path, index=False)
        if stem in {"parent_candidates", "parent_edges_debug", "host_candidates"}:
            _link_large_scale_parent_partial(dirs, parquet_path)
            _link_large_scale_parent_partial(dirs, csv_path)


def _read_table(path: Path) -> pd.DataFrame:
    csv_path = path.with_suffix(".csv")
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)


def _link_large_scale_parent_partial(dirs: Any, source: Path) -> None:
    if not source.exists():
        return
    link_dir = dirs.association_catalogs / "large_scale_parents" / "partials"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / source.name
    if link.is_symlink():
        try:
            if link.resolve() == source.resolve():
                return
        except FileNotFoundError:
            pass
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(source)


def _save_all_parent_zooms(
    field_id: str,
    cutout: Cutout,
    segmentation: Any,
    components: pd.DataFrame,
    groups: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    dirs: Any,
    config: dict[str, Any],
) -> int:
    if parent_candidates is None or parent_candidates.empty:
        return 0
    sort_col = "parent_score_final" if "parent_score_final" in parent_candidates else None
    ordered = parent_candidates.sort_values(sort_col, ascending=False) if sort_col else parent_candidates
    output_dir = dirs.association_figures_full / "parent_zoom" / field_id
    written_count = 0
    for idx, (_, row) in enumerate(ordered.iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_parent_link_parent_zoom(
            cutout,
            segmentation,
            components,
            groups,
            row,
            host_candidates,
            output_dir / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            written_count += 1
    return written_count


def _process_field(
    row: pd.Series,
    args: argparse.Namespace,
    config: dict[str, Any],
    input_format: str,
    figure_budget: dict[str, int],
) -> dict[str, Any]:
    # 一个 field 的生产路径：读取/生成 PyBDSF Gaussian -> 局部关联 -> production parent-linking parent-link -> 写字段级 partial。
    dirs = ensure_output_dirs(args.output_root)
    field_id = str(row["file_id"])
    h5_path = Path(str(row.get("h5_path", "")))
    catalog_path = Path(str(row.get("pybdsf_catalog_path", "")))
    ok, _, reason = pybdsf_catalog_sane(catalog_path)
    if not ok:
        raise RuntimeError(f"PyBDSF catalog not ready for {field_id}: {reason}")
    image_path = str(row.get("scratch_fits_path") or row.get("fits_path") or "")
    if image_path and not Path(image_path).exists():
        image_path = str(row.get("fits_path") or "")
    field_work = dirs.checkpoints / "fields" / field_id / "local"
    local_dirs = ensure_local_output_tree(field_work)
    logger = setup_logging(args.debug, dirs.association_logs / f"{field_id}.log")
    if input_format == "h5":
        if not h5_path.exists():
            raise RuntimeError(f"H5 cutouts missing for field {field_id}: {h5_path}")
        reader = H5CutoutReader(h5_path, config_h5=config.get("h5", {}))
    else:
        reader = FitsTileCutoutReader(image_path)
    gaussians, _ = normalized_gaussian_dataframe(read_gaussian_catalog(catalog_path))
    indices = reader.iter_indices(limit=args.limit)
    local_status_records: list[dict[str, Any]] = []
    for index in indices:
        cutout = reader.read(index)
        try:
            process_cutout(cutout, gaussians, config, local_dirs, make_figures=False, association_mode=True)
            local_status_records.append({"cutout_id": cutout.cutout_id, "status": "done", "failure_reason": ""})
        except Exception as exc:
            logger.error("local association failed %s %s: %s", field_id, cutout.cutout_id, exc)
            local_status_records.append({"cutout_id": cutout.cutout_id, "status": "failed", "failure_reason": str(exc)})
    pd.DataFrame(local_status_records).to_csv(local_dirs["logs"] / "status.csv", index=False)
    combine_partials(field_work)
    local_groups = _read_table(field_work / "catalogs" / "radio_association_groups.parquet")
    local_components = _read_table(field_work / "catalogs" / "radio_association_components.parquet")

    cfg = parent_link_config(config)
    host_client = HostQueryClient(
        cache_dir=dirs.association_host_cache,
        offline_cache_only=not bool(args.query_wise_host),
        skip_query=not bool(args.query_wise_host),
        max_results=int(cfg["host_support"].get("max_host_results_per_query", 20)),
    )
    created_at = _now()
    all_candidates: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    all_hosts: list[pd.DataFrame] = []
    all_logs: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    all_needs: list[pd.DataFrame] = []
    all_morph: list[pd.DataFrame] = []
    max_host_queries_state = {"count": 0}
    for index in indices:
        cutout = reader.read(index)
        cutout_id = str(cutout.cutout_id)
        seg_path = field_work / "segmentation" / f"{cutout_id}_seg.npz"
        if not seg_path.exists():
            continue
        segmentation = load_segmentation(seg_path)
        groups = local_groups[local_groups["cutout_id"].astype(str) == cutout_id].copy()
        components = local_components[local_components["cutout_id"].astype(str) == cutout_id].copy()
        result = run_parent_links(
            cutout_id,
            groups,
            components,
            segmentation,
            config,
            host_client,
            max_host_queries_state=max_host_queries_state,
            max_host_queries=args.max_host_queries_per_field,
        )
        all_candidates.append(result.candidates)
        all_edges.append(result.edges_debug)
        all_hosts.append(result.host_candidates)
        all_logs.append(result.host_query_log)
        all_diag.append(result.diagnostics)
        all_needs.append(result.needs_visual_check)
        all_morph.append(result.source_morph_table)
        if getattr(args, "save_all_parent_zoom", False):
            _save_all_parent_zooms(
                field_id,
                cutout,
                segmentation,
                components,
                groups,
                result.candidates,
                result.host_candidates,
                dirs,
                config,
            )
        if figure_budget["remaining"] > 0:
            plot_parent_link_cutout_all(cutout, segmentation, components, groups, result.source_morph_table, result.candidates, result.host_candidates, dirs.association_figures, config)
            figure_budget["remaining"] -= 1
    host_client.save_cache()
    if hasattr(reader, "close"):
        reader.close()
    candidates = _concat(all_candidates, PARENT_CANDIDATE_COLUMNS)
    edges = _concat(all_edges, PARENT_EDGE_DEBUG_COLUMNS)
    hosts = _concat(all_hosts, PARENT_HOST_CANDIDATE_COLUMNS)
    logs = _concat(all_logs, HOST_QUERY_LOG_COLUMNS)
    diagnostics = _concat(all_diag, PARENT_DIAGNOSTIC_COLUMNS)
    needs = _concat(all_needs, ["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    morph = _concat(all_morph, SOURCE_MORPH_TABLE_COLUMNS)
    for frame in [local_groups, local_components, candidates, edges, hosts, logs, diagnostics, needs, morph]:
        if frame is not None and not frame.empty:
            frame.insert(0, "field_id", field_id)
            frame.insert(1, "image_path", image_path)
            frame.insert(2, "input_format_used", input_format)
    merged = pd.concat(
        [
            _local_group_rows(field_id, image_path, input_format, local_groups, created_at),
            _parent_rows(field_id, image_path, input_format, local_groups, candidates, edges, created_at),
        ],
        ignore_index=True,
    )
    for col in FINAL_MERGED_COLUMNS:
        if col not in merged:
            merged[col] = np.nan
    merged = merged[FINAL_MERGED_COLUMNS]
    _write_field_outputs(
        dirs,
        field_id,
        {
            "local_groups": local_groups,
            "local_components": local_components,
            "merged_components": merged,
            "parent_candidates": candidates,
            "parent_edges_debug": edges,
            "host_candidates": hosts,
            "host_query_log": logs,
            "diagnostics": diagnostics,
            "needs_visual_check": needs,
            "source_morph_table": morph,
        },
    )
    return {
        "file_id": field_id,
        "status": "done",
        "n_cutouts": len(indices),
        "n_local_groups": len(local_groups),
        "n_parent_candidates": len(candidates),
        "n_merged_rows": len(merged),
        "n_host_candidates": len(hosts),
        "message": "",
        "ended_at": _now(),
    }


def _merge_final_outputs(output_root: Path) -> dict[str, Any]:
    # 所有 field 完成后再统一合并 partial，避免单个字段失败时破坏已有结果。
    dirs = ensure_output_dirs(output_root)
    partials = _field_partials_dir(dirs)

    def read_partials(stem: str) -> pd.DataFrame:
        frames = []
        paths = {path.with_suffix("") for path in partials.glob(f"*_{stem}.parquet")}
        paths.update(path.with_suffix("") for path in partials.glob(f"*_{stem}.csv"))
        for stem_path in sorted(paths):
            path = stem_path.with_suffix(".parquet")
            try:
                frames.append(_read_table(path))
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    merged = read_partials("merged_components")
    local_groups = read_partials("local_groups")
    parent_edges = read_partials("parent_edges_debug")
    host_candidates = read_partials("host_candidates")
    if merged.empty:
        merged = pd.DataFrame(columns=FINAL_MERGED_COLUMNS)
    for col in FINAL_MERGED_COLUMNS:
        if col not in merged:
            merged[col] = np.nan
    merged = merged[FINAL_MERGED_COLUMNS]
    merged_path = dirs.association_catalogs / "lotss_dr3_association_merged_components_full.parquet"
    merged_csv = dirs.association_catalogs / "lotss_dr3_association_merged_components_full.csv.gz"
    local_path = dirs.association_catalogs / "lotss_dr3_association_local_groups_full.parquet"
    edge_path = dirs.association_catalogs / "lotss_dr3_association_parent_edges_debug_full.parquet"
    host_path = dirs.association_catalogs / "lotss_dr3_association_host_candidates_full.parquet"
    write_dataframe(merged, merged_path)
    write_csv_gz(merged, merged_csv)
    write_dataframe(local_groups, local_path)
    write_dataframe(parent_edges, edge_path)
    write_dataframe(host_candidates, host_path)
    return {
        "merged_rows": len(merged),
        "local_group_rows": len(local_groups),
        "parent_edge_rows": len(parent_edges),
        "host_candidate_rows": len(host_candidates),
        "merged_path": str(merged_path),
        "merged_csv": str(merged_csv),
        "local_path": str(local_path),
        "edge_path": str(edge_path),
        "host_path": str(host_path),
    }


def _write_report(args: argparse.Namespace, manifest: pd.DataFrame, benchmark: dict[str, Any], final: dict[str, Any], job_id: str = "") -> None:
    dirs = ensure_output_dirs(args.output_root)
    pybdsf_ready = int((manifest["needs_pybdsf"].astype(str) == "False").sum()) if not manifest.empty else 0
    pybdsf_needed = int((manifest["needs_pybdsf"].astype(str) == "True").sum()) if not manifest.empty else 0
    status_path = dirs.checkpoints / "association_field_status.csv"
    status = _read_status(status_path)
    host_status_counts: dict[str, int] = {}
    merged_path = Path(final.get("merged_path", dirs.association_catalogs / "lotss_dr3_association_merged_components_full.parquet"))
    if merged_path.exists():
        try:
            merged = pd.read_parquet(merged_path, columns=["host_query_status"])
            host_status_counts = {str(k): int(v) for k, v in merged["host_query_status"].value_counts(dropna=False).to_dict().items()}
        except Exception:
            host_status_counts = {}
    lines = [
        "# LoTSS DR3 Full production parent-linking Status",
        "",
        f"- data_root: {args.data_root}",
        f"- original_data_root: {args.original_data_root}",
        f"- fits_count: {benchmark.get('fits_count', len(manifest))}",
        f"- h5_count: {benchmark.get('h5_count', 0)}",
        f"- selected_association_input_format: {benchmark.get('selected', '')}",
        f"- selected_reason: {benchmark.get('reason', '')}",
        f"- pybdsf_frequency_hz: {float(args.pybdsf_frequency_hz)}",
        "- pybdsf_frequency_source: explicit_argument",
        f"- pybdsf_existing_or_successful: {pybdsf_ready}",
        f"- pybdsf_needs_or_failed: {pybdsf_needed}",
        f"- association_fields_done: {int((status.get('status', pd.Series(dtype=str)) == 'done').sum()) if not status.empty else 0}",
        f"- association_fields_failed: {int((status.get('status', pd.Series(dtype=str)) == 'failed').sum()) if not status.empty else 0}",
        f"- gaussian_catalog_parquet: {dirs.pybdsf_processed / 'lotss_dr3_pybdsf_gaussians_full.parquet'}",
        f"- gaussian_catalog_csv_gz: {dirs.pybdsf_processed / 'lotss_dr3_pybdsf_gaussians_full.csv.gz'}",
        f"- merged_catalog_parquet: {final.get('merged_path', '')}",
        f"- merged_catalog_csv_gz: {final.get('merged_csv', '')}",
        f"- host_cache: {dirs.association_host_cache}",
        f"- wise_host_status_counts: {json.dumps(host_status_counts, sort_keys=True)}",
        f"- slurm_job_id: {job_id}",
        f"- single_slurm_job: {bool(job_id)}",
        f"- full_run_started: {not status.empty}",
        f"- final_merged_rows_current: {final.get('merged_rows', 0)}",
        "",
    ]
    (dirs.reports / "lotss_dr3_full_status.md").write_text("\n".join(lines), encoding="utf-8")


def _sanity_check(output_root: Path, limit_mode: bool) -> dict[str, Any]:
    dirs = ensure_output_dirs(output_root)
    result: dict[str, Any] = {}
    gauss_path = dirs.pybdsf_processed / "lotss_dr3_pybdsf_gaussians_full.parquet"
    merged_path = dirs.association_catalogs / "lotss_dr3_association_merged_components_full.parquet"
    result["gaussian_path_exists"] = gauss_path.exists()
    result["merged_path_exists"] = merged_path.exists()
    result["gaussian_rows"] = 0
    result["merged_rows"] = 0
    result["wise_columns_present"] = False
    result["processing_version_ok"] = False
    if gauss_path.exists():
        result["gaussian_rows"] = len(pd.read_parquet(gauss_path))
    if merged_path.exists():
        merged = pd.read_parquet(merged_path)
        result["merged_rows"] = len(merged)
        result["wise_columns_present"] = all(col in merged.columns for col in HOST_COLUMNS[:-2])
        result["processing_version_ok"] = bool(len(merged) == 0 or (merged.get("processing_version", pd.Series(dtype=str)).astype(str) == PROCESSING_VERSION).all())
    bad_dirs = []
    for token in ["obsolete_experimental_path_a", "obsolete_experimental_path_b", "obsolete_experimental_path_c"]:
        bad_dirs.extend([str(path) for path in dirs.root.rglob(f"*{token}*")])
    result["no_obsolete_experimental_outputs"] = len(bad_dirs) == 0
    result["bad_version_paths"] = bad_dirs[:10]
    if limit_mode and (result["gaussian_rows"] <= 0 or result["merged_rows"] <= 0 or not result["wise_columns_present"] or not result["processing_version_ok"]):
        raise SystemExit(f"Smoke sanity check failed: {result}")
    (dirs.reports / "sanity_check.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    logger = setup_logging(args.debug, dirs.logs / "run_lotss_dr3_full.log")
    copy_repro_files(args.output_root, [Path(__file__), PROJECT_ROOT / "scripts" / "lotss_dr3_common.py", args.config])
    if not (dirs.root / "README.md").exists():
        (dirs.root / "README.md").write_text(
            "# LoTSS DR3 Production Association Run\n\nProduction run using `lofar_det_vsex.parent_links.run_parent_links`.\n",
            encoding="utf-8",
        )
    logger.info("Building manifest")
    manifest_path = dirs.manifests / "lotss_dr3_fits_manifest.csv"
    if args.manifest is not None:
        manifest = read_manifest(args.manifest)
        write_manifest(manifest, manifest_path)
    else:
        manifest = _build_manifest(args)
    if args.skip_data_benchmark:
        benchmark = {
            "fits_count": int(len(manifest)),
            "h5_count": int((manifest.get("h5_path", pd.Series(dtype=str)).astype(str) != "").sum()),
            "selected": str(args.input_format),
            "reason": f"Skipped benchmark; using requested input_format={args.input_format}.",
            "h5_complete": bool((manifest.get("h5_path", pd.Series(dtype=str)).astype(str) != "").all()) if not manifest.empty else False,
            "h5_stable": np.nan,
            "fits_stable": np.nan,
        }
        (dirs.manifests / "data_format_benchmark.md").write_text(
            "\n".join(
                [
                    "# LoTSS DR3 Data Format Benchmark",
                    "",
                    f"- fits_count: {benchmark['fits_count']}",
                    f"- h5_count: {benchmark['h5_count']}",
                    "- benchmark_skipped: True",
                    f"- selected_association_input_format: {benchmark['selected']}",
                    f"- reason: {benchmark['reason']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    else:
        benchmark = _benchmark_data_format(args, manifest)
    if args.run_missing_pybdsf:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_missing_pybdsf_lotss_dr3.py"),
            "--output-root",
            str(args.output_root),
            "--manifest",
            str(manifest_path),
            "--num-workers",
            str(max(1, min(args.num_workers, 4))),
            "--max-retries",
            "1",
            "--pybdsf-frequency-hz",
            str(float(args.pybdsf_frequency_hz)),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        logger.info("Running PyBDSF missing stage")
        _run_stage(cmd)
        manifest = read_manifest(manifest_path)
    if args.use_existing_pybdsf or args.run_missing_pybdsf:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "merge_pybdsf_gaussian_catalogs.py"),
            "--output-root",
            str(args.output_root),
            "--manifest",
            str(manifest_path),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        logger.info("Merging PyBDSF Gaussian catalogs")
        _run_stage(cmd)

    config = load_yaml(args.config)
    status_path = dirs.checkpoints / "association_field_status.csv"
    status = _read_status(status_path)
    done = set(status.loc[status["status"].astype(str) == "done", "file_id"].astype(str)) if not status.empty and args.resume else set()
    ready = manifest[manifest["needs_pybdsf"].astype(str) == "False"].copy()
    if benchmark["selected"] == "h5":
        ready = ready[ready["h5_path"].astype(str) != ""].copy()
    if args.limit is not None:
        ready = ready.head(args.limit)
    records = status.to_dict(orient="records") if args.resume and not status.empty else []
    figure_budget = {"remaining": int(args.debug_sample_figures)}
    logger.info("Starting production parent-linking association fields=%d", len(ready))
    for _, row in ready.iterrows():
        field_id = str(row["file_id"])
        if field_id in done:
            continue
        started = _now()
        try:
            rec = _process_field(row, args, config, benchmark["selected"], figure_budget)
            rec["started_at"] = started
        except Exception as exc:
            message = traceback.format_exc() if args.debug else str(exc)
            logger.error("Field failed %s: %s", field_id, message)
            rec = {
                "file_id": field_id,
                "status": "failed",
                "n_cutouts": 0,
                "n_local_groups": 0,
                "n_parent_candidates": 0,
                "n_merged_rows": 0,
                "n_host_candidates": 0,
                "message": message[:4000],
                "started_at": started,
                "ended_at": _now(),
            }
        records = [r for r in records if str(r.get("file_id")) != field_id]
        records.append(rec)
        _write_field_status(status_path, records)

    final = _merge_final_outputs(args.output_root)
    _write_report(args, read_manifest(manifest_path), benchmark, final, job_id="")
    sanity = _sanity_check(args.output_root, limit_mode=args.limit is not None)
    logger.info("Sanity: %s", sanity)
    logger.info("Done current run: %s", final)


if __name__ == "__main__":
    main()
