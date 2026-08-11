#!/usr/bin/env python
"""Merge per-field PyBDSF Gaussian catalogs into full LoTSS DR3 tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lofar_det_vsex.catalog import normalized_gaussian_dataframe, read_gaussian_catalog
from lotss_dr3_common import (
    DEFAULT_OUTPUT_ROOT,
    STANDARD_GAUSSIAN_COLUMNS,
    bool_text,
    default_pybdsf_catalog_path,
    ensure_output_dirs,
    pybdsf_catalog_sane,
    read_manifest,
    write_manifest,
    write_csv_gz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _first(frame: pd.DataFrame, names: list[str], default: Any = np.nan) -> Any:
    lower = {str(col).lower(): col for col in frame.columns}
    for name in names:
        col = name if name in frame.columns else lower.get(name.lower())
        if col is not None:
            return frame[col]
    return default


def _beam_from_header(path: str) -> tuple[float, float, float]:
    try:
        from astropy.io import fits

        header = fits.getheader(path, 0)
        major = float(header.get("BMAJ", np.nan)) * 3600.0
        minor = float(header.get("BMIN", np.nan)) * 3600.0
        pa = float(header.get("BPA", np.nan))
        return major, minor, pa
    except Exception:
        return np.nan, np.nan, np.nan


def normalize_one(row: pd.Series) -> pd.DataFrame:
    field_id = str(row["file_id"])
    catalog_path = str(row.get("pybdsf_catalog_path", ""))
    ok, _, _ = pybdsf_catalog_sane(catalog_path)
    if not ok:
        return pd.DataFrame(columns=STANDARD_GAUSSIAN_COLUMNS)
    table = read_gaussian_catalog(catalog_path)
    frame, _ = normalized_gaussian_dataframe(table)
    image_path = str(row.get("scratch_fits_path", "") or row.get("fits_path", ""))
    if image_path and not Path(image_path).exists():
        image_path = str(row.get("fits_path", ""))
    beam_major, beam_minor, beam_pa = _beam_from_header(image_path)
    out = frame.copy()
    out.insert(0, "field_id", field_id)
    out.insert(1, "image_path", image_path)
    out["source_id_pybdsf"] = out.get("_source_id")
    out["island_id"] = out.get("_island_id")
    out["ra"] = out.get("_ra")
    out["dec"] = out.get("_dec")
    out["peak_flux"] = out.get("_peak_flux")
    out["total_flux"] = out.get("_total_flux")
    local_rms = _first(out, ["Isl_rms", "isl_rms", "Local_rms", "local_rms", "RMS", "rms"], np.nan)
    out["local_rms"] = pd.to_numeric(local_rms, errors="coerce") if hasattr(local_rms, "__len__") else np.nan
    out["maj"] = out.get("_maj")
    out["min"] = out.get("_min")
    out["pa"] = out.get("_pa")
    snr = _first(out, ["S_Code_SNR", "SNR", "snr_peak"], np.nan)
    if hasattr(snr, "__len__"):
        out["snr_peak"] = pd.to_numeric(snr, errors="coerce")
    else:
        out["snr_peak"] = pd.to_numeric(out["peak_flux"], errors="coerce") / pd.to_numeric(out["local_rms"], errors="coerce").replace(0, np.nan)
    out["bbox"] = ""
    out["footprint"] = ""
    out["beam_major"] = beam_major
    out["beam_minor"] = beam_minor
    out["beam_pa"] = beam_pa
    out["flags"] = _first(out, ["Flags", "flags", "S_Code", "s_code"], "")
    out["pybdsf_status"] = "done"
    out["gaussian_id_global"] = field_id + ":" + out.get("_gaussian_id").astype(str)
    for col in STANDARD_GAUSSIAN_COLUMNS:
        if col not in out:
            out[col] = np.nan
    ordered = STANDARD_GAUSSIAN_COLUMNS + [col for col in out.columns if col not in STANDARD_GAUSSIAN_COLUMNS]
    return out[ordered]


def update_manifest_from_pybdsf_status(manifest: pd.DataFrame, dirs: object) -> pd.DataFrame:
    status_path = dirs.checkpoints / "pybdsf_status.csv"
    if not status_path.exists():
        return manifest
    status = pd.read_csv(status_path, dtype=str).fillna("")
    done = status[status.get("status", "").astype(str) == "done"].copy()
    if done.empty:
        return manifest
    done = done.drop_duplicates("file_id", keep="last").set_index("file_id")
    updated = manifest.copy()
    for idx, row in updated.iterrows():
        field_id = str(row["file_id"])
        if field_id not in done.index:
            continue
        catalog = str(done.loc[field_id].get("pybdsf_catalog_path", ""))
        if not catalog:
            catalog = str(default_pybdsf_catalog_path(dirs.pybdsf_raw, field_id))
        updated.loc[idx, "pybdsf_catalog_path"] = catalog
        updated.loc[idx, "has_existing_pybdsf_catalog"] = "True"
        updated.loc[idx, "needs_pybdsf"] = "False"
        updated.loc[idx, "status"] = "ready"
    return updated


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    manifest_path = args.manifest or dirs.manifests / "lotss_dr3_fits_manifest.csv"
    manifest = read_manifest(manifest_path)
    if manifest.empty:
        raise SystemExit(f"Manifest is empty or missing: {manifest_path}")
    manifest = update_manifest_from_pybdsf_status(manifest, dirs)
    write_manifest(manifest, manifest_path)
    ready = manifest[~manifest["needs_pybdsf"].map(bool_text)].copy()
    if args.limit is not None:
        ready = ready.head(args.limit)
    frames = []
    failed_rows = []
    for _, row in ready.iterrows():
        try:
            frame = normalize_one(row)
            if frame.empty:
                failed_rows.append({"file_id": row["file_id"], "pybdsf_catalog_path": row.get("pybdsf_catalog_path", ""), "reason": "empty_or_bad"})
            else:
                frames.append(frame)
        except Exception as exc:
            failed_rows.append({"file_id": row["file_id"], "pybdsf_catalog_path": row.get("pybdsf_catalog_path", ""), "reason": f"{type(exc).__name__}: {exc}"})
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STANDARD_GAUSSIAN_COLUMNS)
    parquet_path = dirs.pybdsf_processed / "lotss_dr3_pybdsf_gaussians_full.parquet"
    csv_path = dirs.pybdsf_processed / "lotss_dr3_pybdsf_gaussians_full.csv.gz"
    merged.to_parquet(parquet_path, index=False)
    write_csv_gz(merged, csv_path)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(dirs.reports / "pybdsf_merge_failed_files.csv", index=False)
    print(f"rows={len(merged)}")
    print(f"parquet={parquet_path}")
    print(f"csv_gz={csv_path}")
    print(f"failed={len(failed_rows)}")


if __name__ == "__main__":
    main()
