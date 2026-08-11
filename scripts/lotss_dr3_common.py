#!/usr/bin/env python
"""Shared helpers for the LoTSS DR3 production association run."""

from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_full"))
DEFAULT_DATA_ROOT = Path(os.environ.get("LOTSS_ASSOC_DATA_ROOT", "data/LoTSS_scratch"))
DEFAULT_ORIGINAL_DATA_ROOT = Path(os.environ.get("LOTSS_ASSOC_ORIGINAL_DATA_ROOT", "data/LoTSS_DR3"))
DEFAULT_H5_ROOT = Path(os.environ.get("LOTSS_ASSOC_H5_ROOT", "data/lotss_cutout_2048"))
PROCESSING_VERSION = "lotss_dr3_association"

MANIFEST_COLUMNS = [
    "file_id",
    "fits_path",
    "scratch_fits_path",
    "h5_path",
    "field_name",
    "has_existing_pybdsf_catalog",
    "pybdsf_catalog_path",
    "needs_pybdsf",
    "status",
]

STANDARD_GAUSSIAN_COLUMNS = [
    "field_id",
    "image_path",
    "gaussian_id_global",
    "source_id_pybdsf",
    "island_id",
    "ra",
    "dec",
    "peak_flux",
    "total_flux",
    "local_rms",
    "maj",
    "min",
    "pa",
    "snr_peak",
    "bbox",
    "footprint",
    "beam_major",
    "beam_minor",
    "beam_pa",
    "flags",
    "pybdsf_status",
]

HOST_COLUMNS = [
    "host_catalog",
    "host_ra",
    "host_dec",
    "host_sep_arcsec",
    "host_score",
    "host_quality",
    "W1",
    "W2",
    "W1_minus_W2",
    "midpoint_host_found",
    "lobe1_peak_host_found",
    "lobe2_peak_host_found",
    "lobe_peak_host_contradiction",
    "host_query_status",
    "host_cache_key",
]

FINAL_MERGED_COLUMNS = [
    "merged_source_id",
    "field_id",
    "image_path",
    "input_format_used",
    "ra",
    "dec",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "union_bbox",
    "parent_union_box",
    "n_gaussians",
    "member_gaussian_ids",
    "member_local_group_ids",
    "association_type",
    "association_quality",
    "parent_candidate_quality",
    "parent_score",
    "is_parent_candidate",
    "is_local_group_only",
    "LAS_arcsec",
    "LAS_beam",
    "area_3sigma_beam",
    "major_axis_beam",
    "minor_axis_beam",
    "axis_ratio",
    "peak_snr",
    "total_flux",
    "flux_ratio",
    "size_ratio",
    "box_gap_beam_robust",
    "center_distance_beam",
    "axis_alignment_score",
    "facing_score",
    "symmetry_score",
    *HOST_COLUMNS[:-2],
    "pybdsf_status",
    "association_status",
    "host_query_status",
    "host_cache_key",
    "processing_version",
    "created_at",
]


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    scripts: Path
    slurm: Path
    logs: Path
    config: Path
    manifests: Path
    pybdsf_raw: Path
    pybdsf_processed: Path
    pybdsf_logs: Path
    association_catalogs: Path
    association_host_cache: Path
    association_figures: Path
    association_figures_full: Path
    association_logs: Path
    reports: Path
    checkpoints: Path


def ensure_output_dirs(output_root: str | Path) -> OutputDirs:
    root = Path(output_root)
    dirs = OutputDirs(
        root=root,
        scripts=root / "scripts",
        slurm=root / "slurm",
        logs=root / "logs",
        config=root / "config",
        manifests=root / "manifests",
        pybdsf_raw=root / "pybdsf" / "raw",
        pybdsf_processed=root / "pybdsf" / "processed",
        pybdsf_logs=root / "pybdsf" / "logs",
        association_catalogs=root / "association" / "catalogs",
        association_host_cache=root / "association" / "host_cache",
        association_figures=root / "association" / "figures_sample",
        association_figures_full=root / "association" / "figures_full",
        association_logs=root / "association" / "logs",
        reports=root / "reports",
        checkpoints=root / "checkpoints",
    )
    for path in dirs.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return dirs


def field_id_from_path(path: str | Path) -> str:
    stem = Path(path).name
    for suffix in [".fits.gz", ".FITS", ".fits", ".hdf5", ".h5"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def lotss_numeric_id(field_id: str) -> str | None:
    match = re.search(r"lotss_dr3_(\d+)_", field_id)
    return match.group(1) if match else None


def list_fits(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    patterns = ["*.fits", "*.fits.gz", "*.FITS"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(files, key=lambda p: (lotss_numeric_id(field_id_from_path(p)) is None, int(lotss_numeric_id(field_id_from_path(p)) or -1), str(p)))


def list_h5(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))
    return sorted(files, key=lambda p: (lotss_numeric_id(field_id_from_path(p)) is None, int(lotss_numeric_id(field_id_from_path(p)) or -1), str(p)))


def h5_map(h5_roots: Iterable[str | Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in h5_roots:
        for path in list_h5(root):
            out.setdefault(field_id_from_path(path), path)
    return out


def scratch_path_for(original_path: str | Path, original_root: str | Path, scratch_root: str | Path) -> Path:
    original_path = Path(original_path)
    try:
        rel = original_path.relative_to(Path(original_root))
    except ValueError:
        rel = Path(original_path.name)
    return Path(scratch_root) / rel


def default_pybdsf_catalog_path(raw_root: str | Path, field_id: str) -> Path:
    return Path(raw_root) / field_id / f"{field_id}.pybdsf.gaul.fits"


def pybdsf_catalog_sane(path: str | Path) -> tuple[bool, int, str]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False, 0, "missing_or_empty"
    try:
        from astropy.table import Table

        table = Table.read(path)
        n_rows = len(table)
        has_pos = any(col.lower() == "ra" for col in table.colnames) and any(col.lower() in {"dec", "decl"} for col in table.colnames)
        if n_rows <= 0:
            return False, 0, "zero_rows"
        if not has_pos:
            return False, n_rows, "missing_ra_dec"
        return True, n_rows, "ok"
    except Exception as exc:
        return False, 0, f"read_failed:{type(exc).__name__}:{exc}"


def find_existing_pybdsf_catalog(field_id: str, search_roots: Iterable[str | Path], raw_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if raw_root is not None:
        candidates.append(default_pybdsf_catalog_path(raw_root, field_id))
    tokens = {field_id}
    numeric = lotss_numeric_id(field_id)
    if numeric is not None:
        tokens.add(f"lotss_dr3_{numeric}_")
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ["*.gaul.fits", "*.gaul.FITS", "*gaul*.fits", "*gaussian*.fits", "*pybdsf*.fits", "*.gaul"]:
            for path in root.rglob(pattern):
                name = path.name
                if any(token in name for token in tokens):
                    candidates.append(path)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ok, _, _ = pybdsf_catalog_sane(path)
        if ok:
            return path
    return None


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for col in MANIFEST_COLUMNS:
        if col not in frame:
            frame[col] = ""
    return frame[MANIFEST_COLUMNS]


def write_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    out = frame.copy()
    for col in MANIFEST_COLUMNS:
        if col not in out:
            out[col] = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out[MANIFEST_COLUMNS].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_table_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    if suffixes.endswith(".fits") or suffixes.endswith(".fits.gz") or suffixes.endswith(".gaul"):
        from astropy.table import Table

        table = Table.read(path)
        return table.to_pandas()
    raise ValueError(f"Unsupported table path: {path}")


def write_csv_gz(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def copy_repro_files(output_root: str | Path, files: Iterable[str | Path]) -> None:
    dirs = ensure_output_dirs(output_root)
    for file_path in files:
        src = Path(file_path)
        if not src.exists():
            continue
        if src.suffix == ".sbatch":
            dst = dirs.slurm / src.name
        elif src.suffix in {".yaml", ".yml"}:
            dst = dirs.config / src.name
        else:
            dst = dirs.scripts / src.name
        try:
            if src.resolve() == dst.resolve():
                continue
        except FileNotFoundError:
            pass
        shutil.copy2(src, dst)


def bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
