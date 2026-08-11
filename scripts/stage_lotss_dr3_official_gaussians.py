#!/usr/bin/env python3
"""Stage LoTSS DR3 official Gaussian catalogues as per-field production parent-linking inputs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OFFICIAL_GAUSSIAN_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_official_gaussian_catalogs"))
DEFAULT_IMAGE_ROOT = Path(os.environ.get("LOTSS_ASSOC_ORIGINAL_DATA_ROOT", "data/LoTSS_DR3"))
DEFAULT_DATA_ROOT = Path(os.environ.get("LOTSS_ASSOC_DATA_ROOT", "data/LoTSS_scratch"))
DEFAULT_H5_ROOT = Path(os.environ.get("LOTSS_ASSOC_H5_ROOT", "data/lotss_cutout_2048"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-gaul",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "official_gaussians" / "raw" / "LoTSS_DR3_v1.0.gaul.fits",
        help="Official LoTSS DR3 global Gaussian FITS catalogue.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--h5-root", type=Path, default=DEFAULT_H5_ROOT)
    parser.add_argument("--max-fields", type=int, default=None, help="Optional smoke limit.")
    parser.add_argument("--field-id", action="append", default=[], help="Stage only this field id; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-write-manifest", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def field_id_from_image(path: Path) -> str:
    name = path.name
    for suffix in [".fits.gz", ".FITS", ".fits"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def healpix_from_field_id(field_id: str) -> int | None:
    match = re.search(r"lotss_dr3_(\d+)_", field_id)
    if not match:
        return None
    return int(match.group(1))


def image_fields(image_root: Path) -> list[dict[str, object]]:
    fields: list[dict[str, object]] = []
    for path in sorted(image_root.glob("lotss_dr3_*_stokesI_fullres.fits")):
        field_id = field_id_from_image(path)
        healpix = healpix_from_field_id(field_id)
        if healpix is None:
            continue
        fields.append({"field_id": field_id, "healpix": healpix, "fits_path": path})
    return fields


def h5_paths(h5_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not h5_root.exists():
        return out
    for path in list(h5_root.rglob("*.h5")) + list(h5_root.rglob("*.hdf5")):
        out.setdefault(field_id_from_image(path), path)
    return out


def scratch_path_for(fits_path: Path, image_root: Path, data_root: Path) -> Path:
    try:
        rel = fits_path.relative_to(image_root)
    except ValueError:
        rel = Path(fits_path.name)
    return data_root / rel


def find_healpix_column(header: fits.Header) -> str:
    names = [header.get(f"TTYPE{i}") for i in range(1, int(header.get("TFIELDS", 0)) + 1)]
    for wanted in ["HEALPIX", "Healpix", "healpix", "Mosaic_ID", "MOSAIC_ID", "mosaic_id"]:
        if wanted in names:
            return wanted
    raise RuntimeError(f"No HEALPIX/Mosaic_ID-like column found. Columns: {names}")


def normalize_healpix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.int64, copy=False)
    if arr.dtype.kind in {"S", "a"}:
        text = np.char.decode(arr, "utf-8", errors="ignore")
    else:
        text = arr.astype(str)
    out = np.empty(len(text), dtype=np.int64)
    for idx, value in enumerate(text):
        match = re.search(r"(\d+)", str(value))
        out[idx] = int(match.group(1)) if match else -1
    return out


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["field_id", "healpix", "n_rows", "status", "catalog_path", "source_catalog", "created_at"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_manifest(
    path: Path,
    fields: list[dict[str, object]],
    rows: list[dict[str, object]],
    image_root: Path,
    data_root: Path,
    h5_root: Path,
) -> None:
    status_by_field = {str(row["field_id"]): row for row in rows}
    h5_by_field = h5_paths(h5_root)
    manifest_rows = []
    for field in fields:
        field_id = str(field["field_id"])
        fits_path = Path(field["fits_path"])
        status_row = status_by_field.get(field_id, {})
        catalog_path = Path(str(status_row.get("catalog_path", "")))
        ready = catalog_path.exists() and str(status_row.get("status", "")) in {"written", "skipped_existing"}
        manifest_rows.append(
            {
                "file_id": field_id,
                "fits_path": str(fits_path),
                "scratch_fits_path": str(scratch_path_for(fits_path, image_root, data_root)),
                "h5_path": str(h5_by_field.get(field_id, "")),
                "field_name": field_id,
                "has_existing_pybdsf_catalog": str(bool(ready)),
                "pybdsf_catalog_path": str(catalog_path),
                "needs_pybdsf": str(not ready),
                "status": "ready" if ready else "needs_pybdsf",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> int:
    args = parse_args()
    if not args.global_gaul.exists():
        raise SystemExit(f"Missing global Gaussian catalogue: {args.global_gaul}")
    if not args.image_root.exists():
        raise SystemExit(f"Missing image root: {args.image_root}")

    fields = image_fields(args.image_root)
    if args.field_id:
        wanted = set(args.field_id)
        fields = [field for field in fields if str(field["field_id"]) in wanted]
    if args.max_fields is not None:
        fields = fields[: max(0, int(args.max_fields))]
    if not fields:
        raise SystemExit("No matching LoTSS DR3 fields found")

    target_root = args.output_root / "pybdsf" / "raw"
    summary_path = args.output_root / "manifests" / "official_gaussian_field_catalogs.csv"
    target_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with fits.open(args.global_gaul, memmap=True) as hdul:
        hdu = hdul[1]
        healpix_col = find_healpix_column(hdu.header)
        data = hdu.data
        healpix_values = normalize_healpix(data[healpix_col])
        order = np.argsort(healpix_values, kind="stable")
        sorted_healpix = healpix_values[order]
        unique_healpix, starts, counts = np.unique(sorted_healpix, return_index=True, return_counts=True)
        index = {
            int(hp): (int(start), int(count))
            for hp, start, count in zip(unique_healpix, starts, counts)
            if int(hp) >= 0
        }

        for field in fields:
            field_id = str(field["field_id"])
            healpix = int(field["healpix"])
            out_path = target_root / field_id / f"{field_id}.pybdsf.gaul.fits"
            row = {
                "field_id": field_id,
                "healpix": healpix,
                "n_rows": 0,
                "status": "missing_healpix",
                "catalog_path": str(out_path),
                "source_catalog": str(args.global_gaul),
                "created_at": utc_now(),
            }
            if out_path.exists() and not args.overwrite:
                row["status"] = "skipped_existing"
                rows.append(row)
                continue
            if healpix not in index:
                rows.append(row)
                continue
            start, count = index[healpix]
            indices = np.sort(order[start : start + count])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Table(data[indices]).write(out_path, format="fits", overwrite=True)
            row["n_rows"] = int(count)
            row["status"] = "written"
            rows.append(row)
            print(f"{field_id}: wrote {count} rows -> {out_path}", flush=True)

    write_summary(summary_path, rows)
    if not args.no_write_manifest:
        write_manifest(
            args.output_root / "manifests" / "lotss_dr3_fits_manifest.csv",
            fields,
            rows,
            args.image_root,
            args.data_root,
            args.h5_root,
        )
    n_written = sum(1 for row in rows if row["status"] == "written")
    n_existing = sum(1 for row in rows if row["status"] == "skipped_existing")
    n_missing = sum(1 for row in rows if row["status"] == "missing_healpix")
    print(f"summary={summary_path}")
    print(f"written={n_written} skipped_existing={n_existing} missing_healpix={n_missing}")
    return 0 if n_missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
