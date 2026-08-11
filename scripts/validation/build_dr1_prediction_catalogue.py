#!/usr/bin/env python
"""Build a standardized DR1-validation prediction catalogue from final science outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", required=True)
    parser.add_argument("--local-catalog", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parent_predictions(parent: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    work = parent.copy()
    quality = work.get("parent_quality_class", pd.Series("", index=work.index)).astype(str)
    work = work.loc[quality == "high"].copy()
    out = pd.DataFrame(
        {
            "source_id": work.get("global_parent_id", work.get("parent_id", work.index.astype(str))).astype(str),
            "sample": "parent_high",
            "ra_centre": pd.to_numeric(work.get("ra_radio_center", work.get("ra", pd.Series(dtype=float))), errors="coerce"),
            "dec_centre": pd.to_numeric(work.get("dec_radio_center", work.get("dec", pd.Series(dtype=float))), errors="coerce"),
            "bbox_ra_min": pd.to_numeric(work["bbox_ra_min"], errors="coerce"),
            "bbox_ra_max": pd.to_numeric(work["bbox_ra_max"], errors="coerce"),
            "bbox_dec_min": pd.to_numeric(work["bbox_dec_min"], errors="coerce"),
            "bbox_dec_max": pd.to_numeric(work["bbox_dec_max"], errors="coerce"),
            "total_flux_jy": pd.to_numeric(work.get("total_flux", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
            "n_gaussians": pd.to_numeric(work.get("n_gaussians", pd.Series(dtype=float)), errors="coerce"),
            "quality_class": quality.loc[work.index].astype(str),
        }
    )
    promoted: set[str] = set()
    for col in ("lobe1_local_group_id", "lobe2_local_group_id"):
        if col in parent:
            promoted.update(parent[col].dropna().astype(str).tolist())
    promoted.discard("")
    return out, promoted


def local_predictions(local: pd.DataFrame, promoted: set[str]) -> pd.DataFrame:
    work = local.copy()
    if {"field_name", "local_group_id"}.issubset(work.columns):
        join_id = work["field_name"].astype(str) + "_" + work["local_group_id"].astype(str)
    else:
        join_id = work.get("global_local_group_id", work.index.astype(str)).astype(str)
    mask = (pd.to_numeric(work.get("n_gaussians", pd.Series(0, index=work.index)), errors="coerce").fillna(0) >= 2) & (~join_id.astype(str).isin(promoted))
    work = work.loc[mask].copy()
    return pd.DataFrame(
        {
            "source_id": work.get("global_local_group_id", work.get("local_group_id", work.index.astype(str))).astype(str),
            "sample": "local_multigaussian_extended",
            "ra_centre": pd.to_numeric(work.get("ra_bbox_center", work.get("ra", pd.Series(dtype=float))), errors="coerce"),
            "dec_centre": pd.to_numeric(work.get("dec_bbox_center", work.get("dec", pd.Series(dtype=float))), errors="coerce"),
            "bbox_ra_min": pd.to_numeric(work["bbox_ra_min"], errors="coerce"),
            "bbox_ra_max": pd.to_numeric(work["bbox_ra_max"], errors="coerce"),
            "bbox_dec_min": pd.to_numeric(work["bbox_dec_min"], errors="coerce"),
            "bbox_dec_max": pd.to_numeric(work["bbox_dec_max"], errors="coerce"),
            "total_flux_jy": pd.to_numeric(work.get("total_flux", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
            "n_gaussians": pd.to_numeric(work.get("n_gaussians", pd.Series(dtype=float)), errors="coerce"),
            "quality_class": work.get("association_quality", work.get("group_class", pd.Series("", index=work.index))).astype(str),
        }
    )


def main() -> None:
    args = parse_args()
    parent = _read(args.parent_catalog)
    local = _read(args.local_catalog)
    parent_out, promoted = parent_predictions(parent)
    local_out = local_predictions(local, promoted)
    out = pd.concat([parent_out, local_out], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.suffix == ".gz" else None
    out.to_csv(output, index=False, compression=compression)
    print(f"Wrote {len(out)} DR1-validation prediction rows to {output}")


if __name__ == "__main__":
    main()
