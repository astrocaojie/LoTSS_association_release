#!/usr/bin/env python
"""Reproduce full-method DR1 component-only bbox support rates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from lotss_association.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue
from lotss_association.validation.footprint import build_dr1_sky_footprint, filter_predictions_in_footprint, footprint_summary
from lotss_association.validation.metrics import compute_support_rates, support_table_to_latex


DEFAULT_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_full"))
DEFAULT_PARENT_CATALOG = Path(
    os.environ.get("LOTSS_ASSOC_PARENT_CATALOG", DEFAULT_OUTPUT_ROOT / "association" / "catalogs" / "final_science_catalogs" / "parent_host_catalog.parquet")
)
DEFAULT_LOCAL_CATALOG = Path(
    os.environ.get("LOTSS_ASSOC_LOCAL_CATALOG", DEFAULT_OUTPUT_ROOT / "association" / "catalogs" / "final_science_catalogs" / "local_group_catalog.parquet")
)
DEFAULT_COMPONENT_MATCH_SHARDS = Path(os.environ.get("LOTSS_ASSOC_COMPONENT_MATCH_SHARDS", "data/dr1/crossmatch_parent_external_support/matches/shards"))
DR1_COMPONENT_CATALOG_KEY = "lotss_dr1_component_catalogue"
DR1_COMPONENT_REFERENCE_GROUP = "LoTSS DR1 component/association"


EXPECTED_SANITY = {
    ("parent_high", 0.0): (153, 146, 0.954248366013072),
    ("parent_high", 0.05): (106, 103, 0.9716981132075472),
    ("parent_high", 0.1): (64, 61, 0.953125),
    ("local_multigaussian_extended", 0.0): (276468, 156471, 0.5659642345587916),
    ("local_multigaussian_extended", 0.05): (34673, 32679, 0.9424912756323364),
    ("local_multigaussian_extended", 0.1): (20241, 19243, 0.950694135665234),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr1-catalogue", default=str(DEFAULT_DR1_COMPONENT_CSV))
    parser.add_argument("--parent-catalog", default=str(DEFAULT_PARENT_CATALOG))
    parser.add_argument("--local-catalog", default=str(DEFAULT_LOCAL_CATALOG))
    parser.add_argument("--component-match-shards", default=str(DEFAULT_COMPONENT_MATCH_SHARDS))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/dr1_validation/full_method_sanity"))
    parser.add_argument("--grid-size-deg", type=float, default=0.25)
    parser.add_argument("--flux-cuts", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--max-local-rows", type=int, default=None, help="Smoke mode: limit local rows after filtering.")
    parser.add_argument("--max-parent-rows", type=int, default=None, help="Smoke mode: limit parent rows after filtering.")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _read_parquet(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    keep = [col for col in columns if col in available]
    missing = sorted(set(columns) - set(keep))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return pd.read_parquet(path, columns=keep)


def _prepare_parent_predictions(parent_path: str | Path, max_rows: int | None = None) -> tuple[pd.DataFrame, set[str]]:
    columns = [
        "global_parent_id",
        "parent_quality_class",
        "ra_radio_center",
        "dec_radio_center",
        "bbox_ra_min",
        "bbox_ra_max",
        "bbox_dec_min",
        "bbox_dec_max",
        "total_flux",
        "n_gaussians",
        "parent_score",
        "geometry_score",
        "host_support_flag",
        "lobe1_local_group_id",
        "lobe2_local_group_id",
    ]
    parent = _read_parquet(parent_path, columns)
    parent_high = parent.loc[parent["parent_quality_class"].astype(str) == "high"].copy()
    if max_rows is not None:
        parent_high = parent_high.head(int(max_rows)).copy()
    parent_high["source_id"] = parent_high["global_parent_id"].astype(str)
    parent_high["sample"] = "parent_high"
    parent_high["ra_centre"] = pd.to_numeric(parent_high["ra_radio_center"], errors="coerce")
    parent_high["dec_centre"] = pd.to_numeric(parent_high["dec_radio_center"], errors="coerce")
    parent_high["total_flux_jy"] = pd.to_numeric(parent_high["total_flux"], errors="coerce").fillna(0.0)
    parent_high["quality_class"] = parent_high["parent_quality_class"].astype(str)

    promoted = set()
    for col in ("lobe1_local_group_id", "lobe2_local_group_id"):
        promoted.update(parent[col].dropna().astype(str).tolist())
    promoted.discard("")
    keep = [
        "source_id",
        "sample",
        "ra_centre",
        "dec_centre",
        "bbox_ra_min",
        "bbox_ra_max",
        "bbox_dec_min",
        "bbox_dec_max",
        "total_flux_jy",
        "n_gaussians",
        "quality_class",
        "parent_score",
        "geometry_score",
        "host_support_flag",
    ]
    return parent_high.loc[:, keep].copy(), promoted


def _prepare_local_predictions(local_path: str | Path, promoted_local_ids: set[str], max_rows: int | None = None) -> pd.DataFrame:
    columns = [
        "field_name",
        "local_group_id",
        "global_local_group_id",
        "association_quality",
        "group_class",
        "ra_bbox_center",
        "dec_bbox_center",
        "bbox_ra_min",
        "bbox_ra_max",
        "bbox_dec_min",
        "bbox_dec_max",
        "total_flux",
        "n_gaussians",
    ]
    local = _read_parquet(local_path, columns)
    local["local_join_id"] = local["field_name"].astype(str) + "_" + local["local_group_id"].astype(str)
    mask = (pd.to_numeric(local["n_gaussians"], errors="coerce").fillna(0) >= 2) & (~local["local_join_id"].astype(str).isin(promoted_local_ids))
    work = local.loc[mask].copy()
    if max_rows is not None:
        work = work.head(int(max_rows)).copy()
    work["source_id"] = work["global_local_group_id"].astype(str)
    work["sample"] = "local_multigaussian_extended"
    work["ra_centre"] = pd.to_numeric(work["ra_bbox_center"], errors="coerce")
    work["dec_centre"] = pd.to_numeric(work["dec_bbox_center"], errors="coerce")
    work["total_flux_jy"] = pd.to_numeric(work["total_flux"], errors="coerce").fillna(0.0)
    work["quality_class"] = work["association_quality"].astype(str)
    keep = [
        "source_id",
        "sample",
        "ra_centre",
        "dec_centre",
        "bbox_ra_min",
        "bbox_ra_max",
        "bbox_dec_min",
        "bbox_dec_max",
        "total_flux_jy",
        "n_gaussians",
        "quality_class",
        "group_class",
    ]
    return work.loc[:, keep].copy()


def _load_component_support_ids(shards_dir: str | Path) -> set[str]:
    shards_dir = Path(shards_dir)
    if not shards_dir.exists():
        raise FileNotFoundError(f"Component-match shard directory not found: {shards_dir}")
    dataset = ds.dataset(str(shards_dir), format="parquet")
    filt = (ds.field("catalog_key") == DR1_COMPONENT_CATALOG_KEY) & (ds.field("reference_group") == DR1_COMPONENT_REFERENCE_GROUP)
    table = dataset.to_table(columns=["our_global_id"], filter=filt)
    ids = table.column("our_global_id").to_pandas().dropna().astype(str)
    return set(ids.tolist())


def _diff_expected(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        key = (str(row["sample"]), float(row["flux_cut_jy"]))
        exp = EXPECTED_SANITY.get(key)
        if exp is None:
            continue
        expected_n, expected_supported, expected_rate = exp
        rows.append(
            {
                "sample": row["sample"],
                "flux_cut_jy": float(row["flux_cut_jy"]),
                "expected_n_in_dr1_footprint": expected_n,
                "observed_n_in_dr1_footprint": int(row["n_in_dr1_footprint"]),
                "delta_n_in_dr1_footprint": int(row["n_in_dr1_footprint"]) - expected_n,
                "expected_supported": expected_supported,
                "observed_supported": int(row["n_supported_by_dr1_component"]),
                "delta_supported": int(row["n_supported_by_dr1_component"]) - expected_supported,
                "expected_support_rate": expected_rate,
                "observed_support_rate": float(row["support_rate"]),
                "delta_support_rate": float(row["support_rate"]) - expected_rate,
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path, summary: pd.DataFrame, diff: pd.DataFrame, metadata: dict[str, Any]) -> None:
    def csv_block(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"

    lines = [
        "# Full-Method DR1 Component Sanity Check",
        "",
        "Metric wording: DR1-supported agreement rate / support fraction under bbox containment.",
        "",
        "A predicted source is supported when at least one manually curated DR1 component position falls within the predicted radio bbox. The denominator is restricted to predicted bbox centres inside the DR1 component-catalogue footprint.",
        "",
        "Reference policy: only `lotss_dr1_component_catalogue.csv.gz` was used; no DR1 optical-ID catalogue was used.",
        "",
        "## Summary",
        "",
        csv_block(summary),
        "",
        "## Expected-Value Difference",
        "",
        csv_block(diff) if not diff.empty else "No expected comparison rows were available.",
        "",
        "## Metadata",
        "",
    ]
    for key in sorted(metadata):
        lines.append(f"- {key}: {metadata[key]}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    dr1, dr1_summary = load_dr1_component_catalogue(args.dr1_catalogue, report_path=output_dir / "dr1_catalog_columns_report.txt")
    footprint = build_dr1_sky_footprint(dr1, grid_size_deg=args.grid_size_deg, save_path=output_dir / "dr1_component_footprint_cells.csv")
    parent_pred, promoted = _prepare_parent_predictions(args.parent_catalog, args.max_parent_rows)
    local_pred = _prepare_local_predictions(args.local_catalog, promoted, args.max_local_rows)
    predictions = pd.concat([parent_pred, local_pred], ignore_index=True)
    predictions = filter_predictions_in_footprint(predictions, footprint)
    support_ids = _load_component_support_ids(args.component_match_shards)
    predictions["supported_by_dr1"] = predictions["source_id"].astype(str).isin(support_ids)
    predictions["matched_dr1_component_ids"] = ""
    predictions["n_matched_dr1_components"] = predictions["supported_by_dr1"].astype(int)

    support = compute_support_rates(predictions, args.flux_cuts, variant="full_method_component_only")
    diff = _diff_expected(support)

    predictions.to_csv(output_dir / "full_method_predictions_with_dr1_support.csv.gz", index=False, compression="gzip")
    support.to_csv(output_dir / "full_method_dr1_component_support_summary.csv", index=False)
    support_table_to_latex(support, str(output_dir / "full_method_dr1_component_support_summary.tex"))
    diff.to_csv(output_dir / "full_method_sanity_expected_difference.csv", index=False)

    metadata = {
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "runtime_seconds": round(perf_counter() - t0, 3),
        "python": sys.version.split()[0],
        "dr1_catalogue": str(args.dr1_catalogue),
        "parent_catalog": str(args.parent_catalog),
        "local_catalog": str(args.local_catalog),
        "component_match_shards": str(args.component_match_shards),
        "dr1_reference_policy": "LoTSS DR1 component catalogue only; no optical IDs.",
        "footprint": footprint_summary(footprint),
        "dr1_summary": dr1_summary,
        "n_prediction_rows": int(len(predictions)),
        "n_support_ids_from_component_shards": int(len(support_ids)),
        "support_definition": "DR1 component coordinate inside predicted radio bbox",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_readme(output_dir, support, diff, metadata)
    print(support.to_string(index=False))
    if not diff.empty:
        print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
