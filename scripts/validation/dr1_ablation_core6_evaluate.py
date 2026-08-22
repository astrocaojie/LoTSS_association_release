#!/usr/bin/env python
"""Evaluate core DR1 ablation outputs when full-run catalogues are present."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lotss_association.utils import load_yaml
from lotss_association.validation.bbox_support import match_predictions_to_dr1_components
from lotss_association.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue
from lotss_association.validation.footprint import build_dr1_sky_footprint, filter_predictions_in_footprint
from lotss_association.validation.metrics import compute_support_rates


CORE_VARIANTS = [
    "full_method",
    "no_ridge_continuity",
    "no_weak_edge_anti_chaining",
    "no_artifact_penalties",
    "no_host_support",
    "no_lobe_peak_host_contradiction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dr1_validation/core_ablation_variants.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--evaluate-missing-as-error", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _variant_names(config: dict[str, Any]) -> list[str]:
    names = [str(row.get("name")) for row in config.get("variants", []) if row.get("name")]
    return names or CORE_VARIANTS


def _catalogue_path(config: dict[str, Any], variant: str) -> Path | None:
    entry = (config.get("catalogues") or {}).get(variant, {}) or {}
    value = entry.get("predictions")
    if value:
        return _resolve(value)
    return None


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"source_id", "sample", "bbox_ra_min", "bbox_ra_max", "bbox_dec_min", "bbox_dec_max", "total_flux_jy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns required for DR1 evaluation: {missing}")
    return frame


def _add_bbox_area(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    ra_min = pd.to_numeric(out["bbox_ra_min"], errors="coerce") % 360.0
    ra_max = pd.to_numeric(out["bbox_ra_max"], errors="coerce") % 360.0
    dec_min = pd.to_numeric(out["bbox_dec_min"], errors="coerce")
    dec_max = pd.to_numeric(out["bbox_dec_max"], errors="coerce")
    width = (ra_max - ra_min) % 360.0
    height = (dec_max - dec_min).abs()
    dec_centre = 0.5 * (dec_min + dec_max)
    cos_dec = np.cos(np.deg2rad(dec_centre)).clip(0.0, 1.0)
    out["bbox_width_deg"] = width
    out["bbox_height_deg"] = height
    out["bbox_area_deg2_approx"] = width * height * cos_dec
    out["bbox_area_arcsec2_approx"] = out["bbox_area_deg2_approx"] * 3600.0 * 3600.0
    return out


def _shifted_random_rates(predictions: pd.DataFrame, dr1: pd.DataFrame, shifts: list[list[float]], flux_cuts: list[float], grid_size: float, variant: str) -> pd.DataFrame:
    rows = []
    random_input = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    for i, pair in enumerate(shifts):
        ra_shift, dec_shift = float(pair[0]), float(pair[1])
        shifted = dr1.copy()
        shifted["ra"] = (pd.to_numeric(shifted["ra"], errors="coerce") + ra_shift) % 360.0
        shifted["dec"] = pd.to_numeric(shifted["dec"], errors="coerce") + dec_shift
        shifted["valid_position"] = shifted["dec"].between(-90.0, 90.0, inclusive="both")
        matched = match_predictions_to_dr1_components(random_input, shifted, grid_size_deg=grid_size)
        matched["supported_by_dr1"] = matched["supported_by_dr1"].astype(bool)
        rates = compute_support_rates(matched, flux_cuts, variant=variant)
        rates["random_shift_index"] = i
        rates["ra_shift_deg"] = ra_shift
        rates["dec_shift_deg"] = dec_shift
        rows.append(rates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _corrected(observed: pd.DataFrame, random_rates: pd.DataFrame) -> pd.DataFrame:
    if random_rates.empty:
        out = observed.copy()
        out["random_support_rate_mean"] = np.nan
        out["random_support_rate_std"] = np.nan
        out["chance_corrected_support_rate"] = np.nan
        return out
    random_summary = (
        random_rates.groupby(["variant", "sample", "flux_cut_jy"], as_index=False)
        .agg(random_support_rate_mean=("support_rate", "mean"), random_support_rate_std=("support_rate", "std"), n_random_shifts=("support_rate", "count"))
    )
    out = observed.rename(columns={"support_rate": "observed_support_rate"}).merge(random_summary, on=["variant", "sample", "flux_cut_jy"], how="left")
    r = pd.to_numeric(out["random_support_rate_mean"], errors="coerce")
    o = pd.to_numeric(out["observed_support_rate"], errors="coerce")
    out["chance_corrected_support_rate"] = ((o - r) / (1.0 - r)).clip(lower=0.0, upper=1.0)
    return out


def _area_diagnostics(predictions: pd.DataFrame, flux_cuts: list[float], variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    if work.empty or "bbox_area_arcsec2_approx" not in work:
        return pd.DataFrame(), pd.DataFrame()
    for sample, sample_df in work.groupby("sample"):
        for cut in flux_cuts:
            subset = sample_df.loc[pd.to_numeric(sample_df["total_flux_jy"], errors="coerce").fillna(0.0) >= float(cut)].copy()
            if subset.empty:
                continue
            area = pd.to_numeric(subset["bbox_area_arcsec2_approx"], errors="coerce")
            supported = subset["supported_by_dr1"].astype(bool)
            p50 = float(area.quantile(0.50))
            p90 = float(area.quantile(0.90))
            p99 = float(area.quantile(0.99))
            max_area = float(area.max())
            top10 = area >= p90
            low90 = area < p90
            n_supported = int(supported.sum())
            n_supported_top10 = int((supported & top10).sum())
            summary_rows.append(
                {
                    "variant": variant,
                    "sample": sample,
                    "flux_cut_jy": float(cut),
                    "n_in_dr1_footprint": int(len(subset)),
                    "n_supported": n_supported,
                    "median_bbox_area_arcsec2": p50,
                    "p90_bbox_area_arcsec2": p90,
                    "p99_bbox_area_arcsec2": p99,
                    "max_bbox_area_arcsec2": max_area,
                    "support_rate_all": float(supported.mean()),
                    "support_rate_lower_90pct_area": float(supported[low90].mean()) if int(low90.sum()) else np.nan,
                    "support_rate_top_10pct_area": float(supported[top10].mean()) if int(top10.sum()) else np.nan,
                    "fraction_supported_from_top_10pct_area": float(n_supported_top10 / n_supported) if n_supported else np.nan,
                    "large_bbox_driven_flag": bool(n_supported > 0 and n_supported_top10 / n_supported > 0.5),
                }
            )
            edges = [-np.inf, p50, p90, p99, np.inf]
            labels = ["<=p50", "p50-p90", "p90-p99", ">p99"]
            if len(set(float(edge) for edge in edges if np.isfinite(edge))) < 3:
                subset["_area_bin"] = pd.Series("all", index=subset.index, dtype="object")
            else:
                subset["_area_bin"] = pd.cut(area, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
            for label, frame in subset.groupby("_area_bin", observed=False):
                if frame.empty:
                    continue
                frame_area = pd.to_numeric(frame["bbox_area_arcsec2_approx"], errors="coerce")
                frame_supported = frame["supported_by_dr1"].astype(bool)
                bin_rows.append(
                    {
                        "variant": variant,
                        "sample": sample,
                        "flux_cut_jy": float(cut),
                        "bbox_area_bin": str(label),
                        "n_in_bin": int(len(frame)),
                        "n_supported": int(frame_supported.sum()),
                        "support_rate": float(frame_supported.mean()),
                        "bbox_area_min_arcsec2": float(frame_area.min()),
                        "bbox_area_median_arcsec2": float(frame_area.median()),
                        "bbox_area_max_arcsec2": float(frame_area.max()),
                        "supported_contribution_fraction": float(frame_supported.sum() / n_supported) if n_supported else np.nan,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def _diagnostics(predictions: pd.DataFrame, variant: str) -> dict[str, Any]:
    in_fp = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    n_gauss = pd.to_numeric(in_fp.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0)
    parent = in_fp[in_fp["sample"].astype(str) == "parent_high"]
    local = in_fp[in_fp["sample"].astype(str) == "local_multigaussian_extended"]
    return {
        "variant": variant,
        "parent_high_count": int(len(parent)),
        "local_multigaussian_extended_count": int(len(local)),
        "singleton_count": int((n_gauss <= 1).sum()),
        "multi_component_group_count": int((n_gauss >= 2).sum()),
        "max_group_size": int(n_gauss.max()) if len(n_gauss) else 0,
    }


def _variant_root_from_prediction(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        return path.parent.parent
    except Exception:
        return None


def _read_table_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _count_quality(frame: pd.DataFrame, column: str, token: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].astype(str).str.contains(token, case=False, na=False).sum())


def _layer1_diagnostics(variant: str, prediction_path: Path | None, predictions: pd.DataFrame) -> dict[str, Any]:
    root = _variant_root_from_prediction(prediction_path)
    local_path = root / "association/catalogs/final_science_catalogs/local_group_catalog.parquet" if root else Path()
    local = _read_table_if_exists(local_path) if root else pd.DataFrame()
    fallback = _diagnostics(predictions, variant) if not predictions.empty else {}
    if local.empty:
        return {
            "variant": variant,
            "local_group_catalog": str(local_path) if root else "",
            "catalogue_exists": False,
            "n_local_groups": np.nan,
            "singleton_count": fallback.get("singleton_count", np.nan),
            "multi_component_group_count": fallback.get("multi_component_group_count", np.nan),
            "local_multigaussian_extended_count": fallback.get("local_multigaussian_extended_count", np.nan),
            "max_group_size": fallback.get("max_group_size", np.nan),
            "good_or_basic_count": np.nan,
            "needs_visual_check_count": np.nan,
        }
    n_gauss = pd.to_numeric(local.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0)
    quality = local.get("association_quality", pd.Series("", index=local.index)).astype(str)
    return {
        "variant": variant,
        "local_group_catalog": str(local_path),
        "catalogue_exists": True,
        "n_local_groups": int(len(local)),
        "singleton_count": int((n_gauss <= 1).sum()),
        "multi_component_group_count": int((n_gauss >= 2).sum()),
        "local_multigaussian_extended_count": int((n_gauss >= 2).sum()),
        "max_group_size": int(n_gauss.max()) if len(n_gauss) else 0,
        "good_or_basic_count": int(quality.str.contains("good|basic", case=False, na=False).sum()),
        "needs_visual_check_count": int(local.get("needs_visual_check", pd.Series(False, index=local.index)).astype(bool).sum()) if "needs_visual_check" in local else 0,
    }


def _layer2_diagnostics(variant: str, prediction_path: Path | None, predictions: pd.DataFrame) -> dict[str, Any]:
    root = _variant_root_from_prediction(prediction_path)
    final_parent_path = root / "association/catalogs/final_science_catalogs/parent_host_catalog.parquet" if root else Path()
    edge_path = root / "association/catalogs/lotss_dr3_association_parent_edges_debug_full.parquet" if root else Path()
    parent = _read_table_if_exists(final_parent_path) if root else pd.DataFrame()
    edges = _read_table_if_exists(edge_path) if root else pd.DataFrame()
    fallback = _diagnostics(predictions, variant) if not predictions.empty else {}
    quality_col = "parent_quality_class"
    edge_quality_col = "parent_candidate_quality"
    n_gauss = pd.to_numeric(parent.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "variant": variant,
        "parent_host_catalog": str(final_parent_path) if root else "",
        "parent_catalogue_exists": bool(not parent.empty),
        "parent_edge_debug_catalogue": str(edge_path) if root else "",
        "edge_catalogue_exists": bool(not edges.empty),
        "n_parent_rows": int(len(parent)) if not parent.empty else np.nan,
        "parent_high_count": _count_quality(parent, quality_col, "high") if not parent.empty else fallback.get("parent_high_count", np.nan),
        "parent_medium_count": _count_quality(parent, quality_col, "medium") if not parent.empty else np.nan,
        "parent_suspicious_or_reject_count": _count_quality(parent, quality_col, "suspicious|reject") if not parent.empty else np.nan,
        "parent_needs_host_check_count": _count_quality(parent, quality_col, "needs_host_check") if not parent.empty else np.nan,
        "max_parent_group_size": int(n_gauss.max()) if len(n_gauss) else np.nan,
        "n_parent_edge_rows": int(len(edges)) if not edges.empty else np.nan,
        "edge_high_count": _count_quality(edges, edge_quality_col, "high") if not edges.empty else np.nan,
        "edge_medium_count": _count_quality(edges, edge_quality_col, "medium") if not edges.empty else np.nan,
        "edge_suspicious_or_reject_count": _count_quality(edges, edge_quality_col, "suspicious|reject") if not edges.empty else np.nan,
        "edge_needs_host_check_count": _count_quality(edges, edge_quality_col, "needs_host_check") if not edges.empty else np.nan,
    }


def _split_merge(full_predictions: pd.DataFrame, predictions: pd.DataFrame, variant: str) -> dict[str, Any]:
    if variant == "full_method" or full_predictions.empty or predictions.empty:
        return {"variant": variant, "number_of_full_sources_lost": 0, "number_of_new_sources": 0, "number_of_sources_with_changed_bbox": 0, "number_of_possible_overmerged_cases": 0, "number_of_possible_split_cases": 0}
    full = full_predictions.set_index("source_id", drop=False)
    other = predictions.set_index("source_id", drop=False)
    full_ids = set(full.index.astype(str))
    other_ids = set(other.index.astype(str))
    common = sorted(full_ids & other_ids)
    changed = 0
    cols = ["bbox_ra_min", "bbox_ra_max", "bbox_dec_min", "bbox_dec_max"]
    for sid in common:
        a = pd.to_numeric(full.loc[sid, cols], errors="coerce").to_numpy(float)
        b = pd.to_numeric(other.loc[sid, cols], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-8, equal_nan=True):
            changed += 1
    return {
        "variant": variant,
        "number_of_full_sources_lost": int(len(full_ids - other_ids)),
        "number_of_new_sources": int(len(other_ids - full_ids)),
        "number_of_sources_with_changed_bbox": int(changed),
        "number_of_possible_overmerged_cases": np.nan,
        "number_of_possible_split_cases": np.nan,
    }


def _delta_relative_to_full(summary: pd.DataFrame, split_merge: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metric_cols = [
        "observed_support_rate",
        "random_support_rate_mean",
        "chance_corrected_support_rate",
        "n_in_dr1_footprint",
        "n_supported_by_dr1_component",
        "parent_high_count",
        "local_multigaussian_extended_count",
        "singleton_count",
        "multi_component_group_count",
        "max_group_size",
    ]
    full = summary.loc[summary["variant"] == "full_method"].copy()
    if full.empty:
        return pd.DataFrame()
    full = full[["sample", "flux_cut_jy", *metric_cols]].rename(columns={col: f"full_{col}" for col in metric_cols})
    out = summary.merge(full, on=["sample", "flux_cut_jy"], how="left")
    for col in metric_cols:
        out[f"delta_{col}"] = pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(out[f"full_{col}"], errors="coerce")
    keep = [
        "variant",
        "sample",
        "flux_cut_jy",
        *[f"delta_{col}" for col in metric_cols],
        "number_of_full_sources_lost",
        "number_of_new_sources",
        "number_of_sources_with_changed_bbox",
        "number_of_possible_overmerged_cases",
        "number_of_possible_split_cases",
    ]
    for col in keep:
        if col not in out:
            out[col] = np.nan
    return out[keep].merge(split_merge, on="variant", how="left", suffixes=("", "_split_merge"))


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir or config.get("outputs", {}).get("root", PROJECT_ROOT / "outputs/dr1_validation/ablation_core6"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = config.get("reference", {}) or {}
    flux_cuts = [float(x) for x in ref.get("flux_cuts_jy", [0.0, 0.05, 0.1])]
    shifts = ref.get("random_shifts_ra_dec_deg", [[1.0, 0.0], [-1.0, 0.0]])
    grid_size = float(ref.get("footprint_grid_size_deg", 0.25))
    dr1_path = ref.get("dr1_catalogue", str(DEFAULT_DR1_COMPONENT_CSV))
    dr1, dr1_summary = load_dr1_component_catalogue(dr1_path)
    footprint = build_dr1_sky_footprint(dr1, grid_size_deg=grid_size)

    observed_rows = []
    random_rows = []
    diagnostic_rows = []
    split_rows = []
    area_summary_rows = []
    area_bin_rows = []
    layer1_rows = []
    layer2_rows = []
    missing_rows = []
    predictions_by_variant: dict[str, pd.DataFrame] = {}
    path_by_variant: dict[str, Path | None] = {}
    for variant in _variant_names(config):
        path = _catalogue_path(config, variant)
        path_by_variant[variant] = path
        if path is None or not path.exists():
            missing_rows.append({"variant": variant, "missing_prediction_catalogue": str(path) if path is not None else "", "status": "missing_input"})
            layer1_rows.append(_layer1_diagnostics(variant, path, pd.DataFrame()))
            layer2_rows.append(_layer2_diagnostics(variant, path, pd.DataFrame()))
            continue
        predictions = _read_predictions(path)
        predictions = _add_bbox_area(predictions)
        if "in_dr1_footprint" not in predictions:
            predictions = filter_predictions_in_footprint(predictions, footprint)
        if "supported_by_dr1" not in predictions or variant != "full_method":
            predictions = match_predictions_to_dr1_components(predictions, dr1, grid_size_deg=grid_size)
        predictions_by_variant[variant] = predictions
        observed_rows.append(compute_support_rates(predictions, flux_cuts, variant=variant))
        random_rows.append(_shifted_random_rates(predictions, dr1, shifts, flux_cuts, grid_size, variant))
        area_summary, area_bins = _area_diagnostics(predictions, flux_cuts, variant)
        if not area_summary.empty:
            area_summary_rows.append(area_summary)
        if not area_bins.empty:
            area_bin_rows.append(area_bins)
        diagnostic_rows.append(_diagnostics(predictions, variant))
        layer1_rows.append(_layer1_diagnostics(variant, path, predictions))
        layer2_rows.append(_layer2_diagnostics(variant, path, predictions))

    full_predictions = predictions_by_variant.get("full_method", pd.DataFrame())
    for variant, predictions in predictions_by_variant.items():
        split_rows.append(_split_merge(full_predictions, predictions, variant))
    for row in missing_rows:
        split_rows.append({"variant": row["variant"], "number_of_full_sources_lost": np.nan, "number_of_new_sources": np.nan, "number_of_sources_with_changed_bbox": np.nan, "number_of_possible_overmerged_cases": np.nan, "number_of_possible_split_cases": np.nan})

    observed = pd.concat(observed_rows, ignore_index=True) if observed_rows else pd.DataFrame()
    random_rates = pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame()
    corrected = _corrected(observed, random_rates) if not observed.empty else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    split_merge = pd.DataFrame(split_rows)
    area_diagnostics = pd.concat(area_summary_rows, ignore_index=True) if area_summary_rows else pd.DataFrame()
    area_bin_support = pd.concat(area_bin_rows, ignore_index=True) if area_bin_rows else pd.DataFrame()
    layer1 = pd.DataFrame(layer1_rows)
    layer2 = pd.DataFrame(layer2_rows)
    missing = pd.DataFrame(missing_rows)
    summary = corrected.merge(diagnostics, on="variant", how="left").merge(split_merge, on="variant", how="left") if not corrected.empty else pd.DataFrame()

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
        "parent_high_count",
        "local_multigaussian_extended_count",
        "singleton_count",
        "multi_component_group_count",
        "max_group_size",
        "number_of_full_sources_lost",
        "number_of_new_sources",
        "number_of_sources_with_changed_bbox",
        "number_of_possible_overmerged_cases",
        "number_of_possible_split_cases",
    ]
    if summary.empty:
        summary = pd.DataFrame(columns=support_cols)
    else:
        for col in support_cols:
            if col not in summary:
                summary[col] = np.nan
        summary = summary[support_cols]
    summary.to_csv(output_dir / "ablation_support_summary.csv", index=False)
    summary.to_latex(output_dir / "ablation_support_summary.tex", index=False, float_format="%.4f")
    corrected.to_csv(output_dir / "ablation_random_shift_corrected_support.csv", index=False)
    observed.to_csv(output_dir / "ablation_observed_support.csv", index=False)
    random_rates.to_csv(output_dir / "ablation_random_shift_support_by_shift.csv", index=False)
    area_diagnostics.to_csv(output_dir / "ablation_bbox_area_diagnostics.csv", index=False)
    area_bin_support.to_csv(output_dir / "ablation_bbox_area_bin_support.csv", index=False)
    diagnostics.to_csv(output_dir / "ablation_diagnostics_summary.csv", index=False)
    split_merge.to_csv(output_dir / "ablation_split_merge_vs_full.csv", index=False)
    _delta_relative_to_full(summary, split_merge).to_csv(output_dir / "ablation_delta_relative_to_full.csv", index=False)
    layer1.to_csv(output_dir / "layer1_diagnostics_summary.csv", index=False)
    layer2.to_csv(output_dir / "layer2_diagnostics_summary.csv", index=False)
    missing.to_csv(output_dir / "ablation_missing_inputs.csv", index=False)
    metadata = {
        "mode": args.mode,
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "runtime_seconds": round(perf_counter() - t0, 3),
        "config": str(args.config),
        "dr1_catalogue": str(dr1_path),
        "dr1_summary": dr1_summary,
        "flux_cuts_jy": flux_cuts,
        "random_shifts_ra_dec_deg": shifts,
        "chance_correction": "(F_obs - F_rand) / (1 - F_rand)",
        "support_definition": "DR1 component coordinate inside predicted radio bbox",
        "n_variants_with_catalogues": int(len(predictions_by_variant)),
        "n_missing_variants": int(len(missing_rows)),
        "prediction_catalogues": {variant: str(path) if path else "" for variant, path in path_by_variant.items()},
        "note": "Only variants with full-run predicted source catalogues are evaluated. Missing variants are not inferred from tile smoke outputs.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.evaluate_missing_as_error and missing_rows:
        raise SystemExit(f"Missing full-run prediction catalogues for variants: {[row['variant'] for row in missing_rows]}")
    print(summary.to_string(index=False))
    if not missing.empty:
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
