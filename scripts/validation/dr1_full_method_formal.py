#!/usr/bin/env python
"""Formal full-method DR1 component-only validation tables and controls."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lofar_det_vsex.validation.bbox_support import match_predictions_to_dr1_components
from lofar_det_vsex.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue
from lofar_det_vsex.validation.metrics import compute_support_rates, support_table_to_latex


DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/dr1_validation/full_method_sanity/full_method_predictions_with_dr1_support.csv.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/dr1_validation/full_method_formal"
DEFAULT_SHIFTS = "1,0;-1,0;0,1;0,-1;1,1;-1,-1;2,0;0,2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--dr1-catalogue", default=str(DEFAULT_DR1_COMPONENT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--grid-size-deg", type=float, default=0.25)
    parser.add_argument("--flux-cuts", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--shifts", default=DEFAULT_SHIFTS, help="Semicolon separated RA,Dec shifts in degrees.")
    parser.add_argument("--max-random-rows", type=int, default=None, help="Debug only: cap in-footprint rows used for random-shift matching.")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_shifts(value: str) -> list[tuple[float, float]]:
    shifts = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid shift pair {item!r}; expected ra,dec")
        shifts.append((float(parts[0]), float(parts[1])))
    if not shifts:
        raise ValueError("At least one random-shift pair is required")
    return shifts


def load_predictions(path: str | Path) -> pd.DataFrame:
    cols = [
        "source_id",
        "sample",
        "bbox_ra_min",
        "bbox_ra_max",
        "bbox_dec_min",
        "bbox_dec_max",
        "total_flux_jy",
        "n_gaussians",
        "quality_class",
        "in_dr1_footprint",
        "supported_by_dr1",
    ]
    return pd.read_csv(path, usecols=cols)


def add_bbox_area(predictions: pd.DataFrame) -> pd.DataFrame:
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


def random_shift_summary(
    predictions: pd.DataFrame,
    dr1: pd.DataFrame,
    flux_cuts: list[float],
    shifts: list[tuple[float, float]],
    grid_size_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_shift = []
    shifted_rows = []
    random_input = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    for idx, (ra_shift, dec_shift) in enumerate(shifts):
        shifted = dr1.copy()
        shifted["ra"] = (pd.to_numeric(shifted["ra"], errors="coerce") + ra_shift) % 360.0
        shifted["dec"] = pd.to_numeric(shifted["dec"], errors="coerce") + dec_shift
        shifted["valid_position"] = shifted["dec"].between(-90.0, 90.0, inclusive="both")
        matched = match_predictions_to_dr1_components(random_input, shifted, grid_size_deg=grid_size_deg)
        matched["supported_by_dr1_random_shift"] = matched["supported_by_dr1"].astype(bool)
        matched["supported_by_dr1"] = matched["supported_by_dr1_random_shift"]
        rates = compute_support_rates(matched, flux_cuts, variant=f"random_shift_{idx:02d}")
        rates["random_shift_index"] = idx
        rates["ra_shift_deg"] = ra_shift
        rates["dec_shift_deg"] = dec_shift
        per_shift.append(rates)
        shifted_rows.append(
            {
                "random_shift_index": idx,
                "ra_shift_deg": ra_shift,
                "dec_shift_deg": dec_shift,
                "n_shifted_dr1_components_valid": int(shifted["valid_position"].sum()),
            }
        )
    return pd.concat(per_shift, ignore_index=True), pd.DataFrame(shifted_rows)


def corrected_support_table(observed: pd.DataFrame, random_rates: pd.DataFrame) -> pd.DataFrame:
    random_summary = (
        random_rates.groupby(["sample", "flux_cut_jy"], as_index=False)
        .agg(
            random_support_rate_mean=("support_rate", "mean"),
            random_support_rate_std=("support_rate", "std"),
            random_support_rate_min=("support_rate", "min"),
            random_support_rate_max=("support_rate", "max"),
            n_random_shifts=("support_rate", "count"),
        )
    )
    merged = observed.merge(random_summary, on=["sample", "flux_cut_jy"], how="left")
    merged = merged.rename(columns={"support_rate": "observed_support_rate"})
    random_mean = pd.to_numeric(merged["random_support_rate_mean"], errors="coerce").fillna(0.0)
    observed_rate = pd.to_numeric(merged["observed_support_rate"], errors="coerce")
    denom = (1.0 - random_mean).replace(0.0, np.nan)
    merged["excess_support_rate_over_random"] = observed_rate - random_mean
    merged["chance_corrected_support_rate"] = (observed_rate - random_mean) / denom
    merged["chance_corrected_support_rate"] = merged["chance_corrected_support_rate"].clip(lower=0.0, upper=1.0)
    merged["metric_name"] = "DR1-supported agreement rate with random-shift control"
    return merged


def area_diagnostics(predictions: pd.DataFrame, flux_cuts: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    rows = []
    bin_rows = []
    largest_rows = []
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
            top10_supported = int((supported & top10).sum())
            total_supported = int(supported.sum())
            rows.append(
                {
                    "sample": sample,
                    "flux_cut_jy": float(cut),
                    "n_in_dr1_footprint": int(len(subset)),
                    "median_bbox_area_arcsec2": p50,
                    "p90_bbox_area_arcsec2": p90,
                    "p99_bbox_area_arcsec2": p99,
                    "max_bbox_area_arcsec2": max_area,
                    "support_rate_all": float(supported.mean()),
                    "support_rate_lower_90pct_area": float(supported[low90].mean()) if int(low90.sum()) else np.nan,
                    "support_rate_top_10pct_area": float(supported[top10].mean()) if int(top10.sum()) else np.nan,
                    "fraction_supported_from_top_10pct_area": float(top10_supported / total_supported) if total_supported else np.nan,
                    "large_bbox_driven_flag": bool(total_supported > 0 and top10_supported / total_supported > 0.5),
                }
            )
            edges = [-np.inf, p50, p90, p99, np.inf]
            labels = ["<=p50", "p50-p90", "p90-p99", ">p99"]
            subset["_area_bin"] = pd.cut(area, bins=edges, labels=labels, include_lowest=True)
            for label, frame in subset.groupby("_area_bin", observed=False):
                if frame.empty:
                    continue
                frame_area = pd.to_numeric(frame["bbox_area_arcsec2_approx"], errors="coerce")
                frame_supported = frame["supported_by_dr1"].astype(bool)
                bin_rows.append(
                    {
                        "sample": sample,
                        "flux_cut_jy": float(cut),
                        "bbox_area_bin": str(label),
                        "n_in_bin": int(len(frame)),
                        "n_supported": int(frame_supported.sum()),
                        "support_rate": float(frame_supported.mean()),
                        "bbox_area_min_arcsec2": float(frame_area.min()),
                        "bbox_area_median_arcsec2": float(frame_area.median()),
                        "bbox_area_max_arcsec2": float(frame_area.max()),
                        "supported_contribution_fraction": float(frame_supported.sum() / total_supported) if total_supported else np.nan,
                    }
                )
            largest = subset.nlargest(20, "bbox_area_arcsec2_approx").copy()
            largest["flux_cut_jy"] = float(cut)
            largest_rows.append(
                largest[
                    [
                        "sample",
                        "flux_cut_jy",
                        "source_id",
                        "total_flux_jy",
                        "n_gaussians",
                        "supported_by_dr1",
                        "bbox_width_deg",
                        "bbox_height_deg",
                        "bbox_area_arcsec2_approx",
                    ]
                ]
            )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows), pd.concat(largest_rows, ignore_index=True) if largest_rows else pd.DataFrame()


def write_plots(corrected: pd.DataFrame, random_rates: pd.DataFrame, area_bins: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sample, frame in corrected.groupby("sample"):
        frame = frame.sort_values("flux_cut_jy")
        x = frame["flux_cut_jy"].to_numpy(float)
        y = frame["observed_support_rate"].to_numpy(float)
        low = frame["binomial_or_wilson_95ci_low"].to_numpy(float)
        high = frame["binomial_or_wilson_95ci_high"].to_numpy(float)
        ax.plot(x, y, marker="o", label=str(sample))
        ax.fill_between(x, low, high, alpha=0.18)
    ax.set_xlabel("Flux cut (Jy)")
    ax.set_ylabel("DR1-supported agreement rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "support_rate_vs_flux_cut.png", dpi=180)
    fig.savefig(plot_dir / "support_rate_vs_flux_cut.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    x = np.arange(len(corrected))
    width = 0.26
    sorted_frame = corrected.sort_values(["sample", "flux_cut_jy"]).reset_index(drop=True)
    labels = [f"{row.sample}\n>={row.flux_cut_jy:g} Jy" for row in sorted_frame.itertuples(index=False)]
    ax.bar(x - width, sorted_frame["observed_support_rate"], width, label="observed")
    ax.bar(x, sorted_frame["random_support_rate_mean"], width, label="random-shift mean")
    ax.bar(x + width, sorted_frame["chance_corrected_support_rate"], width, label="chance-corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Support fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "observed_vs_random_support.png", dpi=180)
    fig.savefig(plot_dir / "observed_vs_random_support.pdf")
    plt.close(fig)

    if not area_bins.empty:
        for sample, sample_df in area_bins.groupby("sample"):
            fig, ax = plt.subplots(figsize=(7, 4.2))
            pivot = sample_df.pivot_table(index="bbox_area_bin", columns="flux_cut_jy", values="support_rate", observed=False)
            pivot = pivot.reindex(["<=p50", "p50-p90", "p90-p99", ">p99"])
            pivot.plot(kind="bar", ax=ax)
            ax.set_xlabel("BBox area bin")
            ax.set_ylabel("DR1-supported agreement rate")
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(title="Flux cut Jy")
            fig.tight_layout()
            safe = str(sample).replace("/", "_")
            fig.savefig(plot_dir / f"bbox_area_bin_support_{safe}.png", dpi=180)
            fig.savefig(plot_dir / f"bbox_area_bin_support_{safe}.pdf")
            plt.close(fig)


def write_readme(output_dir: Path, corrected: pd.DataFrame, area_diag: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Full-Method Formal DR1 Validation",
        "",
        "Reference policy: LoTSS DR1 component catalogue only; no DR1 optical-ID catalogue was used.",
        "",
        "Metric wording: DR1-supported agreement rate / support fraction under bbox containment.",
        "",
        "Chance correction: `(observed_support_rate - random_shift_mean) / (1 - random_shift_mean)`.",
        "",
        "## Corrected Support",
        "",
        "```csv",
        corrected.to_csv(index=False).strip(),
        "```",
        "",
        "## BBox Area Diagnostics",
        "",
        "```csv",
        area_diag.to_csv(index=False).strip(),
        "```",
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
    shifts = parse_shifts(args.shifts)

    predictions = load_predictions(args.predictions)
    predictions = add_bbox_area(predictions)
    if args.max_random_rows is not None:
        in_fp = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].head(int(args.max_random_rows)).copy()
        predictions = pd.concat([in_fp, predictions.loc[~predictions["in_dr1_footprint"].astype(bool)].head(0)], ignore_index=True)

    observed = compute_support_rates(predictions, args.flux_cuts, variant="full_method_component_only")
    dr1, dr1_summary = load_dr1_component_catalogue(args.dr1_catalogue, print_summary=True)
    random_rates, shift_inventory = random_shift_summary(predictions, dr1, args.flux_cuts, shifts, args.grid_size_deg)
    corrected = corrected_support_table(observed, random_rates)
    area_diag, area_bins, largest = area_diagnostics(predictions, args.flux_cuts)
    paper = corrected.merge(area_diag, on=["sample", "flux_cut_jy"], how="left")

    in_fp = predictions.loc[predictions["in_dr1_footprint"].astype(bool)].copy()
    in_fp.to_csv(output_dir / "full_method_predictions_in_dr1_footprint_with_area.csv.gz", index=False, compression="gzip")
    observed.to_csv(output_dir / "full_method_observed_support_summary.csv", index=False)
    support_table_to_latex(observed, str(output_dir / "full_method_observed_support_summary.tex"))
    random_rates.to_csv(output_dir / "full_method_random_shift_support_by_shift.csv", index=False)
    corrected.to_csv(output_dir / "full_method_random_shift_corrected_support.csv", index=False)
    corrected.to_latex(output_dir / "full_method_random_shift_corrected_support.tex", index=False, float_format="%.4f")
    paper.to_csv(output_dir / "full_method_paper_validation_results.csv", index=False)
    paper.to_latex(output_dir / "full_method_paper_validation_results.tex", index=False, float_format="%.4f")
    area_diag.to_csv(output_dir / "full_method_bbox_area_diagnostics.csv", index=False)
    area_bins.to_csv(output_dir / "full_method_bbox_area_bin_support.csv", index=False)
    largest.to_csv(output_dir / "full_method_largest_bbox_examples.csv", index=False)
    shift_inventory.to_csv(output_dir / "random_shift_inventory.csv", index=False)
    write_plots(corrected, random_rates, area_bins, output_dir)

    metadata = {
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "runtime_seconds": round(perf_counter() - t0, 3),
        "predictions": str(args.predictions),
        "dr1_catalogue": str(args.dr1_catalogue),
        "grid_size_deg": args.grid_size_deg,
        "flux_cuts": args.flux_cuts,
        "random_shifts_ra_dec_deg": shifts,
        "n_prediction_rows_loaded": int(len(predictions)),
        "n_predictions_in_dr1_footprint": int(predictions["in_dr1_footprint"].astype(bool).sum()),
        "dr1_summary": dr1_summary,
        "support_definition": "DR1 component coordinate inside predicted radio bbox",
        "chance_correction": "(observed - random_mean) / (1 - random_mean)",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_readme(output_dir, corrected, area_diag, metadata)
    print(corrected.to_string(index=False))
    print(area_diag.to_string(index=False))


if __name__ == "__main__":
    main()
