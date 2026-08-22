"""Support-rate metrics for DR1 component-reference validation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _flux_series(frame: pd.DataFrame, flux_col: str) -> pd.Series:
    candidates = [flux_col, "total_flux_jy", "total_flux", "total_flux_gaussian", "Total_flux"]
    for candidate in candidates:
        if candidate in frame:
            return pd.to_numeric(frame[candidate], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)


def compute_support_rates(
    predictions: pd.DataFrame,
    flux_cuts: Iterable[float] = (0.0, 0.05, 0.1),
    *,
    sample_col: str = "sample",
    flux_col: str = "total_flux_jy",
    footprint_col: str = "in_dr1_footprint",
    support_col: str = "supported_by_dr1",
    variant: str | None = None,
) -> pd.DataFrame:
    """Compute DR1-supported agreement rates by sample and flux threshold."""

    if sample_col not in predictions:
        raise ValueError(f"Prediction table missing sample column: {sample_col}")
    if footprint_col not in predictions:
        raise ValueError(f"Prediction table missing footprint flag: {footprint_col}")
    if support_col not in predictions:
        raise ValueError(f"Prediction table missing support flag: {support_col}")

    work = predictions.copy()
    work["_flux_for_cut"] = _flux_series(work, flux_col)
    rows = []
    for sample, group in work.groupby(sample_col, dropna=False):
        for cut in flux_cuts:
            in_footprint = group.loc[group[footprint_col].astype(bool) & (group["_flux_for_cut"] >= float(cut))]
            n = int(len(in_footprint))
            k = int(in_footprint[support_col].astype(bool).sum()) if n else 0
            low, high = wilson_interval(k, n)
            rows.append(
                {
                    "variant": variant,
                    "sample": sample,
                    "flux_cut_jy": float(cut),
                    "n_in_dr1_footprint": n,
                    "n_supported_by_dr1_component": k,
                    "support_rate": (k / n) if n else float("nan"),
                    "binomial_or_wilson_95ci_low": low,
                    "binomial_or_wilson_95ci_high": high,
                    "metric_name": "DR1-supported agreement rate",
                    "support_definition": "support fraction under bbox containment",
                }
            )
    return pd.DataFrame(rows)


def support_table_to_latex(table: pd.DataFrame, path: str) -> None:
    """Write a compact LaTeX table."""

    path_obj = pd.io.common.stringify_path(path)
    cols = [
        "variant",
        "sample",
        "flux_cut_jy",
        "n_in_dr1_footprint",
        "n_supported_by_dr1_component",
        "support_rate",
        "binomial_or_wilson_95ci_low",
        "binomial_or_wilson_95ci_high",
    ]
    table.loc[:, [col for col in cols if col in table.columns]].to_latex(path_obj, index=False, float_format="%.4f")
