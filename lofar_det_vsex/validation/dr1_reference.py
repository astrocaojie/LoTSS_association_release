"""Load and normalize the LoTSS DR1 component-reference catalogue.

This module intentionally uses only the manually curated DR1 component
catalogue.  DR1 optical-ID tables are not valid reference truth for this
validation path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DR1_COMPONENT_CSV = Path(
    os.environ.get("LOTSS_DR1_COMPONENT_CSV", "data/dr1/lotss_dr1_component_catalogue.csv.gz")
)
DEFAULT_DR1_COMPONENT_FITS = Path(
    os.environ.get("LOTSS_DR1_COMPONENT_FITS", "data/dr1/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits")
)

RA_CANDIDATES = (
    "ra",
    "radio_ra",
    "component_ra",
    "RA",
    "RAJ2000",
    "RA_deg",
    "ra_deg",
)
DEC_CANDIDATES = (
    "dec",
    "decl",
    "radio_dec",
    "component_dec",
    "DEC",
    "DEJ2000",
    "Dec_deg",
    "dec_deg",
)
COMPONENT_ID_CANDIDATES = ("component_id", "Component_Name", "component_name", "component", "gaussian_id")
SOURCE_ID_CANDIDATES = ("source_id", "Source_Name", "source_name", "Source_ID", "assoc_id", "association_id")


def _find_column(columns: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    exact = {col: col for col in columns}
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    if required:
        raise ValueError(f"Could not identify required column among candidates {candidates}; columns={columns}")
    return None


def load_dr1_component_catalogue(
    path: str | Path = DEFAULT_DR1_COMPONENT_CSV,
    *,
    ra_col: str | None = None,
    dec_col: str | None = None,
    report_path: str | Path | None = None,
    print_summary: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the standardized DR1 component catalogue and return a normalized copy.

    The returned table always contains numeric ``ra``/``dec`` columns, a
    ``valid_position`` flag, and string ``dr1_component_id`` /
    ``dr1_source_id`` columns when the corresponding identifiers are present.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DR1 component catalogue not found: {path}")

    frame = pd.read_csv(path)
    columns = list(frame.columns)
    ra_name = ra_col or _find_column(columns, RA_CANDIDATES)
    dec_name = dec_col or _find_column(columns, DEC_CANDIDATES)
    component_name = _find_column(columns, COMPONENT_ID_CANDIDATES, required=False)
    source_name = _find_column(columns, SOURCE_ID_CANDIDATES, required=False)

    out = frame.copy()
    out["ra"] = pd.to_numeric(out[ra_name], errors="coerce") % 360.0
    out["dec"] = pd.to_numeric(out[dec_name], errors="coerce")
    out["valid_position"] = out["ra"].between(0.0, 360.0, inclusive="left") & out["dec"].between(-90.0, 90.0, inclusive="both")
    if component_name is not None:
        out["dr1_component_id"] = out[component_name].astype(str)
    else:
        out["dr1_component_id"] = [f"dr1_component_row_{idx}" for idx in range(len(out))]
    if source_name is not None:
        out["dr1_source_id"] = out[source_name].astype(str)
    else:
        out["dr1_source_id"] = out["dr1_component_id"].astype(str)

    duplicate_subset = ["dr1_source_id", "dr1_component_id", "ra", "dec"]
    valid = out.loc[out["valid_position"]].copy()
    summary: dict[str, Any] = {
        "path": str(path),
        "n_rows": int(len(out)),
        "n_columns": int(len(out.columns)),
        "identified_ra_column": ra_name,
        "identified_dec_column": dec_name,
        "identified_component_id_column": component_name,
        "identified_source_id_column": source_name,
        "n_valid_position": int(out["valid_position"].sum()),
        "n_invalid_position": int((~out["valid_position"]).sum()),
        "n_unique_component_id": int(valid["dr1_component_id"].nunique(dropna=True)),
        "n_unique_source_id": int(valid["dr1_source_id"].nunique(dropna=True)),
        "n_duplicate_source_component_position_rows": int(out.duplicated(duplicate_subset).sum()),
        "ra_min": float(valid["ra"].min()) if not valid.empty else float("nan"),
        "ra_max": float(valid["ra"].max()) if not valid.empty else float("nan"),
        "dec_min": float(valid["dec"].min()) if not valid.empty else float("nan"),
        "dec_max": float(valid["dec"].max()) if not valid.empty else float("nan"),
        "reference_policy": "LoTSS DR1 component catalogue only; no DR1 optical-ID catalogue used.",
        "no_optical_ids_used": True,
    }

    message = (
        "DR1 component catalogue loaded: "
        f"rows={summary['n_rows']} valid_positions={summary['n_valid_position']} "
        f"RA={ra_name} Dec={dec_name} component_id={component_name} source_id={source_name}"
    )
    if print_summary:
        print(message)

    if report_path is not None:
        write_dr1_catalogue_report(report_path, summary, columns)

    return out, summary


def write_dr1_catalogue_report(path: str | Path, summary: dict[str, Any], columns: list[str]) -> None:
    """Write a compact human-readable inspection report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LoTSS DR1 Component Catalogue Inspection",
        "",
        "Reference policy: use the DR1 component catalogue only; do not use DR1 optical IDs as truth.",
        "",
        "Identified columns:",
        f"- RA: {summary.get('identified_ra_column')}",
        f"- Dec: {summary.get('identified_dec_column')}",
        f"- Component ID: {summary.get('identified_component_id_column')}",
        f"- Source/association ID: {summary.get('identified_source_id_column')}",
        "",
        "Summary:",
    ]
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, (float, np.floating)):
            lines.append(f"- {key}: {value:.8g}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "Columns:", *[f"- {col}" for col in columns]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
