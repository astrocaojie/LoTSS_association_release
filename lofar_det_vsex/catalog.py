"""PyBDSF Gaussian catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from astropy.table import Table

from .utils import first_existing_key, get_logger, safe_float


FIELD_ALIASES = {
    "source_id": ["Source_id", "SOURCE_ID", "source_id", "Source_id_1"],
    "island_id": ["Isl_id", "ISL_ID", "Island_id", "island_id"],
    "gaussian_id": ["Gaussian_id", "GAUSSIAN_ID", "Gaus_id", "gaussian_id"],
    "ra": ["RA", "ra", "RA_deg", "ra_deg"],
    "dec": ["DEC", "Dec", "dec", "DEC_deg", "dec_deg"],
    "e_ra": ["E_RA", "e_RA", "ra_err", "E_RA_deg"],
    "e_dec": ["E_DEC", "e_DEC", "dec_err", "E_DEC_deg"],
    "total_flux": ["Total_flux", "TOTAL_FLUX", "total_flux", "Flux", "flux"],
    "peak_flux": ["Peak_flux", "PEAK_FLUX", "peak_flux"],
    "maj": ["Maj", "MAJ", "maj"],
    "min": ["Min", "MIN", "min"],
    "pa": ["PA", "pa"],
    "dc_maj": ["DC_Maj", "DC_MAJ", "dc_maj"],
    "dc_min": ["DC_Min", "DC_MIN", "dc_min"],
    "dc_pa": ["DC_PA", "dc_pa"],
    "s_code": ["S_Code", "S_CODE", "s_code"],
    "x": ["x", "X", "x_pix", "X_IMAGE"],
    "y": ["y", "Y", "y_pix", "Y_IMAGE"],
}


@dataclass
class CatalogColumns:
    source_id: str | None = None
    island_id: str | None = None
    gaussian_id: str | None = None
    ra: str | None = None
    dec: str | None = None
    e_ra: str | None = None
    e_dec: str | None = None
    total_flux: str | None = None
    peak_flux: str | None = None
    maj: str | None = None
    min: str | None = None
    pa: str | None = None
    dc_maj: str | None = None
    dc_min: str | None = None
    dc_pa: str | None = None
    s_code: str | None = None
    x: str | None = None
    y: str | None = None


def read_gaussian_catalog(path: str | Path) -> Table:
    """Read a PyBDSF Gaussian FITS catalog."""

    path = Path(path)
    table = Table.read(path)
    return table


def detect_catalog_columns(table: Table) -> CatalogColumns:
    """Detect common PyBDSF Gaussian catalog fields."""

    mapping = {name: table[name] for name in table.colnames}
    detected = {}
    for canonical, aliases in FIELD_ALIASES.items():
        detected[canonical] = first_existing_key(mapping, aliases)
    return CatalogColumns(**detected)


def warn_missing_columns(columns: CatalogColumns, required: list[str] | None = None) -> None:
    """Warn about missing catalog columns."""

    required = required or ["ra", "dec", "total_flux", "peak_flux"]
    missing = [name for name in required if getattr(columns, name) is None]
    if missing:
        warnings.warn(f"Gaussian catalog missing expected columns: {missing}")


def print_catalog_summary(path: str | Path, n_rows: int = 5) -> None:
    """Print column names, dtypes, and the first rows of a FITS catalog."""

    table = read_gaussian_catalog(path)
    columns = detect_catalog_columns(table)
    print(f"Catalog: {path}")
    print(f"Rows: {len(table)}")
    print("Columns:")
    for name in table.colnames:
        print(f"  - {name}: {table[name].dtype}")
    print("\nDetected aliases:")
    for field_name in columns.__dataclass_fields__:
        print(f"  {field_name}: {getattr(columns, field_name)}")
    print(f"\nFirst {min(n_rows, len(table))} rows:")
    if len(table) > 0:
        print(table[:n_rows])


def table_to_dataframe(table: Table) -> pd.DataFrame:
    """Convert an astropy Table to pandas with byte strings decoded."""

    df = table.to_pandas()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda value: value.decode("utf-8", errors="replace")
                if isinstance(value, (bytes, np.bytes_))
                else value
            )
    return df


def normalized_gaussian_dataframe(table: Table) -> tuple[pd.DataFrame, CatalogColumns]:
    """Return a DataFrame with canonical helper columns added where possible."""

    columns = detect_catalog_columns(table)
    warn_missing_columns(columns)
    df = table_to_dataframe(table)
    n = len(df)

    def copy_or_default(canonical: str, default: Any) -> None:
        original = getattr(columns, canonical)
        out_name = f"_{canonical}"
        if original is not None and original in df:
            df[out_name] = df[original]
        else:
            df[out_name] = default() if callable(default) else default

    copy_or_default("source_id", lambda: np.arange(n))
    copy_or_default("island_id", lambda: np.full(n, -1))
    copy_or_default("gaussian_id", lambda: np.arange(n))
    copy_or_default("ra", lambda: np.full(n, np.nan))
    copy_or_default("dec", lambda: np.full(n, np.nan))
    copy_or_default("total_flux", lambda: np.full(n, np.nan))
    copy_or_default("peak_flux", lambda: np.full(n, np.nan))
    copy_or_default("maj", lambda: np.full(n, np.nan))
    copy_or_default("min", lambda: np.full(n, np.nan))
    copy_or_default("pa", lambda: np.full(n, np.nan))
    copy_or_default("dc_maj", lambda: np.full(n, np.nan))
    copy_or_default("dc_min", lambda: np.full(n, np.nan))
    copy_or_default("dc_pa", lambda: np.full(n, np.nan))
    copy_or_default("s_code", lambda: np.full(n, ""))
    copy_or_default("x", lambda: np.full(n, np.nan))
    copy_or_default("y", lambda: np.full(n, np.nan))

    for numeric in [
        "_ra",
        "_dec",
        "_total_flux",
        "_peak_flux",
        "_maj",
        "_min",
        "_pa",
        "_dc_maj",
        "_dc_min",
        "_dc_pa",
        "_x",
        "_y",
    ]:
        df[numeric] = pd.to_numeric(df[numeric], errors="coerce")

    return df, columns


def select_catalog_region(
    df: pd.DataFrame,
    center_ra: float | None,
    center_dec: float | None,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Preselect Gaussian rows near a cutout center."""

    if center_ra is None or center_dec is None:
        return df
    if "_ra" not in df or "_dec" not in df:
        return df
    cos_dec = np.cos(np.deg2rad(center_dec))
    dra_arcsec = (df["_ra"].to_numpy(float) - center_ra) * cos_dec * 3600.0
    ddec_arcsec = (df["_dec"].to_numpy(float) - center_dec) * 3600.0
    dist = np.hypot(dra_arcsec, ddec_arcsec)
    keep = np.isfinite(dist) & (dist <= radius_arcsec)
    return df.loc[keep].copy()


def gaussian_row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a normalized Gaussian row to a compact dict."""

    return {
        "source_id": row.get("_source_id"),
        "island_id": row.get("_island_id"),
        "gaussian_id": row.get("_gaussian_id"),
        "ra": safe_float(row.get("_ra")),
        "dec": safe_float(row.get("_dec")),
        "total_flux": safe_float(row.get("_total_flux")),
        "peak_flux": safe_float(row.get("_peak_flux")),
        "maj": safe_float(row.get("_maj")),
        "min": safe_float(row.get("_min")),
        "pa": safe_float(row.get("_pa")),
        "dc_maj": safe_float(row.get("_dc_maj")),
        "dc_min": safe_float(row.get("_dc_min")),
        "dc_pa": safe_float(row.get("_dc_pa")),
        "s_code": row.get("_s_code"),
        "x": safe_float(row.get("_x")),
        "y": safe_float(row.get("_y")),
    }


def log_catalog_detection(columns: CatalogColumns) -> None:
    """Log detected catalog column mapping."""

    logger = get_logger()
    logger.info("Detected Gaussian catalog columns:")
    for field_name in columns.__dataclass_fields__:
        logger.info("  %s -> %s", field_name, getattr(columns, field_name))
