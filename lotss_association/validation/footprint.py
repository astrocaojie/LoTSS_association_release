"""Sky-footprint helpers for DR1 component-reference validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SkyFootprint:
    """Coarse sky footprint represented as occupied RA/Dec grid cells."""

    grid_size_deg: float
    cells: frozenset[tuple[int, int]]

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def cell_for(self, ra: float, dec: float) -> tuple[int, int]:
        ra_cell = int((float(ra) % 360.0) // self.grid_size_deg)
        dec_cell = int((float(dec) + 90.0) // self.grid_size_deg)
        return ra_cell, dec_cell

    def contains(self, ra: float, dec: float) -> bool:
        if pd.isna(ra) or pd.isna(dec):
            return False
        dec_f = float(dec)
        if dec_f < -90.0 or dec_f > 90.0:
            return False
        return self.cell_for(float(ra), dec_f) in self.cells

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"ra_cell": ra_cell, "dec_cell": dec_cell} for ra_cell, dec_cell in sorted(self.cells)])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        frame.insert(0, "grid_size_deg", float(self.grid_size_deg))
        frame.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "SkyFootprint":
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Empty footprint mask: {path}")
        grid_size = float(frame["grid_size_deg"].iloc[0])
        cells = frozenset((int(row.ra_cell), int(row.dec_cell)) for row in frame.itertuples(index=False))
        return cls(grid_size_deg=grid_size, cells=cells)


def build_dr1_sky_footprint(dr1_table: pd.DataFrame, grid_size_deg: float = 0.25, save_path: str | Path | None = None) -> SkyFootprint:
    """Build a coarse DR1 component sky footprint with no boundary dilation."""

    if "ra" not in dr1_table or "dec" not in dr1_table:
        raise ValueError("DR1 table must contain normalized ra/dec columns")
    valid_mask = dr1_table["valid_position"].astype(bool) if "valid_position" in dr1_table else pd.Series(True, index=dr1_table.index)
    valid = dr1_table.loc[valid_mask].copy()
    if valid.empty:
        raise ValueError("DR1 table has no valid sky positions")
    cells = frozenset(
        (int((float(row.ra) % 360.0) // grid_size_deg), int((float(row.dec) + 90.0) // grid_size_deg))
        for row in valid[["ra", "dec"]].itertuples(index=False)
    )
    footprint = SkyFootprint(grid_size_deg=float(grid_size_deg), cells=cells)
    if save_path is not None:
        footprint.save(save_path)
    return footprint


def add_bbox_centre(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return predictions with ``ra_centre``/``dec_centre`` filled from bbox columns."""

    out = predictions.copy()
    if "ra_centre" not in out:
        ra_min = pd.to_numeric(out["bbox_ra_min"], errors="coerce") % 360.0
        ra_max = pd.to_numeric(out["bbox_ra_max"], errors="coerce") % 360.0
        width = (ra_max - ra_min) % 360.0
        out["ra_centre"] = (ra_min + 0.5 * width) % 360.0
    if "dec_centre" not in out:
        out["dec_centre"] = (pd.to_numeric(out["bbox_dec_min"], errors="coerce") + pd.to_numeric(out["bbox_dec_max"], errors="coerce")) / 2.0
    return out


def filter_predictions_in_footprint(
    predictions: pd.DataFrame,
    footprint: SkyFootprint,
    *,
    use_bbox_centre: bool = True,
    ra_col: str = "ra_centre",
    dec_col: str = "dec_centre",
) -> pd.DataFrame:
    """Add an ``in_dr1_footprint`` flag to predicted-source rows."""

    out = predictions.copy()
    if use_bbox_centre:
        out = add_bbox_centre(out)
    if ra_col not in out or dec_col not in out:
        raise ValueError(f"Prediction table must contain {ra_col}/{dec_col} or bbox columns")
    out["in_dr1_footprint"] = [
        footprint.contains(ra, dec)
        for ra, dec in zip(pd.to_numeric(out[ra_col], errors="coerce"), pd.to_numeric(out[dec_col], errors="coerce"))
    ]
    out.attrs["dr1_footprint_grid_size_deg"] = footprint.grid_size_deg
    out.attrs["dr1_footprint_n_cells"] = footprint.n_cells
    return out


def footprint_summary(footprint: SkyFootprint) -> dict[str, Any]:
    return {"grid_size_deg": float(footprint.grid_size_deg), "n_cells": int(footprint.n_cells), "boundary_dilation": 0}
