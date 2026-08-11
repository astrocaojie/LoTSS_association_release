"""BBox-containment matching against DR1 component positions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


BBOX_COLUMNS = ("bbox_ra_min", "bbox_ra_max", "bbox_dec_min", "bbox_dec_max")


def _normalise_bbox_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    aliases = {
        "bbox_ra_min": ("bbox_ra_min", "ra_min", "ra_min_deg", "radio_bbox_ra_min"),
        "bbox_ra_max": ("bbox_ra_max", "ra_max", "ra_max_deg", "radio_bbox_ra_max"),
        "bbox_dec_min": ("bbox_dec_min", "dec_min", "dec_min_deg", "radio_bbox_dec_min"),
        "bbox_dec_max": ("bbox_dec_max", "dec_max", "dec_max_deg", "radio_bbox_dec_max"),
    }
    for target, candidates in aliases.items():
        if target in out:
            continue
        for candidate in candidates:
            if candidate in out:
                out[target] = out[candidate]
                break
    missing = [col for col in BBOX_COLUMNS if col not in out]
    if missing:
        raise ValueError(f"Prediction table missing bbox columns: {missing}")
    for col in BBOX_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["bbox_ra_min"] = out["bbox_ra_min"] % 360.0
    out["bbox_ra_max"] = out["bbox_ra_max"] % 360.0
    return out


def bbox_contains(
    ra: np.ndarray | pd.Series,
    dec: np.ndarray | pd.Series,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    *,
    include_edges: bool = True,
) -> np.ndarray:
    """Return mask for points inside one sky bbox, including RA wrap-around."""

    ra_values = np.asarray(ra, dtype=float) % 360.0
    dec_values = np.asarray(dec, dtype=float)
    ra_min = float(ra_min) % 360.0
    ra_max = float(ra_max) % 360.0
    dec_min = float(dec_min)
    dec_max = float(dec_max)
    if dec_min > dec_max:
        dec_min, dec_max = dec_max, dec_min
    if include_edges:
        dec_mask = (dec_values >= dec_min) & (dec_values <= dec_max)
        if ra_min <= ra_max:
            ra_mask = (ra_values >= ra_min) & (ra_values <= ra_max)
        else:
            ra_mask = (ra_values >= ra_min) | (ra_values <= ra_max)
    else:
        dec_mask = (dec_values > dec_min) & (dec_values < dec_max)
        if ra_min <= ra_max:
            ra_mask = (ra_values > ra_min) & (ra_values < ra_max)
        else:
            ra_mask = (ra_values > ra_min) | (ra_values < ra_max)
    return ra_mask & dec_mask


def _cell_ranges(ra_min: float, ra_max: float, dec_min: float, dec_max: float, grid_size_deg: float) -> list[tuple[int, int]]:
    if dec_min > dec_max:
        dec_min, dec_max = dec_max, dec_min
    dec_start = int(max(-90.0, dec_min) + 90.0) // int(grid_size_deg) if float(grid_size_deg).is_integer() else int((max(-90.0, dec_min) + 90.0) // grid_size_deg)
    dec_stop = int((min(90.0, dec_max) + 90.0) // grid_size_deg)
    ra_min = ra_min % 360.0
    ra_max = ra_max % 360.0
    n_ra = int(np.ceil(360.0 / grid_size_deg))

    def ra_cells(start: float, stop: float) -> list[int]:
        first = int(start // grid_size_deg)
        last = int(stop // grid_size_deg)
        return [cell % n_ra for cell in range(first, last + 1)]

    if ra_min <= ra_max:
        ra_list = ra_cells(ra_min, ra_max)
    else:
        ra_list = ra_cells(ra_min, 360.0 - 1e-12) + ra_cells(0.0, ra_max)
    return [(ra_cell, dec_cell) for ra_cell in ra_list for dec_cell in range(dec_start, dec_stop + 1)]


def _build_component_cell_index(dr1_components: pd.DataFrame, grid_size_deg: float) -> dict[tuple[int, int], list[int]]:
    valid_mask = dr1_components["valid_position"].astype(bool) if "valid_position" in dr1_components else pd.Series(True, index=dr1_components.index)
    valid = dr1_components.loc[valid_mask].copy()
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, row in valid[["ra", "dec"]].iterrows():
        ra_cell = int((float(row["ra"]) % 360.0) // grid_size_deg)
        dec_cell = int((float(row["dec"]) + 90.0) // grid_size_deg)
        cells[(ra_cell, dec_cell)].append(idx)
    return cells


def match_predictions_to_dr1_components(
    predictions: pd.DataFrame,
    dr1_components: pd.DataFrame,
    *,
    prediction_id_col: str = "source_id",
    grid_size_deg: float = 0.25,
    include_edges: bool = True,
) -> pd.DataFrame:
    """Mark each predicted source supported when any DR1 component lies in its bbox."""

    preds = _normalise_bbox_columns(predictions)
    if preds.empty:
        out = preds.copy()
        out["supported_by_dr1"] = pd.Series(dtype=bool)
        out["n_matched_dr1_components"] = pd.Series(dtype=int)
        out["n_matched_dr1_sources"] = pd.Series(dtype=int)
        out["matched_dr1_component_ids"] = pd.Series(dtype=str)
        out["matched_dr1_source_ids"] = pd.Series(dtype=str)
        out.attrs["dr1_bbox_support_grid_size_deg"] = float(grid_size_deg)
        out.attrs["dr1_bbox_support_include_edges"] = bool(include_edges)
        return out
    valid_mask = dr1_components["valid_position"].astype(bool) if "valid_position" in dr1_components else pd.Series(True, index=dr1_components.index)
    valid = dr1_components.loc[valid_mask].copy()
    if valid.empty:
        raise ValueError("DR1 component table has no valid positions")
    if "dr1_component_id" not in valid:
        valid["dr1_component_id"] = valid.index.astype(str)
    if "dr1_source_id" not in valid:
        valid["dr1_source_id"] = valid["dr1_component_id"].astype(str)

    cell_index = _build_component_cell_index(valid, grid_size_deg)
    ra = valid["ra"].to_numpy(dtype=float)
    dec = valid["dec"].to_numpy(dtype=float)
    component_ids = valid["dr1_component_id"].astype(str).to_numpy()
    source_ids = valid["dr1_source_id"].astype(str).to_numpy()
    index_to_pos = {idx: pos for pos, idx in enumerate(valid.index)}

    rows: list[dict[str, Any]] = []
    for row in preds.itertuples(index=False):
        item = row._asdict()
        ra_min = float(item["bbox_ra_min"])
        ra_max = float(item["bbox_ra_max"])
        dec_min = float(item["bbox_dec_min"])
        dec_max = float(item["bbox_dec_max"])
        candidate_indices: set[int] = set()
        for cell in _cell_ranges(ra_min, ra_max, dec_min, dec_max, grid_size_deg):
            candidate_indices.update(cell_index.get(cell, []))
        candidate_positions = np.array([index_to_pos[idx] for idx in candidate_indices], dtype=int)
        if candidate_positions.size:
            mask = bbox_contains(ra[candidate_positions], dec[candidate_positions], ra_min, ra_max, dec_min, dec_max, include_edges=include_edges)
            matched_pos = candidate_positions[mask]
        else:
            matched_pos = np.array([], dtype=int)
        matched_component_ids = sorted(set(component_ids[matched_pos].tolist()))
        matched_source_ids = sorted(set(source_ids[matched_pos].tolist()))
        item["supported_by_dr1"] = bool(len(matched_component_ids) > 0)
        item["n_matched_dr1_components"] = int(len(matched_component_ids))
        item["n_matched_dr1_sources"] = int(len(matched_source_ids))
        item["matched_dr1_component_ids"] = ";".join(matched_component_ids)
        item["matched_dr1_source_ids"] = ";".join(matched_source_ids)
        if prediction_id_col not in item:
            item[prediction_id_col] = str(len(rows))
        rows.append(item)
    out = pd.DataFrame(rows)
    out.attrs["dr1_bbox_support_grid_size_deg"] = float(grid_size_deg)
    out.attrs["dr1_bbox_support_include_edges"] = bool(include_edges)
    return out


def random_shift_control(
    predictions: pd.DataFrame,
    dr1_components: pd.DataFrame,
    *,
    ra_shifts_deg: tuple[float, ...] = (1.0, 2.0, 5.0),
    grid_size_deg: float = 0.25,
) -> pd.DataFrame:
    """Estimate chance containment by shifting DR1 RA positions by fixed offsets."""

    rows = []
    for shift in ra_shifts_deg:
        shifted = dr1_components.copy()
        shifted["ra"] = (pd.to_numeric(shifted["ra"], errors="coerce") + float(shift)) % 360.0
        matched = match_predictions_to_dr1_components(predictions, shifted, grid_size_deg=grid_size_deg)
        rows.append({"ra_shift_deg": float(shift), "random_support_fraction": float(matched["supported_by_dr1"].mean()) if len(matched) else 0.0})
    return pd.DataFrame(rows)
