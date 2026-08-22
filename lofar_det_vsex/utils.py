"""Shared utilities for the LoTSS Association pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LOGGER_NAME = "lofar_det_vsex"


def setup_logging(debug: bool = False, log_path: str | Path | None = None) -> logging.Logger:
    """Configure package logging."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream_handler)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the package logger."""

    return logging.getLogger(LOGGER_NAME)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_dumps_safe(value: Any) -> str:
    """Serialize simple debug payloads, converting numpy scalars and arrays."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    return json.dumps(value, default=default, sort_keys=True)


def first_existing_key(mapping: Mapping[str, Any], candidates: list[str]) -> str | None:
    """Return the first candidate key present in a mapping, case-insensitively."""

    lower_to_key = {str(key).lower(): str(key) for key in mapping.keys()}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key
    return None


def robust_mad_rms(image: np.ndarray) -> float:
    """Estimate robust rms using MAD."""

    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return float("nan")
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(finite))
    return float(rms)


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Convert a value to float without raising."""

    try:
        out = float(value)
    except Exception:
        return default
    if math.isfinite(out):
        return out
    return default


def safe_int(value: Any, default: int = -1) -> int:
    """Convert a value to int without raising."""

    try:
        return int(value)
    except Exception:
        return default


def decode_if_bytes(value: Any) -> Any:
    """Decode bytes from H5 attributes/datasets."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def normalize_id(value: Any) -> str:
    """Return a stable string identifier."""

    value = decode_if_bytes(value)
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def path_for_import(project_root: str | Path) -> None:
    """Ensure scripts can import the local package when run from the project root."""

    import sys

    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def write_dataframe(df: Any, path: str | Path) -> None:
    """Write a pandas DataFrame, using parquet when available and CSV as fallback."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception as exc:
            try:
                safe_df = parquet_safe_dataframe(df)
                safe_df.to_parquet(path, index=False)
                get_logger().warning("Wrote parquet with string-normalized object columns %s after: %s", path, exc)
                return
            except Exception as safe_exc:
                get_logger().warning("Could not write parquet %s: %s", path, safe_exc)
                csv_path = path.with_suffix(".csv")
                df.to_csv(csv_path, index=False)
                return
    df.to_csv(path, index=False)


def parquet_safe_dataframe(df: Any) -> Any:
    """Return a copy whose mixed object columns are stable for pyarrow parquet."""

    try:
        import pandas as pd
    except Exception:
        return df
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.dtype != object:
            continue
        values = [value for value in series.dropna().head(1000).tolist()]
        if not values:
            continue
        types = {_parquet_type_key(value) for value in values}
        has_nested = any(isinstance(value, (dict, list, tuple, set, np.ndarray)) for value in values)
        if len(types) > 1 or has_nested:
            out[col] = series.map(_string_for_parquet_object)
    return out


def _parquet_type_key(value: Any) -> type:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return str
    return type(value)


def _string_for_parquet_object(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    value = decode_if_bytes(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return json_dumps_safe(value)
    return str(value)


def angular_separation_arcsec(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra2_deg: np.ndarray | float,
    dec2_deg: np.ndarray | float,
) -> np.ndarray | float:
    """Small-angle-safe angular separation in arcsec."""

    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c) * 3600.0


def infer_pixel_scale_arcsec(wcs: Any | None, default: float = 1.5) -> float:
    """Infer approximate pixel scale in arcsec from an astropy WCS."""

    if wcs is None:
        return default
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales_deg = proj_plane_pixel_scales(wcs.celestial)
        scale_arcsec = float(np.mean(np.abs(scales_deg)) * 3600.0)
        if np.isfinite(scale_arcsec) and scale_arcsec > 0:
            return scale_arcsec
    except Exception:
        return default
    return default


def env_int(name: str, default: int) -> int:
    """Read integer environment variables for small operational switches."""

    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default
