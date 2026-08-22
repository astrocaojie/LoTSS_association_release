"""H5 inspection and cutout reading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import warnings

import h5py
import numpy as np

from .utils import decode_if_bytes, get_logger, normalize_id


IMAGE_CANDIDATES = ["image", "data", "cutout", "radio", "img", "map"]
RMS_CANDIDATES = ["rms", "rms_map", "noise"]
MEAN_CANDIDATES = ["mean", "mean_map", "background"]
ID_CANDIDATES = ["source_id", "object_id", "cutout_id", "id", "name"]
RA_CANDIDATES = ["ra", "RA", "center_ra", "ra_deg", "ra_center_deg"]
DEC_CANDIDATES = ["dec", "DEC", "center_dec", "dec_deg", "dec_center_deg"]
WCS_CANDIDATES = ["header", "wcs", "fits_header", "cutout_header", "cutout_wcs_header"]


@dataclass
class H5Keys:
    """Detected H5 dataset names for image, noise, position, and WCS fields."""

    # H5 文件结构在不同数据准备流程中可能不一致；这里统一保存自动识别出的关键数据集名称。
    image_key: str | None = None
    rms_key: str | None = None
    mean_key: str | None = None
    id_key: str | None = None
    ra_key: str | None = None
    dec_key: str | None = None
    wcs_key: str | None = None


@dataclass
class Cutout:
    """One radio image cutout and its optional noise, sky-position, and WCS metadata."""

    # 下游 association 只依赖这个轻量 Cutout 对象，不直接依赖某一种 H5/FITS 存储布局。
    cutout_id: str
    image: np.ndarray
    rms: np.ndarray | float | None = None
    mean: np.ndarray | float | None = None
    ra: float | None = None
    dec: float | None = None
    header: Any | None = None
    wcs: Any | None = None
    index: int | None = None
    metadata: dict[str, Any] | None = None


def _dataset_summary(name: str, obj: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    attrs = {key: decode_if_bytes(value) for key, value in obj.attrs.items()}
    if isinstance(obj, h5py.Dataset):
        return {
            "path": name,
            "type": "dataset",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "attrs": attrs,
        }
    return {
        "path": name,
        "type": "group",
        "attrs": attrs,
    }


def inspect_h5_structure(
    h5_path: str | Path,
    max_attrs: int = 20,
    max_objects: int | None = None,
) -> list[dict[str, Any]]:
    """Return a recursive summary of an H5 file."""

    summaries: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        root_attrs = {key: decode_if_bytes(value) for key, value in handle.attrs.items()}
        summaries.append({"path": "/", "type": "group", "attrs": root_attrs})

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if max_objects is not None and len(summaries) >= max_objects + 1:
                return
            item = _dataset_summary(name, obj)
            if len(item.get("attrs", {})) > max_attrs:
                attrs = dict(list(item["attrs"].items())[:max_attrs])
                attrs["..."] = f"{len(item['attrs']) - max_attrs} more attrs"
                item["attrs"] = attrs
            summaries.append(item)

        handle.visititems(visitor)
    return summaries


def print_h5_structure(h5_path: str | Path, max_attrs: int = 20, max_objects: int | None = None) -> None:
    """Print groups, datasets, shapes, dtypes, and attributes."""

    for item in inspect_h5_structure(h5_path, max_attrs=max_attrs, max_objects=max_objects):
        indent = "  " * (0 if item["path"] == "/" else item["path"].count("/"))
        if item["type"] == "dataset":
            print(
                f"{indent}- {item['path']} [dataset] "
                f"shape={item['shape']} dtype={item['dtype']}"
            )
        else:
            print(f"{indent}- {item['path']} [group]")
        attrs = item.get("attrs", {})
        if attrs:
            print(f"{indent}  attrs={attrs}")
    if max_objects is not None:
        print(f"... output limited to first {max_objects} H5 objects")


def summarize_h5_file(h5_path: str | Path, config_h5: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a concise H5 summary useful for reports."""

    keys = detect_h5_keys(h5_path, config_h5=config_h5)
    layout = detect_h5_layout(h5_path, keys.image_key)
    reader = H5CutoutReader(h5_path, config_h5=config_h5)
    with h5py.File(h5_path, "r") as handle:
        root_attrs = {key: decode_if_bytes(value) for key, value in handle.attrs.items()}
    cutout = reader.read(0)
    finite = np.isfinite(cutout.image)
    return {
        "path": str(h5_path),
        "layout": layout,
        "n_cutouts": len(reader),
        "image_key": cutout.metadata.get("image_key") if cutout.metadata else keys.image_key,
        "image_shape": tuple(cutout.image.shape),
        "image_dtype_after_read": str(cutout.image.dtype),
        "has_rms": cutout.rms is not None,
        "has_mean": cutout.mean is not None,
        "has_cutout_id": bool(cutout.cutout_id),
        "has_ra_dec": cutout.ra is not None and cutout.dec is not None,
        "has_wcs": cutout.wcs is not None,
        "ra": cutout.ra,
        "dec": cutout.dec,
        "pixel_scale_arcsec": root_attrs.get("pixel_scale_arcsec"),
        "beam_major_arcsec": root_attrs.get("beam_major_arcsec"),
        "beam_minor_arcsec": root_attrs.get("beam_minor_arcsec"),
        "beam_pa_deg": root_attrs.get("beam_pa_deg"),
        "nan_count_first_cutout": int(np.isnan(cutout.image).sum()),
        "inf_count_first_cutout": int(np.isinf(cutout.image).sum()),
        "finite_fraction_first_cutout": float(finite.mean()),
    }


def _all_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    handle.visititems(visitor)
    return datasets


def _candidate_cutout_groups(handle: h5py.File) -> list[str]:
    """Return groups that look like one-cutout containers."""

    groups: list[str] = []
    for name, obj in handle.items():
        if not isinstance(obj, h5py.Group):
            continue
        datasets = [value for value in obj.values() if isinstance(value, h5py.Dataset)]
        has_image = any(
            dataset.ndim == 2
            and np.issubdtype(dataset.dtype, np.number)
            and _score_dataset_name(dataset.name.rsplit("/", 1)[-1], IMAGE_CANDIDATES) > 0
            for dataset in datasets
        )
        if has_image:
            groups.append(name)
    groups.sort()
    return groups


def detect_h5_layout(h5_path: str | Path, image_key: str | None = None) -> str:
    """Detect whether images are stored as an array dataset or group-per-cutout."""

    # 自动判断 H5 是 N x H x W 数组还是每个 cutout 一个 group，供 reader 选择读取策略。
    with h5py.File(h5_path, "r") as handle:
        if image_key and image_key in handle:
            obj = handle[image_key]
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
                return "array"
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and "/" in image_key:
                return "group_per_cutout"
        if _candidate_cutout_groups(handle):
            return "group_per_cutout"
    return "array"


def _score_dataset_name(path: str, candidates: Iterable[str]) -> int:
    base = path.rsplit("/", 1)[-1].lower()
    parts = [part.lower() for part in path.split("/")]
    score = 0
    for idx, candidate in enumerate(candidates):
        cand = candidate.lower()
        if base == cand:
            score = max(score, 100 - idx)
        elif cand in parts:
            score = max(score, 80 - idx)
        elif cand in base:
            score = max(score, 50 - idx)
    return score


def _choose_dataset(
    datasets: dict[str, h5py.Dataset],
    candidates: Iterable[str],
    require_numeric: bool = False,
    min_ndim: int | None = None,
) -> str | None:
    scored: list[tuple[int, str]] = []
    for path, dataset in datasets.items():
        if require_numeric and not np.issubdtype(dataset.dtype, np.number):
            continue
        if min_ndim is not None and len(dataset.shape) < min_ndim:
            continue
        score = _score_dataset_name(path, candidates)
        if score > 0:
            scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return scored[0][1]


def detect_h5_keys(h5_path: str | Path, config_h5: dict[str, Any] | None = None) -> H5Keys:
    """Auto-detect likely H5 dataset keys, overridden by config values."""

    config_h5 = config_h5 or {}
    with h5py.File(h5_path, "r") as handle:
        datasets = _all_datasets(handle)
        cutout_groups = _candidate_cutout_groups(handle)
        if cutout_groups:
            first_group = cutout_groups[0]
            group_datasets = {
                path: dataset for path, dataset in datasets.items() if path.startswith(f"{first_group}/")
            }
            keys = H5Keys(
                image_key=_choose_dataset(group_datasets, IMAGE_CANDIDATES, require_numeric=True, min_ndim=2),
                rms_key=_choose_dataset(group_datasets, RMS_CANDIDATES, require_numeric=True),
                mean_key=_choose_dataset(group_datasets, MEAN_CANDIDATES, require_numeric=True),
                id_key=None,
                ra_key=None,
                dec_key=None,
                wcs_key=None,
            )
            attrs = handle[first_group].attrs
            attr_names = {name.lower(): name for name in attrs.keys()}
            for candidate in ID_CANDIDATES:
                if candidate.lower() in attr_names:
                    keys.id_key = f"{first_group}@{attr_names[candidate.lower()]}"
                    break
            for candidate in RA_CANDIDATES:
                if candidate.lower() in attr_names:
                    keys.ra_key = f"{first_group}@{attr_names[candidate.lower()]}"
                    break
            for candidate in DEC_CANDIDATES:
                if candidate.lower() in attr_names:
                    keys.dec_key = f"{first_group}@{attr_names[candidate.lower()]}"
                    break
            for candidate in WCS_CANDIDATES:
                if candidate.lower() in attr_names:
                    keys.wcs_key = f"{first_group}@{attr_names[candidate.lower()]}"
                    break
        else:
            keys = H5Keys(
                image_key=_choose_dataset(datasets, IMAGE_CANDIDATES, require_numeric=True, min_ndim=2),
                rms_key=_choose_dataset(datasets, RMS_CANDIDATES, require_numeric=True),
                mean_key=_choose_dataset(datasets, MEAN_CANDIDATES, require_numeric=True),
                id_key=_choose_dataset(datasets, ID_CANDIDATES),
                ra_key=_choose_dataset(datasets, RA_CANDIDATES, require_numeric=True),
                dec_key=_choose_dataset(datasets, DEC_CANDIDATES, require_numeric=True),
                wcs_key=_choose_dataset(datasets, WCS_CANDIDATES),
            )

    for field_name in keys.__dataclass_fields__:
        explicit = config_h5.get(field_name)
        if explicit:
            setattr(keys, field_name, explicit)

    return keys


def _read_indexed_dataset(handle: h5py.File, key: str | None, index: int) -> Any | None:
    if not key:
        return None
    if key not in handle:
        warnings.warn(f"H5 key '{key}' not found")
        return None
    data = handle[key]
    if not isinstance(data, h5py.Dataset):
        return None
    if len(data.shape) == 0:
        value = data[()]
    elif len(data.shape) >= 3:
        value = data[index]
    elif len(data.shape) == 2:
        if np.issubdtype(data.dtype, np.number) and min(data.shape) > 8:
            value = data[()]
        else:
            value = data[index]
    elif len(data.shape) == 1:
        value = data[index]
    else:
        value = data[()]
    return decode_if_bytes(value)


def _read_attr(obj: h5py.Group | h5py.Dataset, key: str | None) -> Any | None:
    if not key:
        return None
    if key not in obj.attrs:
        return None
    return decode_if_bytes(obj.attrs[key])


def _read_image_dataset(handle: h5py.File, key: str, index: int) -> np.ndarray:
    data = handle[key]
    if len(data.shape) == 2:
        image = data[()]
    elif len(data.shape) == 3:
        image = data[index]
    elif len(data.shape) == 4 and data.shape[1] == 1:
        image = data[index, 0]
    elif len(data.shape) == 4 and data.shape[-1] == 1:
        image = data[index, :, :, 0]
    else:
        raise ValueError(f"Unsupported image dataset shape for {key}: {data.shape}")
    return np.asarray(image, dtype=float)


def _dataset_length(handle: h5py.File, key: str) -> int:
    data = handle[key]
    if len(data.shape) <= 2:
        return 1
    return int(data.shape[0])


def _parse_header(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        from astropy.io import fits

        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.dtype.kind in {"S", "U", "O"}:
                value = "\n".join(str(decode_if_bytes(v)) for v in value.ravel())
        value = decode_if_bytes(value)
        if isinstance(value, str):
            return fits.Header.fromstring(value, sep="\n")
        if isinstance(value, fits.Header):
            return value
    except Exception as exc:
        get_logger().debug("Could not parse WCS/header metadata: %s", exc)
    return None


def wcs_from_header_value(value: Any) -> Any | None:
    """Build an astropy WCS from a header-like H5 value."""

    header = _parse_header(value)
    if header is None:
        return None
    try:
        from astropy.wcs import WCS

        return WCS(header)
    except Exception as exc:
        get_logger().debug("Could not build WCS: %s", exc)
        return None


class H5CutoutReader:
    """Read cutout images and metadata from an H5 file."""

    def __init__(self, h5_path: str | Path, config_h5: dict[str, Any] | None = None):
        # reader 持有打开的 H5 handle，避免批量处理时反复打开大文件。
        self.h5_path = Path(h5_path)
        self.keys = detect_h5_keys(self.h5_path, config_h5=config_h5)
        if not self.keys.image_key:
            raise ValueError(
                "Could not auto-detect image dataset. Set h5.image_key in the config."
            )
        self.layout = detect_h5_layout(self.h5_path, self.keys.image_key)
        self._cutout_groups: list[str] | None = None
        self._length: int | None = None

    def _groups(self) -> list[str]:
        if self._cutout_groups is None:
            with h5py.File(self.h5_path, "r") as handle:
                self._cutout_groups = _candidate_cutout_groups(handle)
        return self._cutout_groups

    def __len__(self) -> int:
        if self._length is None:
            with h5py.File(self.h5_path, "r") as handle:
                if self.layout == "group_per_cutout":
                    self._length = len(_candidate_cutout_groups(handle))
                else:
                    self._length = _dataset_length(handle, self.keys.image_key or "")
        return self._length

    def iter_indices(
        self,
        start_index: int = 0,
        end_index: int | None = None,
        limit: int | None = None,
    ) -> list[int]:
        total = len(self)
        end = total if end_index is None else min(end_index, total)
        indices = list(range(max(0, start_index), end))
        if limit is not None:
            indices = indices[:limit]
        return indices

    def read(self, index: int) -> Cutout:
        if self.layout == "group_per_cutout":
            return self._read_group_cutout(index)

        with h5py.File(self.h5_path, "r") as handle:
            image = _read_image_dataset(handle, self.keys.image_key or "", index)
            rms = _read_indexed_dataset(handle, self.keys.rms_key, index)
            mean = _read_indexed_dataset(handle, self.keys.mean_key, index)
            raw_id = _read_indexed_dataset(handle, self.keys.id_key, index)
            ra = _read_indexed_dataset(handle, self.keys.ra_key, index)
            dec = _read_indexed_dataset(handle, self.keys.dec_key, index)
            header_value = _read_indexed_dataset(handle, self.keys.wcs_key, index)

        cutout_id = normalize_id(raw_id) if raw_id is not None else f"cutout_{index:06d}"
        header = _parse_header(header_value)
        wcs = wcs_from_header_value(header_value)

        metadata = {
            "h5_path": str(self.h5_path),
            "image_key": self.keys.image_key,
            "rms_key": self.keys.rms_key,
            "mean_key": self.keys.mean_key,
            "id_key": self.keys.id_key,
            "ra_key": self.keys.ra_key,
            "dec_key": self.keys.dec_key,
            "wcs_key": self.keys.wcs_key,
        }

        return Cutout(
            cutout_id=cutout_id,
            image=image,
            rms=np.asarray(rms) if rms is not None and np.ndim(rms) > 0 else rms,
            mean=np.asarray(mean) if mean is not None and np.ndim(mean) > 0 else mean,
            ra=float(ra) if ra is not None and np.ndim(ra) == 0 else None,
            dec=float(dec) if dec is not None and np.ndim(dec) == 0 else None,
            header=header,
            wcs=wcs,
            index=index,
            metadata=metadata,
        )

    def _read_group_cutout(self, index: int) -> Cutout:
        groups = self._groups()
        if index < 0 or index >= len(groups):
            raise IndexError(f"Cutout index {index} out of range for {len(groups)} groups")
        group_name = groups[index]

        with h5py.File(self.h5_path, "r") as handle:
            group = handle[group_name]
            image_name = (self.keys.image_key or "").rsplit("/", 1)[-1]
            if image_name not in group:
                image_name = _choose_dataset(
                    {name: obj for name, obj in group.items() if isinstance(obj, h5py.Dataset)},
                    IMAGE_CANDIDATES,
                    require_numeric=True,
                    min_ndim=2,
                )
            if not image_name:
                raise ValueError(f"No image dataset found inside group {group_name}")
            image = np.asarray(group[image_name][()], dtype=float)

            rms_name = (self.keys.rms_key or "").rsplit("/", 1)[-1] if self.keys.rms_key else None
            mean_name = (self.keys.mean_key or "").rsplit("/", 1)[-1] if self.keys.mean_key else None
            rms = np.asarray(group[rms_name][()]) if rms_name and rms_name in group else None
            mean = np.asarray(group[mean_name][()]) if mean_name and mean_name in group else None

            raw_id = _read_attr(group, "cutout_id") or group_name
            ra = _read_attr(group, "ra_center_deg")
            dec = _read_attr(group, "dec_center_deg")
            header_value = _read_attr(group, "cutout_wcs_header") or _read_attr(group, "cutout_header")
            attrs = {key: decode_if_bytes(value) for key, value in group.attrs.items()}
            root_attrs = {key: decode_if_bytes(value) for key, value in handle.attrs.items()}

        header = _parse_header(header_value)
        wcs = wcs_from_header_value(header_value)
        metadata = {
            "h5_path": str(self.h5_path),
            "layout": self.layout,
            "group_name": group_name,
            "image_key": f"{group_name}/{image_name}",
            "rms_key": f"{group_name}/{rms_name}" if rms_name else None,
            "mean_key": f"{group_name}/{mean_name}" if mean_name else None,
            "id_key": "group_name",
            "ra_key": "ra_center_deg",
            "dec_key": "dec_center_deg",
            "wcs_key": "cutout_wcs_header",
            "attrs": attrs,
            "root_attrs": root_attrs,
            "pixel_scale_arcsec": root_attrs.get("pixel_scale_arcsec"),
            "beam_major_arcsec": root_attrs.get("beam_major_arcsec"),
            "beam_minor_arcsec": root_attrs.get("beam_minor_arcsec"),
            "beam_pa_deg": root_attrs.get("beam_pa_deg"),
        }

        return Cutout(
            cutout_id=normalize_id(raw_id),
            image=image,
            rms=rms,
            mean=mean,
            ra=float(ra) if ra is not None else None,
            dec=float(dec) if dec is not None else None,
            header=header,
            wcs=wcs,
            index=index,
            metadata=metadata,
        )
