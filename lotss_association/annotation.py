"""Helpers for group-level radio association annotation.

The annotation tools intentionally stay independent from the association
pipeline. They only read existing catalogs/figures and append human labels.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


MANUAL_ASSOCIATION_LABELS = [
    "correct",
    "overmerge",
    "undermerge",
    "partial",
    "artifact",
    "uncertain",
]

MANUAL_PARENT_STATUS = [
    "local_complete",
    "part_of_larger_source",
    "complete_parent_source",
    "unknown",
]

MANUAL_ASSOCIATION_TYPES = [
    "compact_multi_gaussian",
    "continuous_extended",
    "diffuse_extended",
    "linear_or_tail_like",
    "complex_association",
    "separated_lobe_or_component",
    "artifact_risk",
    "uncertain",
]

MANUAL_QUALITY = [
    "high",
    "medium",
    "low",
    "suspicious",
    "artifact",
    "uncertain",
]

MANUAL_EVIDENCE_FLAGS = [
    "same_bright_region",
    "same_contour",
    "visible_bridge",
    "ridge_continuity",
    "pa_aligned",
    "smooth_flux_transition",
    "shared_tail_or_plume",
    "separated_but_symmetric",
    "possible_core_between",
]

MANUAL_PROBLEM_FLAGS = [
    "noise_bridge",
    "only_low_snr_connection",
    "sidelobe_risk",
    "negative_bowl",
    "crowded_region",
    "unrelated_point_source",
    "bad_gaussian_ellipse",
    "mask_too_large",
    "missing_counter_lobe",
    "missing_tail",
    "unclear_image",
]

MANIFEST_FIELDS = [
    "item_id",
    "image_path",
    "overview_image_path",
    "cutout_id",
    "association_group_id",
    "n_gaussians",
    "gaussian_ids",
    "LAS_arcsec",
    "LAS_beam",
    "association_quality",
    "association_type",
    "association_score_mean",
    "association_score_min",
    "association_score_max",
    "n_strong_edges",
    "n_weak_edges",
    "n_only_2sigma_edges",
    "artifact_risk_flags",
    "debug_info",
    "annotated",
    "annotation_status",
]

ANNOTATION_FIELDS = [
    "timestamp",
    "annotator",
    "item_id",
    "image_path",
    "cutout_id",
    "association_group_id",
    "n_gaussians",
    "gaussian_ids",
    "LAS_arcsec",
    "LAS_beam",
    "association_quality_model",
    "association_type_model",
    "association_score_mean",
    "n_strong_edges",
    "n_weak_edges",
    "n_only_2sigma_edges",
    "manual_association_label",
    "manual_parent_status",
    "manual_association_type",
    "manual_quality",
    "manual_evidence_flags",
    "manual_problem_flags",
    "remove_gaussian_ids",
    "missing_gaussian_ids",
    "missing_group_ids",
    "comment",
    "annotation_status",
]

EXPORT_EXTRA_FIELDS = [
    "association_score_min",
    "association_score_max",
    "artifact_risk_flags",
    "debug_info",
]

PRIORITY_QUALITIES = {"high": 4, "medium": 3, "suspicious": 2, "artifact_risk": 1}
PRIORITY_TYPES = {
    "complex_association": 5,
    "continuous_extended": 4,
    "linear_or_tail_like": 3,
    "diffuse_extended": 2,
    "compact_multi_gaussian": 1,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name, "")) for name in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def parse_list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v)]
        except json.JSONDecodeError:
            pass
    sep = ";" if ";" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_zoom_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    match = re.match(r"^(cutout_\d+)_(.+)$", stem)
    if match:
        return match.group(1), stem
    cutout_match = re.match(r"^(cutout_\d+)", stem)
    return (cutout_match.group(1) if cutout_match else "", stem)


def manifest_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_group_catalog(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_dicts(path)
    by_group: Dict[str, Dict[str, str]] = {}
    for row in rows:
        group_id = row.get("association_group_id", "")
        if group_id:
            by_group[group_id] = row
    return by_group


def make_manifest_row(
    image_path: Path,
    overview_dir: Optional[Path],
    catalog_by_group: Mapping[str, Mapping[str, str]],
    base_dir: Path,
) -> Tuple[Dict[str, str], Optional[str]]:
    cutout_id_from_name, group_id_from_name = parse_zoom_filename(image_path)
    catalog_row = catalog_by_group.get(group_id_from_name, {})
    cutout_id = str(catalog_row.get("cutout_id", cutout_id_from_name))
    group_id = str(catalog_row.get("association_group_id", group_id_from_name))

    overview_path = ""
    if overview_dir and cutout_id:
        candidate = overview_dir / f"{cutout_id}.png"
        if candidate.exists():
            overview_path = manifest_path(candidate, base_dir)

    row = {
        "item_id": group_id or group_id_from_name,
        "image_path": manifest_path(image_path, base_dir),
        "overview_image_path": overview_path,
        "cutout_id": cutout_id,
        "association_group_id": group_id,
        "n_gaussians": str(catalog_row.get("n_gaussians", "")),
        "gaussian_ids": str(catalog_row.get("gaussian_ids", "")),
        "LAS_arcsec": str(catalog_row.get("LAS_arcsec", "")),
        "LAS_beam": str(catalog_row.get("LAS_beam", "")),
        "association_quality": str(catalog_row.get("association_quality", "")),
        "association_type": str(catalog_row.get("association_type", "")),
        "association_score_mean": str(catalog_row.get("association_score_mean", "")),
        "association_score_min": str(catalog_row.get("association_score_min", "")),
        "association_score_max": str(catalog_row.get("association_score_max", "")),
        "n_strong_edges": str(catalog_row.get("n_strong_edges", "")),
        "n_weak_edges": str(catalog_row.get("n_weak_edges", "")),
        "n_only_2sigma_edges": str(catalog_row.get("n_only_2sigma_edges", "")),
        "artifact_risk_flags": str(catalog_row.get("artifact_risk_flags", "")),
        "debug_info": str(catalog_row.get("debug_info", "")),
        "annotated": "false",
        "annotation_status": "unannotated",
    }
    warning = None if catalog_row else f"warning: no catalog match for {image_path.name}"
    return row, warning


def filter_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    qualities: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    min_n_gaussians: Optional[int] = None,
    min_las_beam: Optional[float] = None,
    suspicious_only: bool = False,
    unannotated_only: bool = False,
    search: str = "",
    review_mode: bool = False,
) -> List[Dict[str, Any]]:
    quality_set = {q for q in (qualities or []) if q}
    type_set = {t for t in (types or []) if t}
    search_text = search.strip().lower()
    filtered: List[Dict[str, Any]] = []
    for input_row in rows:
        row = dict(input_row)
        if quality_set and row.get("association_quality", "") not in quality_set:
            continue
        if type_set and row.get("association_type", "") not in type_set:
            continue
        if min_n_gaussians is not None and safe_int(row.get("n_gaussians")) < min_n_gaussians:
            continue
        if min_las_beam is not None and safe_float(row.get("LAS_beam")) < min_las_beam:
            continue
        if suspicious_only and not is_suspicious(row):
            continue
        if unannotated_only and not review_mode and row.get("annotation_status", "unannotated") != "unannotated":
            continue
        if search_text:
            haystack = " ".join(
                str(row.get(key, ""))
                for key in ("item_id", "cutout_id", "association_group_id", "gaussian_ids")
            ).lower()
            if search_text not in haystack:
                continue
        filtered.append(row)
    return filtered


def is_suspicious(row: Mapping[str, Any]) -> bool:
    quality = str(row.get("association_quality", "")).lower()
    flags = str(row.get("artifact_risk_flags", "")).strip()
    status = str(row.get("annotation_status", "")).lower()
    return quality in {"suspicious", "artifact_risk"} or bool(flags) or status == "skipped"


def sort_manifest_rows(rows: Sequence[Mapping[str, Any]], queue_mode: str = "priority", seed: int = 0) -> List[Dict[str, Any]]:
    output = [dict(row) for row in rows]
    if queue_mode == "all":
        return sorted(output, key=lambda row: str(row.get("item_id", "")))
    if queue_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(output)
        return output
    if queue_mode == "suspicious":
        return sorted(output, key=lambda row: (not is_suspicious(row), -safe_float(row.get("LAS_beam")), str(row.get("item_id", ""))))
    if queue_mode == "high":
        return sorted(
            output,
            key=lambda row: (
                row.get("association_quality") != "high",
                -safe_float(row.get("LAS_beam")),
                -safe_int(row.get("n_gaussians")),
                str(row.get("item_id", "")),
            ),
        )
    return sorted(output, key=priority_sort_key)


def priority_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    quality = str(row.get("association_quality", ""))
    assoc_type = str(row.get("association_type", ""))
    return (
        -PRIORITY_QUALITIES.get(quality, 0),
        safe_int(row.get("n_gaussians")) < 2,
        -safe_float(row.get("LAS_beam")),
        -PRIORITY_TYPES.get(assoc_type, 0),
        -safe_int(row.get("n_gaussians")),
        str(row.get("item_id", "")),
    )


def build_manifest(
    catalog_path: Path,
    zoom_dir: Path,
    overview_dir: Optional[Path],
    output_path: Path,
    base_dir: Path,
    qualities: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    min_n_gaussians: Optional[int] = None,
    min_las_beam: Optional[float] = None,
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 0,
    queue_mode: str = "all",
) -> Tuple[List[Dict[str, str]], List[str]]:
    catalog = load_group_catalog(catalog_path)
    warnings: List[str] = []
    rows: List[Dict[str, str]] = []
    for image_path in sorted(zoom_dir.glob("*.png")):
        row, warning = make_manifest_row(image_path, overview_dir, catalog, base_dir)
        if warning:
            warnings.append(warning)
        rows.append(row)

    rows = filter_manifest_rows(
        rows,
        qualities=qualities,
        types=types,
        min_n_gaussians=min_n_gaussians,
        min_las_beam=min_las_beam,
    )
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
    else:
        rows = sort_manifest_rows(rows, queue_mode=queue_mode, seed=seed)
    if limit is not None:
        rows = rows[:limit]
    write_csv_dicts(output_path, rows, MANIFEST_FIELDS)
    return rows, warnings


def read_annotations_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                if isinstance(record, dict):
                    record["_line_number"] = line_number
                    records.append(record)
            except json.JSONDecodeError:
                records.append({"_line_number": line_number, "_invalid_json": text})
    return records


def deduplicate_annotations(records: Sequence[Mapping[str, Any]], mode: str = "last") -> List[Dict[str, Any]]:
    if mode == "none":
        return [dict(record) for record in records if record.get("item_id")]
    by_item: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if mode == "first":
        for record in records:
            item_id = str(record.get("item_id", ""))
            if item_id and item_id not in by_item:
                by_item[item_id] = dict(record)
        return list(by_item.values())
    if mode != "last":
        raise ValueError(f"Unsupported deduplicate mode: {mode}")
    for record in records:
        item_id = str(record.get("item_id", ""))
        if item_id:
            by_item[item_id] = dict(record)
    return list(by_item.values())


def latest_annotations_by_item(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record["item_id"]): dict(record) for record in deduplicate_annotations(records, "last")}


def enrich_manifest_with_annotations(
    manifest_rows: Sequence[Mapping[str, Any]],
    annotations_by_item: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for input_row in manifest_rows:
        row = dict(input_row)
        item_id = str(row.get("item_id", ""))
        record = annotations_by_item.get(item_id, {})
        status = str(record.get("annotation_status", "") or row.get("annotation_status", "unannotated"))
        has_label = bool(str(record.get("manual_association_label", "")).strip())
        if has_label:
            status = "annotated"
        elif status not in {"skipped", "annotated"}:
            status = "unannotated"
        row["annotated"] = "true" if has_label else "false"
        row["annotation_status"] = status
        row["latest_manual_association_label"] = str(record.get("manual_association_label", ""))
        row["latest_manual_quality"] = str(record.get("manual_quality", ""))
        row["latest_manual_parent_status"] = str(record.get("manual_parent_status", ""))
        row["latest_comment"] = str(record.get("comment", ""))
        row["latest_annotator"] = str(record.get("annotator", ""))
        row["latest_timestamp"] = str(record.get("timestamp", ""))
        enriched.append(row)
    return enriched


def make_annotation_record(
    manifest_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    status: str = "annotated",
) -> Dict[str, Any]:
    label = str(payload.get("manual_association_label", "")).strip()
    if status == "annotated" and not label:
        status = "skipped"
    record = {
        "timestamp": str(payload.get("timestamp") or utc_now_iso()),
        "annotator": str(payload.get("annotator", "")).strip() or "anonymous",
        "item_id": str(manifest_row.get("item_id", "")),
        "image_path": str(manifest_row.get("image_path", "")),
        "cutout_id": str(manifest_row.get("cutout_id", "")),
        "association_group_id": str(manifest_row.get("association_group_id", "")),
        "n_gaussians": str(manifest_row.get("n_gaussians", "")),
        "gaussian_ids": str(manifest_row.get("gaussian_ids", "")),
        "LAS_arcsec": str(manifest_row.get("LAS_arcsec", "")),
        "LAS_beam": str(manifest_row.get("LAS_beam", "")),
        "association_quality_model": str(manifest_row.get("association_quality", "")),
        "association_type_model": str(manifest_row.get("association_type", "")),
        "association_score_mean": str(manifest_row.get("association_score_mean", "")),
        "n_strong_edges": str(manifest_row.get("n_strong_edges", "")),
        "n_weak_edges": str(manifest_row.get("n_weak_edges", "")),
        "n_only_2sigma_edges": str(manifest_row.get("n_only_2sigma_edges", "")),
        "manual_association_label": label,
        "manual_parent_status": str(payload.get("manual_parent_status", "")).strip(),
        "manual_association_type": str(payload.get("manual_association_type", "")).strip(),
        "manual_quality": str(payload.get("manual_quality", "")).strip(),
        "manual_evidence_flags": parse_list_value(payload.get("manual_evidence_flags")),
        "manual_problem_flags": parse_list_value(payload.get("manual_problem_flags")),
        "remove_gaussian_ids": str(payload.get("remove_gaussian_ids", "")).strip(),
        "missing_gaussian_ids": str(payload.get("missing_gaussian_ids", "")).strip(),
        "missing_group_ids": str(payload.get("missing_group_ids", "")).strip(),
        "comment": str(payload.get("comment", "")).strip(),
        "annotation_status": status,
    }
    return record


def append_annotation(path: Path, record: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True) + "\n")


def export_annotations(
    manifest_path_arg: Path,
    annotation_file: Path,
    output_path: Path,
    deduplicate: str = "last",
) -> List[Dict[str, Any]]:
    manifest_rows = read_csv_dicts(manifest_path_arg)
    records = [record for record in read_annotations_jsonl(annotation_file) if not record.get("_invalid_json")]
    output_rows: List[Dict[str, Any]]
    if deduplicate == "none":
        manifest_by_item = {row.get("item_id", ""): row for row in manifest_rows}
        output_rows = []
        for index, record in enumerate(deduplicate_annotations(records, "none"), start=1):
            item_id = str(record.get("item_id", ""))
            row = dict(manifest_by_item.get(item_id, {}))
            row.update(record)
            row["annotation_index"] = str(index)
            row["annotated"] = "true" if row.get("manual_association_label") else "false"
            output_rows.append(row)
    else:
        records_by_item = {str(record.get("item_id", "")): record for record in deduplicate_annotations(records, deduplicate)}
        output_rows = []
        for manifest_row in manifest_rows:
            row = dict(manifest_row)
            record = records_by_item.get(str(row.get("item_id", "")), {})
            row.update(record)
            has_label = bool(str(row.get("manual_association_label", "")).strip())
            if has_label:
                row["annotated"] = "true"
                row["annotation_status"] = "annotated"
            else:
                row["annotated"] = "false"
                row["annotation_status"] = str(row.get("annotation_status", "") or "unannotated")
            output_rows.append(row)

    fieldnames = build_export_fieldnames(output_rows, deduplicate)
    write_csv_dicts(output_path, output_rows, fieldnames)
    return output_rows


def build_export_fieldnames(rows: Sequence[Mapping[str, Any]], deduplicate: str = "last") -> List[str]:
    base = list(MANIFEST_FIELDS)
    fields = list(base)
    if deduplicate == "none":
        fields.append("annotation_index")
    for field in ANNOTATION_FIELDS:
        if field not in fields:
            fields.append(field)
    for field in EXPORT_EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fields:
                fields.append(key)
    return fields


def annotation_progress(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    annotated = sum(1 for row in rows if str(row.get("annotated", "")).lower() == "true")
    skipped = sum(1 for row in rows if str(row.get("annotation_status", "")) == "skipped")
    return {
        "total": total,
        "annotated": annotated,
        "skipped": skipped,
        "unannotated": max(total - annotated - skipped, 0),
        "annotated_fraction": annotated / total if total else 0.0,
    }


def compute_dashboard_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    annotated_rows = [row for row in rows if str(row.get("annotated", "")).lower() == "true"]
    return {
        "progress": annotation_progress(rows),
        "manual_association_label": Counter(str(row.get("latest_manual_association_label", "")) for row in annotated_rows if row.get("latest_manual_association_label")),
        "manual_quality": Counter(str(row.get("latest_manual_quality", "")) for row in annotated_rows if row.get("latest_manual_quality")),
        "association_quality_model": Counter(str(row.get("association_quality", "")) for row in rows if row.get("association_quality")),
        "association_type_model": Counter(str(row.get("association_type", "")) for row in rows if row.get("association_type")),
    }


def summarize_annotations_csv(input_path: Path, output_path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_dicts(input_path)
    summary_rows = compute_summary_rows(rows)
    fields = ["section", "group", "metric", "value", "count", "total", "rate"]
    write_csv_dicts(output_path, summary_rows, fields)
    return summary_rows


def compute_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    total_items = len(rows)
    annotated_rows = [row for row in rows if is_human_annotated(row)]
    n_annotated = len(annotated_rows)
    output: List[Dict[str, Any]] = []

    def add(section: str, group: str, metric: str, value: Any = "", count: Any = "", total: Any = "", rate: Any = "") -> None:
        output.append(
            {
                "section": section,
                "group": group,
                "metric": metric,
                "value": value,
                "count": count,
                "total": total,
                "rate": f"{rate:.6f}" if isinstance(rate, float) else rate,
            }
        )

    add("overall", "all", "total_items", count=total_items)
    add("overall", "all", "annotated_items", count=n_annotated, total=total_items, rate=(n_annotated / total_items if total_items else 0.0))
    add("overall", "all", "skipped_items", count=sum(1 for row in rows if row.get("annotation_status") == "skipped"), total=total_items)

    for field in ("manual_association_label", "manual_quality", "manual_association_type"):
        counts = Counter(str(row.get(field, "")) for row in annotated_rows if row.get(field))
        for value, count in sorted(counts.items()):
            add("count", field, "count", value=value, count=count, total=n_annotated, rate=(count / n_annotated if n_annotated else 0.0))

    global_rates = rate_bundle(annotated_rows)
    for metric in ("correct_rate", "usable_rate", "problem_rate", "overmerge_rate", "undermerge_rate", "artifact_rate", "uncertain_rate"):
        add("rate", "all", metric, count=global_rates.get(metric + "_count", ""), total=n_annotated, rate=global_rates[metric])

    for group_field, section in (
        ("association_quality_model", "model_quality"),
        ("association_type_model", "model_type"),
    ):
        grouped = group_rows(annotated_rows, group_field)
        for group, group_rows_list in sorted(grouped.items()):
            rates = rate_bundle(group_rows_list)
            add(section, group, "annotated_count", count=len(group_rows_list), total=n_annotated)
            for metric in ("correct_rate", "usable_rate", "problem_rate", "overmerge_rate", "undermerge_rate", "artifact_rate", "uncertain_rate"):
                add(section, group, metric, count=rates.get(metric + "_count", ""), total=len(group_rows_list), rate=rates[metric])

    for bin_name, bin_rows in sorted(group_rows_by_bin(annotated_rows, "n_gaussians", n_gaussians_bin).items()):
        rates = rate_bundle(bin_rows)
        add("n_gaussians_bin", bin_name, "annotated_count", count=len(bin_rows), total=n_annotated)
        for metric in ("correct_rate", "usable_rate", "problem_rate"):
            add("n_gaussians_bin", bin_name, metric, count=rates.get(metric + "_count", ""), total=len(bin_rows), rate=rates[metric])

    for bin_name, bin_rows in sorted(group_rows_by_bin(annotated_rows, "LAS_beam", las_beam_bin).items()):
        rates = rate_bundle(bin_rows)
        add("LAS_beam_bin", bin_name, "annotated_count", count=len(bin_rows), total=n_annotated)
        for metric in ("correct_rate", "usable_rate", "problem_rate"):
            add("LAS_beam_bin", bin_name, metric, count=rates.get(metric + "_count", ""), total=len(bin_rows), rate=rates[metric])

    return output


def is_human_annotated(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("manual_association_label", "")).strip())


def group_rows(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    fallback = {"association_quality_model": "association_quality", "association_type_model": "association_type"}.get(field)
    for row in rows:
        value = str(row.get(field, "") or (row.get(fallback, "") if fallback else "") or "missing")
        grouped[value].append(row)
    return grouped


def group_rows_by_bin(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    bin_func,
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[bin_func(safe_float(row.get(field)))].append(row)
    return grouped


def rate_bundle(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    counts = {
        "correct_rate_count": sum(1 for row in rows if row.get("manual_association_label") == "correct"),
        "usable_rate_count": sum(1 for row in rows if row.get("manual_association_label") in {"correct", "partial"}),
        "problem_rate_count": sum(1 for row in rows if row.get("manual_association_label") in {"overmerge", "artifact"}),
        "overmerge_rate_count": sum(1 for row in rows if row.get("manual_association_label") == "overmerge"),
        "undermerge_rate_count": sum(1 for row in rows if is_undermerge(row)),
        "artifact_rate_count": sum(1 for row in rows if row.get("manual_association_label") == "artifact"),
        "uncertain_rate_count": sum(1 for row in rows if row.get("manual_association_label") == "uncertain"),
    }
    rates: Dict[str, Any] = dict(counts)
    for count_key, count in counts.items():
        rate_key = count_key.replace("_count", "")
        rates[rate_key] = count / total if total else 0.0
    return rates


def is_undermerge(row: Mapping[str, Any]) -> bool:
    return row.get("manual_association_label") == "undermerge" or row.get("manual_parent_status") == "part_of_larger_source"


def n_gaussians_bin(value: float) -> str:
    n = int(value)
    if n <= 1:
        return "01_singleton"
    if n == 2:
        return "02_two"
    if n <= 4:
        return "03_3_to_4"
    if n <= 9:
        return "04_5_to_9"
    return "05_10_plus"


def las_beam_bin(value: float) -> str:
    if value < 2:
        return "01_lt_2"
    if value < 5:
        return "02_2_to_5"
    if value < 10:
        return "03_5_to_10"
    if value < 20:
        return "04_10_to_20"
    return "05_20_plus"
