#!/usr/bin/env python3
"""Serve the local radio association annotation UI with Python stdlib only."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lotss_association.annotation import (
    MANUAL_ASSOCIATION_LABELS,
    MANUAL_ASSOCIATION_TYPES,
    MANUAL_EVIDENCE_FLAGS,
    MANUAL_PARENT_STATUS,
    MANUAL_PROBLEM_FLAGS,
    MANUAL_QUALITY,
    append_annotation,
    compute_dashboard_stats,
    enrich_manifest_with_annotations,
    filter_manifest_rows,
    latest_annotations_by_item,
    make_annotation_record,
    read_annotations_jsonl,
    read_csv_dicts,
    sort_manifest_rows,
)


def resolve_repo_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


class AnnotationServer:
    def __init__(
        self,
        manifest_path: Path,
        annotation_file: Path,
        static_dir: Path,
        queue_mode: str = "priority",
    ) -> None:
        self.manifest_path = manifest_path
        self.annotation_file = annotation_file
        self.static_dir = static_dir
        self.queue_mode = queue_mode

    def read_manifest_state(self):
        manifest_rows = read_csv_dicts(self.manifest_path)
        annotations = [record for record in read_annotations_jsonl(self.annotation_file) if not record.get("_invalid_json")]
        latest = latest_annotations_by_item(annotations)
        enriched = enrich_manifest_with_annotations(manifest_rows, latest)
        return manifest_rows, annotations, latest, enriched


def make_handler(server_state: AnnotationServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.serve_static("index.html")
            elif parsed.path in {"/index.html", "/app.js", "/style.css"}:
                self.serve_static(parsed.path.lstrip("/"))
            elif parsed.path == "/api/state":
                self.handle_state(parsed)
            elif parsed.path == "/api/file":
                self.handle_file(parsed)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/save":
                self.handle_save(status="annotated")
            elif parsed.path == "/api/skip":
                self.handle_save(status="skipped")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def read_json_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8"))

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def serve_static(self, name: str):
            path = (server_state.static_dir / name).resolve()
            if not str(path).startswith(str(server_state.static_dir.resolve())) or not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_file(self, parsed):
            query = parse_qs(parsed.query)
            requested = unquote(query.get("path", [""])[0])
            if not requested:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
                return
            path = resolve_repo_path(Path(requested))
            repo_root = REPO_ROOT.resolve()
            try:
                path.relative_to(repo_root)
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "File path outside repository")
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=60")
            self.end_headers()
            self.wfile.write(data)

        def handle_state(self, parsed):
            query = parse_qs(parsed.query)
            queue_mode = query.get("queue_mode", [server_state.queue_mode])[0] or server_state.queue_mode
            review_mode = parse_bool(query.get("review_mode", ["false"])[0])
            unannotated_only = parse_bool(query.get("unannotated_only", ["true"])[0])
            suspicious_only = parse_bool(query.get("suspicious_only", ["false"])[0])
            min_n_gaussians = query.get("min_n_gaussians", [""])[0]
            min_las_beam = query.get("min_las_beam", [""])[0]
            qualities = [v for raw in query.get("quality", []) for v in raw.split(",") if v]
            types = [v for raw in query.get("type", []) for v in raw.split(",") if v]
            search = query.get("search", [""])[0]
            _, annotations, latest, enriched = server_state.read_manifest_state()
            filtered = filter_manifest_rows(
                enriched,
                qualities=qualities,
                types=types,
                min_n_gaussians=int(min_n_gaussians) if min_n_gaussians else None,
                min_las_beam=float(min_las_beam) if min_las_beam else None,
                suspicious_only=suspicious_only,
                unannotated_only=unannotated_only,
                search=search,
                review_mode=review_mode,
            )
            filtered = sort_manifest_rows(filtered, queue_mode=queue_mode)
            for row in filtered:
                item_id = str(row.get("item_id", ""))
                row["image_url"] = "/api/file?path=" + quote(str(row.get("image_path", "")))
                overview = str(row.get("overview_image_path", ""))
                row["overview_image_url"] = "/api/file?path=" + quote(overview) if overview else ""
                row["latest_annotation"] = latest.get(item_id, {})
            self.send_json(
                {
                    "items": filtered,
                    "all_progress": compute_dashboard_stats(enriched),
                    "filtered_progress": compute_dashboard_stats(filtered),
                    "annotation_history_count": len(annotations),
                    "options": {
                        "manual_association_label": MANUAL_ASSOCIATION_LABELS,
                        "manual_parent_status": MANUAL_PARENT_STATUS,
                        "manual_association_type": MANUAL_ASSOCIATION_TYPES,
                        "manual_quality": MANUAL_QUALITY,
                        "manual_evidence_flags": MANUAL_EVIDENCE_FLAGS,
                        "manual_problem_flags": MANUAL_PROBLEM_FLAGS,
                    },
                }
            )

        def handle_save(self, status: str):
            try:
                payload = self.read_json_body()
                item_id = str(payload.get("item_id", ""))
                if not item_id:
                    self.send_json({"ok": False, "error": "Missing item_id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                manifest_rows = read_csv_dicts(server_state.manifest_path)
                manifest_by_item = {str(row.get("item_id", "")): row for row in manifest_rows}
                manifest_row = manifest_by_item.get(item_id)
                if manifest_row is None:
                    self.send_json({"ok": False, "error": f"Unknown item_id: {item_id}"}, status=HTTPStatus.NOT_FOUND)
                    return
                record = make_annotation_record(manifest_row, payload, status=status)
                append_annotation(server_state.annotation_file, record)
                self.send_json({"ok": True, "record": record})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="annotations/manifest.csv")
    parser.add_argument("--annotation-file", required=True, type=Path, help="annotations/annotations.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument(
        "--queue-mode",
        choices=["all", "priority", "suspicious", "high", "random"],
        default="priority",
        help="Default queue ordering in the web UI",
    )
    args = parser.parse_args()

    state = AnnotationServer(
        manifest_path=resolve_repo_path(args.manifest),
        annotation_file=resolve_repo_path(args.annotation_file),
        static_dir=(REPO_ROOT / "annotation").resolve(),
        queue_mode=args.queue_mode,
    )
    state.annotation_file.parent.mkdir(parents=True, exist_ok=True)
    if not state.annotation_file.exists():
        state.annotation_file.touch()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving annotation UI at http://{args.host}:{args.port}")
    print(f"Manifest: {state.manifest_path}")
    print(f"Annotations: {state.annotation_file}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

