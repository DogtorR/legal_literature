"""JSONL output helpers for manifests, candidates, failures, metadata, and logs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


MANIFEST_FILE = Path("manifest.jsonl")
FAILURES_FILE = Path("failures.jsonl")
CANDIDATES_FILE = Path("candidates.jsonl")
METADATA_RESULTS_FILE = Path("metadata_results.jsonl")
SEARCH_LOG_FILE = Path("search_log.jsonl")
LEGALITY_AUDIT_FILE = Path("legality_audit.jsonl")
DOWNLOAD_PLAN_FILE = Path("download_plan.jsonl")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default))
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    write_failure("bad jsonl line skipped", {"path": str(input_path), "line_number": line_number, "error": str(exc)})
    return records


def write_manifest(record: dict[str, Any], path: str | Path = MANIFEST_FILE) -> None:
    append_jsonl(path, record)


def write_failure(reason: str | dict[str, Any], context: dict[str, Any] | None = None, path: str | Path = FAILURES_FILE) -> None:
    if isinstance(reason, dict):
        record = reason
    else:
        record = {"reason": reason, "context": context or {}}
    append_jsonl(path, record)


def write_candidate(record: dict[str, Any], path: str | Path = CANDIDATES_FILE) -> None:
    append_jsonl(path, record)


def write_metadata_result(record: dict[str, Any], path: str | Path = METADATA_RESULTS_FILE) -> None:
    append_jsonl(path, record)


def write_metadata_results(records: list[dict[str, Any]], path: str | Path = METADATA_RESULTS_FILE) -> int:
    for record in records:
        append_jsonl(path, record)
    return len(records)


def write_search_log(event: str | dict[str, Any], context: dict[str, Any] | None = None, path: str | Path = SEARCH_LOG_FILE) -> None:
    if isinstance(event, dict):
        record = event
    else:
        record = {"event": event, "context": context or {}}
    append_jsonl(path, record)


def write_candidates(records: list[dict[str, Any]], path: str | Path = CANDIDATES_FILE) -> int:
    for record in records:
        write_candidate(record, path=path)
    return len(records)


def write_legality_audit(record: dict[str, Any], path: str | Path = LEGALITY_AUDIT_FILE) -> None:
    append_jsonl(path, record)


def write_legality_audits(records: list[dict[str, Any]], path: str | Path = LEGALITY_AUDIT_FILE) -> int:
    for record in records:
        write_legality_audit(record, path=path)
    return len(records)


def build_manifest_record(download_request: Any, download_result: Any) -> dict[str, Any]:
    req = download_request if isinstance(download_request, dict) else (download_request.to_dict() if hasattr(download_request, "to_dict") else (asdict(download_request) if is_dataclass(download_request) else getattr(download_request, "__dict__", {})))
    res = download_result if isinstance(download_result, dict) else (download_result.to_dict() if hasattr(download_result, "to_dict") else (asdict(download_result) if is_dataclass(download_result) else getattr(download_result, "__dict__", {})))
    return {
        "input_id": req.get("input_id", ""),
        "title": req.get("title", ""),
        "doi": req.get("doi", ""),
        "pmid": req.get("pmid", ""),
        "openalex_id": req.get("openalex_id", ""),
        "publication_year": req.get("publication_year"),
        "publication_date": req.get("publication_date", ""),
        "publication_type": req.get("publication_type", ""),
        "journal": req.get("journal", ""),
        "authors": req.get("authors", []),
        "keywords_matched": req.get("keywords_matched", []),
        "source": req.get("source", ""),
        "pdf_url": req.get("pdf_url", ""),
        "landing_url": req.get("landing_url", ""),
        "license": req.get("license", ""),
        "oa_status": req.get("oa_status", ""),
        "oa_evidence": req.get("evidence_sources", []),
        "is_legal_oa": req.get("legality_decision") == "allowed_for_future_download",
        "local_path": res.get("local_path", ""),
        "sha256": res.get("sha256", ""),
        "downloaded_at": res.get("downloaded_at", ""),
        "status": res.get("status", ""),
        "reason": res.get("reason", ""),
    }


def write_download_plan(record: dict[str, Any], path: str | Path = DOWNLOAD_PLAN_FILE) -> None:
    append_jsonl(path, record)
