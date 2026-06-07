"""Queue-style batch metadata runner for broad CRISPR corpus collection."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .crispr_scope import evaluate_crispr_scope
from .legality import assess_legal_oa_candidate, is_forbidden_text, is_forbidden_url
from .manifest import append_jsonl, read_jsonl
from .metadata import deduplicate_records
from .query import QuerySpec
from .sources import crossref, europe_pmc, openalex, pubmed


SOURCE_MODULES = {
    "pubmed": pubmed,
    "openalex": openalex,
    "crossref": crossref,
    "europe_pmc": europe_pmc,
}

FULL_TEXT_STATUSES = (
    "downloaded_pdf",
    "planned_legal_oa",
    "metadata_only_non_oa",
    "metadata_only_no_pdf_url",
    "metadata_only_missing_oa_evidence",
    "metadata_only_needs_manual_review",
    "download_failed_retryable",
    "download_failed_final",
    "excluded_out_of_scope",
)
DOWNLOAD_QA_RISK_TERMS = (
    "sci-hub",
    "scihub",
    "libgen",
    "z-library",
    "zlibrary",
    "paywall",
    "login",
    "vpn",
    "cookie",
    "cookies",
    "captcha",
    "bypass",
)


class _BatchMetadataResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, url: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url

    def json(self) -> dict[str, Any]:
        return self._payload


class _BatchMetadataSession:
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = max(1, int(timeout_seconds))

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> _BatchMetadataResponse:
        full_url = url
        if params:
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{urlencode(params, doseq=True)}"
        request = Request(full_url, headers=headers or {})
        effective_timeout = min(int(timeout or self.timeout_seconds), self.timeout_seconds)
        with urlopen(request, timeout=effective_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = int(getattr(response, "status", 200) or 200)
            final_url = str(getattr(response, "url", full_url) or full_url)
        if not isinstance(payload, dict):
            raise ValueError("metadata response JSON must be an object")
        return _BatchMetadataResponse(payload, status_code=status, url=final_url)


def _slug(value: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()[:max_len] or "query"


def _filename_slug(value: Any, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").lower()
    return text[:max_len] or "record"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(path, record)


def _truthy_evidence(value: Any) -> bool:
    if isinstance(value, list):
        return any(bool(item) for item in value)
    return bool(value)


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("doi") or record.get("pmid") or record.get("pmcid") or record.get("title") or "").strip().lower()


def _record_id(record: dict[str, Any], index: int) -> str:
    value = _record_key(record)
    if value:
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()[:120]
    return f"record_{index}"


def _source_provenance(record: dict[str, Any]) -> list[Any]:
    return list(record.get("seen_sources") or record.get("sources") or ([record.get("source")] if record.get("source") else []))


def _record_lookup_key(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or record.get("doi") or record.get("pmid") or record.get("pmcid") or record.get("title") or "").strip().lower()


def _full_text_status(record: dict[str, Any], downloaded: dict[str, Any] | None = None) -> str:
    downloaded = downloaded or {}
    pdf_path = downloaded.get("local_path") or downloaded.get("pdf_path") or record.get("pdf_path")
    download_status = str(downloaded.get("status") or record.get("download_status") or "").lower()
    if pdf_path and download_status in {"downloaded", "ok", "success", ""}:
        return "downloaded_pdf"
    if download_status in {"failed_retryable", "retryable"}:
        return "download_failed_retryable"
    if download_status in {"failed", "error", "blocked_final", "final"}:
        return "download_failed_final"
    legal_pdf_url = record.get("legal_pdf_url") or record.get("pdf_url") or record.get("pdf_url_candidate")
    evidence = record.get("evidence") or record.get("oa_evidence") or record.get("evidence_summary")
    if record.get("can_download") and legal_pdf_url and _truthy_evidence(evidence):
        return "planned_legal_oa"
    if str(record.get("scope_status") or "").lower() in {"needs_manual_review", "manual_review"}:
        return "metadata_only_needs_manual_review"
    if record.get("is_oa") is False or str(record.get("oa_status") or "").lower() in {"closed", "non_oa", "not_oa"}:
        return "metadata_only_non_oa"
    if not legal_pdf_url:
        return "metadata_only_no_pdf_url"
    if not _truthy_evidence(evidence):
        return "metadata_only_missing_oa_evidence"
    if record.get("needs_manual_review"):
        return "metadata_only_needs_manual_review"
    return "metadata_only_missing_oa_evidence"


def _failure_reason_for_status(status: str, record: dict[str, Any]) -> str:
    if record.get("failure_reason"):
        return str(record["failure_reason"])
    return {
        "planned_legal_oa": "legal_oa_pdf_planned_not_downloaded",
        "metadata_only_non_oa": "confirmed_or_inferred_non_oa",
        "metadata_only_no_pdf_url": "no_legal_pdf_url_available",
        "metadata_only_missing_oa_evidence": "missing_explicit_oa_evidence",
        "metadata_only_needs_manual_review": "needs_manual_review",
        "download_failed_retryable": "download_failed_retryable",
        "download_failed_final": "download_failed_final",
        "excluded_out_of_scope": record.get("scope_reason") or record.get("excluded_reason") or "excluded_out_of_scope",
    }.get(status, "metadata_only_full_text_unavailable")


def _manifest_row(record: dict[str, Any], index: int, downloaded: dict[str, Any] | None = None) -> dict[str, Any]:
    status = _full_text_status(record, downloaded)
    downloaded = downloaded or {}
    pdf_path = downloaded.get("local_path") or downloaded.get("pdf_path") or record.get("pdf_path") or ""
    download_success = status == "downloaded_pdf"
    return {
        "record_id": record.get("record_id") or _record_id(record, index),
        "doi": record.get("doi", ""),
        "pmid": record.get("pmid", ""),
        "pmcid": record.get("pmcid", ""),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "journal": record.get("journal", ""),
        "year": record.get("publication_year") or record.get("year"),
        "authors": record.get("authors") or [],
        "sources": _source_provenance(record),
        "source_provenance": _source_provenance(record),
        "matched_queries": record.get("matched_queries") or record.get("keywords") or [],
        "scope_status": record.get("scope_status", ""),
        "scope_reason": record.get("scope_reason") or record.get("excluded_reason") or "",
        "oa_status": record.get("oa_status", ""),
        "is_oa": bool(record.get("is_oa")),
        "license": record.get("license", ""),
        "legal_pdf_url": record.get("legal_pdf_url") or record.get("pdf_url") or record.get("pdf_url_candidate") or "",
        "pdf_path": pdf_path,
        "full_text_status": status,
        "failure_reason": "" if download_success else _failure_reason_for_status(status, record),
        "needs_manual_review": bool(record.get("needs_manual_review") or status == "metadata_only_needs_manual_review"),
        "provenance": {"batch_id": record.get("batch_id", ""), "query_id": record.get("query_id", ""), "sources": _source_provenance(record)},
        "can_use_metadata": True,
        "can_use_full_text": download_success,
        "download_attempted": bool(downloaded or record.get("download_attempted")),
        "download_success": download_success,
        "text_extraction_status": "not_started",
    }


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("batch_id")): row for row in read_jsonl(path)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _continuation_state(batch: dict[str, Any], existing_count: int) -> dict[str, Any]:
    source = str(batch.get("source") or "")
    next_cursor = batch.get("next_cursor")
    next_offset = batch.get("next_offset") if batch.get("next_offset") is not None else batch.get("retstart")
    if source == "pubmed":
        offset = int(next_offset if next_offset is not None else existing_count)
        return {"supported": True, "config": {"retstart": offset, "next_offset": offset}, "next_offset": offset, "next_cursor": str(offset)}
    if source in {"openalex", "crossref", "europe_pmc"}:
        if next_cursor:
            return {"supported": True, "config": {"next_cursor": next_cursor, "start_cursor": next_cursor}, "next_offset": None, "next_cursor": next_cursor}
        return {"supported": False, "config": {}, "next_offset": None, "next_cursor": None}
    return {"supported": False, "config": {}, "next_offset": None, "next_cursor": None}


def _batch_schema_defaults(batch: dict[str, Any], output_file: Path) -> dict[str, Any]:
    return {
        **batch,
        "status": batch.get("status") or "pending",
        "total_count": batch.get("total_count"),
        "retrieved_count": int(batch.get("retrieved_count") or 0),
        "page_count": int(batch.get("page_count") or 0),
        "has_more": bool(batch.get("has_more", False)),
        "next_cursor": batch.get("next_cursor"),
        "next_offset": batch.get("next_offset") if batch.get("next_offset") is not None else batch.get("retstart"),
        "retstart": batch.get("retstart") if batch.get("retstart") is not None else batch.get("next_offset"),
        "error": str(batch.get("error") or ""),
        "output_file": str(output_file),
        "last_run_at": batch.get("last_run_at") or "",
        "continuation_supported": bool(batch.get("continuation_supported", False)),
        "stop_reason": str(batch.get("stop_reason") or ""),
    }


def _rewrite_manifest_latest(manifest_path: Path) -> list[dict[str, Any]]:
    latest = _read_manifest(manifest_path)
    normalized: list[dict[str, Any]] = []
    raw_dir = manifest_path.parent / "raw_records_by_batch"
    for row in latest.values():
        output_file = raw_dir / f"{row.get('batch_id')}.jsonl"
        normalized.append(_batch_schema_defaults(row, output_file))
    if normalized:
        _write_jsonl(manifest_path, normalized)
    return normalized


def build_corpus_search_batches(
    queries: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries, 1):
        for source in sources:
            batch_id = f"{query_index:03d}_{_slug(query)}_{source}"
            batches.append(
                {
                    "batch_id": batch_id,
                    "query": query,
                    "source": source,
                    "year_from": year_from,
                    "year_to": year_to,
                    "max_results": max_results_per_batch,
                    "status": "pending",
                    "total_count": None,
                    "retrieved_count": 0,
                    "page_count": 0,
                    "has_more": False,
                    "next_cursor": None,
                    "next_offset": None,
                    "retstart": None,
                    "output_file": "",
                    "last_run_at": "",
                    "error": "",
                    "continuation_supported": source in SOURCE_MODULES,
                    "stop_reason": "",
                }
            )
    return batches


def run_search_batch(batch: dict[str, Any], output_dir: str, timeout_seconds: int = 30, resume: bool = False) -> dict[str, Any]:
    out = Path(output_dir)
    raw_dir = out / "raw_records_by_batch"
    manifest_path = out / "batch_manifest.jsonl"
    search_log_path = out / "search_log.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(batch["batch_id"])
    source = str(batch["source"])
    output_file = raw_dir / f"{batch_id}.jsonl"
    existing_records = read_jsonl(output_file) if resume and output_file.exists() else []
    result = _batch_schema_defaults(batch, output_file)
    result["last_run_at"] = _utc_now()
    try:
        if source not in SOURCE_MODULES:
            raise ValueError(f"Unsupported source: {source}")
        module = SOURCE_MODULES[source]
        continuation = _continuation_state(batch, len(existing_records)) if resume else {"supported": True, "config": {}, "next_offset": None, "next_cursor": None}
        if resume and not continuation["supported"]:
            result.update(
                {
                    "status": "partial",
                    "retrieved_count": len(existing_records) or int(batch.get("retrieved_count") or 0),
                    "has_more": True,
                    "continuation_supported": False,
                    "stop_reason": "missing_continuation_cursor_from_previous_run",
                    "error": "",
                }
            )
            append_jsonl(manifest_path, result)
            append_jsonl(search_log_path, result)
            return result
        fetch_limit = int(batch.get("fetch_limit") or batch["max_results"])
        spec = QuerySpec(
            input_id=batch_id,
            keywords=[str(batch["query"])],
            year_from=int(batch["year_from"]) if batch.get("year_from") not in (None, "") else None,
            year_to=int(batch["year_to"]) if batch.get("year_to") not in (None, "") else None,
            max_results=fetch_limit,
        )
        config = {"max_results_per_source": fetch_limit, "require_keyword_match": False, **continuation.get("config", {})}
        session = _BatchMetadataSession(timeout_seconds=timeout_seconds)
        records = module.search(spec, config, dry_run=False, allow_network=True, session=session)
        info = dict(getattr(module, "LAST_SEARCH_INFO", {}) or {})
        for record in records:
            record["batch_id"] = batch_id
            record["matched_queries"] = sorted(set([*(record.get("matched_queries") or []), str(batch["query"])]))
            record["seen_sources"] = sorted(set([*(record.get("seen_sources") or []), source]))
        merged_records = deduplicate_records([*existing_records, *records]) if existing_records else records
        _write_jsonl(output_file, merged_records)
        total_count = info.get("total_count", result.get("total_count"))
        retrieved_count = len(merged_records)
        has_more = bool(total_count is not None and retrieved_count < int(total_count or 0) and info.get("has_more", False))
        next_cursor = info.get("next_cursor") if has_more else None
        next_offset = info.get("next_offset") if has_more else None
        if source == "pubmed" and has_more:
            next_offset = info.get("retstart") or next_offset or retrieved_count
            next_cursor = str(next_offset)
        stop_reason = "more_results_available" if has_more else "no_more_results"
        if info.get("errors"):
            stop_reason = "api_error"
        result.update(
            {
                "total_count": total_count,
                "retrieved_count": retrieved_count,
                "page_count": int(result.get("page_count") or 0) + int(info.get("page_count", 0) or 0),
                "has_more": has_more,
                "next_cursor": next_cursor,
                "next_offset": next_offset,
                "retstart": next_offset if source == "pubmed" else None,
                "status": "partial" if has_more else "completed",
                "error": "; ".join(str(item) for item in info.get("errors") or []),
                "continuation_supported": bool(source in SOURCE_MODULES),
                "stop_reason": stop_reason,
            }
        )
        if result["error"]:
            result["status"] = "failed"
    except Exception as exc:
        if not existing_records:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("", encoding="utf-8")
        result.update({"status": "failed", "error": str(exc), "stop_reason": "api_error", "continuation_supported": source in SOURCE_MODULES})
    append_jsonl(manifest_path, result)
    append_jsonl(search_log_path, result)
    return result


def run_corpus_batch_search(
    queries: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int = 500,
    output_dir: str = "agent_runs/batch_search",
    resume: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "batch_manifest.jsonl"
    batches = build_corpus_search_batches(queries, sources, year_from, year_to, max_results_per_batch)
    existing = _read_manifest(manifest_path) if resume else {}
    results: list[dict[str, Any]] = []
    for batch in batches:
        previous = existing.get(str(batch["batch_id"]))
        if previous and previous.get("status") == "completed" and resume:
            results.append(previous)
            continue
        if previous and previous.get("status") == "failed" and resume and not retry_failed:
            results.append(previous)
            continue
        results.append(run_search_batch(batch, output_dir))
    merged = merge_batch_outputs(output_dir)
    statuses = Counter(str(row.get("status")) for row in _read_manifest(manifest_path).values())
    return {
        "status": "partial" if statuses.get("failed") or statuses.get("partial") else "ok",
        "total_batches": len(batches),
        "completed_batches": statuses.get("completed", 0),
        "failed_batches": statuses.get("failed", 0),
        "partial_batches": statuses.get("partial", 0),
        "actual_downloads": 0,
        "artifacts": {
            "batch_manifest": str(manifest_path.resolve()),
            "batch_report": str((out / "batch_report.md").resolve()),
            **(merged.get("artifacts") or {}),
        },
        "merge": merged,
    }


def continue_partial_batches(
    output_dir: str,
    max_additional_results_per_batch: int = 500,
    max_batches: int | None = None,
    sources: list[str] | None = None,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    manifest_path = out / "batch_manifest.jsonl"
    manifest = _read_manifest(manifest_path)
    source_filter = set(sources or [])
    query_filter = set(queries or [])
    candidates = [
        row
        for row in manifest.values()
        if row.get("status") == "partial"
        and (not source_filter or row.get("source") in source_filter)
        and (not query_filter or row.get("query") in query_filter)
    ]
    if max_batches is not None:
        candidates = candidates[: max(0, int(max_batches))]
    before_status = {str(row.get("batch_id")): row.get("status") for row in manifest.values()}
    continued: list[dict[str, Any]] = []
    for row in candidates:
        batch = dict(row)
        batch["fetch_limit"] = int(max_additional_results_per_batch)
        continued.append(run_search_batch(batch, output_dir, resume=True))
    merged = merge_batch_outputs(output_dir)
    latest = _read_manifest(manifest_path)
    continued_ids = {str(row.get("batch_id")) for row in continued}
    completed_now = sum(1 for batch_id in continued_ids if before_status.get(batch_id) == "partial" and latest.get(batch_id, {}).get("status") == "completed")
    failed_now = sum(1 for batch_id in continued_ids if latest.get(batch_id, {}).get("status") == "failed")
    still_partial = sum(1 for row in latest.values() if row.get("status") == "partial")
    status_counts = Counter(str(row.get("status")) for row in latest.values())
    report_path = out / "continuation_report.md"
    report_lines = [
        "# Continuation Report",
        "",
        f"- continued_batches: {len(continued)}",
        f"- completed_now: {completed_now}",
        f"- still_partial: {still_partial}",
        f"- failed_now: {failed_now}",
        f"- total_raw_records: {merged.get('total_raw_records', 0)}",
        f"- unique_records: {merged.get('unique_records', 0)}",
        f"- actual_downloads: 0",
        "",
        "## Current Batch Status",
        *(f"- {key}: {value}" for key, value in sorted(status_counts.items())),
        "",
        "## Continued Batches",
    ]
    for row in continued:
        report_lines.append(
            f"- {row.get('batch_id')}: {row.get('status')} retrieved={row.get('retrieved_count')} "
            f"has_more={row.get('has_more')} continuation_supported={row.get('continuation_supported')} "
            f"stop_reason={row.get('stop_reason')} error={row.get('error')}"
        )
    if not continued:
        report_lines.append("- none")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "partial" if still_partial or failed_now else "ok",
        "continued_batches": len(continued),
        "completed_now": completed_now,
        "still_partial": still_partial,
        "failed_now": failed_now,
        "total_raw_records": merged.get("total_raw_records", 0),
        "unique_records": merged.get("unique_records", 0),
        "actual_downloads": 0,
        "artifacts": {
            "batch_manifest": str(manifest_path.resolve()),
            "batch_report": str((out / "batch_report.md").resolve()),
            "continuation_report": str(report_path.resolve()),
            **(merged.get("artifacts") or {}),
        },
        "merge": merged,
    }


def build_final_corpus_manifests(output_dir: str) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    oa_records = read_jsonl(out / "oa_audit.jsonl")
    excluded_records = read_jsonl(out / "scope_excluded_records.jsonl")
    download_rows = read_jsonl(out / "download_results.jsonl") if (out / "download_results.jsonl").exists() else []
    downloaded_by_key: dict[str, dict[str, Any]] = {}
    for row in download_rows:
        rows = row.get("data") if isinstance(row.get("data"), list) else [row]
        for item in rows:
            if not isinstance(item, dict):
                continue
            key = _record_key(item)
            if key:
                downloaded_by_key[key] = item
    corpus: list[dict[str, Any]] = []
    for index, record in enumerate(oa_records, 1):
        corpus.append(_manifest_row(record, index, downloaded_by_key.get(_record_key(record))))
    downloaded_manifest = [row for row in corpus if row["full_text_status"] == "downloaded_pdf"]
    metadata_only_manifest = [row for row in corpus if row["full_text_status"] != "downloaded_pdf"]
    excluded_failures: list[dict[str, Any]] = []
    for index, record in enumerate(excluded_records, 1):
        excluded = _manifest_row({**record, "scope_status": "excluded"}, index)
        excluded["full_text_status"] = "excluded_out_of_scope"
        excluded["can_use_metadata"] = False
        excluded["can_use_full_text"] = False
        excluded["failure_reason"] = _failure_reason_for_status("excluded_out_of_scope", record)
        excluded_failures.append(
            {
                "record_id": excluded["record_id"],
                "doi": excluded["doi"],
                "pmid": excluded["pmid"],
                "pmcid": excluded["pmcid"],
                "title": excluded["title"],
                "stage": "scope_guard",
                "full_text_status": "excluded_out_of_scope",
                "failure_reason": excluded["failure_reason"],
                "trace_location": "scope_excluded_records.jsonl",
            }
        )
    failure_rows = [
        {
            "record_id": row["record_id"],
            "doi": row["doi"],
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "title": row["title"],
            "stage": "full_text_availability",
            "full_text_status": row["full_text_status"],
            "failure_reason": row["failure_reason"],
            "trace_location": "metadata_only_manifest.jsonl",
        }
        for row in metadata_only_manifest
    ] + excluded_failures
    status_counts = Counter(row["full_text_status"] for row in corpus)
    stats = {
        "total_corpus_records": len(corpus),
        "downloaded_pdf_count": len(downloaded_manifest),
        "metadata_only_count": len(metadata_only_manifest),
        "planned_legal_oa_count": status_counts.get("planned_legal_oa", 0),
        "non_oa_count": status_counts.get("metadata_only_non_oa", 0),
        "no_pdf_url_count": status_counts.get("metadata_only_no_pdf_url", 0),
        "missing_oa_evidence_count": status_counts.get("metadata_only_missing_oa_evidence", 0),
        "manual_review_count": status_counts.get("metadata_only_needs_manual_review", 0),
        "excluded_out_of_scope_count": len(excluded_records),
        "download_failed_count": status_counts.get("download_failed_retryable", 0) + status_counts.get("download_failed_final", 0),
        "actual_downloads": len(downloaded_manifest),
    }
    artifacts = {
        "corpus_manifest": str((out / "corpus_manifest.jsonl").resolve()),
        "metadata_only_manifest": str((out / "metadata_only_manifest.jsonl").resolve()),
        "downloaded_pdfs_manifest": str((out / "downloaded_pdfs_manifest.jsonl").resolve()),
        "failures": str((out / "failures.jsonl").resolve()),
        "dataset_card": str((out / "dataset_card.md").resolve()),
        "coverage_report": str((out / "coverage_report.md").resolve()),
    }
    _write_jsonl(out / "corpus_manifest.jsonl", corpus)
    _write_jsonl(out / "metadata_only_manifest.jsonl", metadata_only_manifest)
    _write_jsonl(out / "downloaded_pdfs_manifest.jsonl", downloaded_manifest)
    _write_jsonl(out / "failures.jsonl", failure_rows)
    statement = "PDF availability is a full-text availability layer; metadata records are retained even when legal PDF download is unavailable."
    stat_lines = [
        f"- total_corpus_records: {stats['total_corpus_records']}",
        f"- downloaded_pdf_count: {stats['downloaded_pdf_count']}",
        f"- metadata_only_count: {stats['metadata_only_count']}",
        f"- planned_legal_oa_count: {stats['planned_legal_oa_count']}",
        f"- non_oa_count: {stats['non_oa_count']}",
        f"- no_pdf_url_count: {stats['no_pdf_url_count']}",
        f"- missing_oa_evidence_count: {stats['missing_oa_evidence_count']}",
        f"- manual_review_count: {stats['manual_review_count']}",
        f"- excluded_out_of_scope_count: {stats['excluded_out_of_scope_count']}",
        f"- download_failed_count: {stats['download_failed_count']}",
    ]
    dataset_lines = [
        "# Dataset Card",
        "",
        "## Corpus Policy",
        statement,
        "",
        "## Final Corpus Statistics",
        *stat_lines,
        "",
        "## Legal Download Policy",
        "No PDF is downloaded by manifest finalization. PDF files may only be downloaded through the legal OA gate with explicit OA evidence, allow_download=True, and yes=True.",
    ]
    (out / "dataset_card.md").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    section = "\n".join(["", "## Final Corpus Manifest Layer", statement, "", *stat_lines, ""])
    for report_name in ("coverage_report.md", "batch_report.md"):
        path = out / report_name
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else f"# {report_name.replace('_', ' ').replace('.md', '').title()}\n"
        marker = "## Final Corpus Manifest Layer"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + section
        else:
            text = text.rstrip() + section
        path.write_text(text + "\n", encoding="utf-8")
    return {
        "status": "ok",
        **stats,
        "corpus_manifest_count": len(corpus),
        "metadata_only_manifest_count": len(metadata_only_manifest),
        "downloaded_pdfs_manifest_count": len(downloaded_manifest),
        "failures_count": len(failure_rows),
        "failures_traceable": all(row.get("trace_location") in {"metadata_only_manifest.jsonl", "scope_excluded_records.jsonl"} for row in failure_rows),
        "artifacts": artifacts,
    }


def _qa_has_risk(record: dict[str, Any]) -> bool:
    urls = [
        str(record.get("legal_pdf_url") or ""),
        str(record.get("pdf_url") or ""),
        str(record.get("pdf_url_candidate") or ""),
        str(record.get("landing_url") or ""),
    ]
    if any(is_forbidden_url(url) for url in urls):
        return True
    text = json.dumps(record, ensure_ascii=False).lower().replace("unpaywall", "")
    return is_forbidden_text(text) or any(term in text for term in DOWNLOAD_QA_RISK_TERMS)


def _qa_evidence(record: dict[str, Any]) -> Any:
    return record.get("oa_evidence") or record.get("evidence") or record.get("evidence_summary") or record.get("approval_reason") or ""


def _qa_pdf_url(record: dict[str, Any]) -> str:
    return str(record.get("legal_pdf_url") or record.get("pdf_url") or record.get("pdf_url_candidate") or "")


def _qa_source(record: dict[str, Any]) -> str:
    provenance = _source_provenance(record)
    if provenance:
        return str(provenance[0])
    return str(record.get("oa_source") or record.get("source") or "")


def _qa_row(record: dict[str, Any], reason_key: str, reason: str) -> dict[str, Any]:
    pdf_url = _qa_pdf_url(record)
    evidence = _qa_evidence(record)
    return {
        "record_id": record.get("record_id") or _record_id(record, 0),
        "doi": record.get("doi", ""),
        "title": record.get("title", ""),
        "journal": record.get("journal", ""),
        "year": record.get("year") or record.get("publication_year"),
        "legal_pdf_url": record.get("legal_pdf_url", ""),
        "pdf_url": pdf_url,
        "oa_source": record.get("oa_source") or _qa_source(record),
        "license": record.get("license", ""),
        "oa_evidence": evidence,
        "source_provenance": record.get("source_provenance") or _source_provenance(record),
        "matched_queries": record.get("matched_queries") or [],
        reason_key: reason,
    }


def pre_download_qa_for_legal_oa_candidates(output_dir: str) -> dict[str, Any]:
    out = Path(output_dir)
    corpus = read_jsonl(out / "corpus_manifest.jsonl")
    metadata_only = read_jsonl(out / "metadata_only_manifest.jsonl")
    oa_audit = read_jsonl(out / "oa_audit.jsonl")
    download_plan = read_jsonl(out / "download_manifest.jsonl")
    failures = read_jsonl(out / "failures.jsonl")
    by_key: dict[str, dict[str, Any]] = {}
    for collection in (corpus, metadata_only, oa_audit, download_plan):
        for row in collection:
            key = _record_lookup_key(row)
            if not key:
                continue
            merged = dict(by_key.get(key) or {})
            merged.update({k: v for k, v in row.items() if v not in (None, "", [])})
            by_key[key] = merged
    candidates = [
        row
        for row in by_key.values()
        if row.get("full_text_status") == "planned_legal_oa"
        or row.get("can_download")
        or _qa_pdf_url(row)
    ]
    approved: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in candidates:
        pdf_url = _qa_pdf_url(record)
        evidence = _qa_evidence(record)
        provenance = record.get("source_provenance") or _source_provenance(record)
        if not pdf_url:
            rejected.append(_qa_row(record, "rejection_reason", "missing_legal_pdf_url"))
        elif not evidence:
            rejected.append(_qa_row(record, "rejection_reason", "missing_oa_evidence"))
        elif not (record.get("doi") or record.get("title")):
            rejected.append(_qa_row(record, "rejection_reason", "missing_doi_or_title"))
        elif not provenance:
            rejected.append(_qa_row(record, "rejection_reason", "missing_source_provenance"))
        elif _qa_has_risk(record):
            rejected.append(_qa_row(record, "rejection_reason", "forbidden_source_or_bypass_risk"))
        elif not record.get("license"):
            manual.append(_qa_row(record, "review_reason", "missing_license_with_oa_evidence"))
        else:
            approved.append(_qa_row(record, "approval_reason", "legal_oa_pdf_url_with_evidence_license_and_provenance"))
    _write_jsonl(out / "approved_for_download.jsonl", approved)
    _write_jsonl(out / "needs_manual_download_review.jsonl", manual)
    _write_jsonl(out / "rejected_before_download.jsonl", rejected)
    report_lines = [
        "# Pre-download QA Report",
        "",
        "- actual_downloads: 0",
        f"- candidate_records: {len(candidates)}",
        f"- approved_for_download: {len(approved)}",
        f"- needs_manual_download_review: {len(manual)}",
        f"- rejected_before_download: {len(rejected)}",
        f"- source_failures_seen: {len(failures)}",
        "",
        "## Policy",
        "This step performs pre-download QA only. It does not download PDFs.",
        "Candidates require legal PDF URL, OA evidence, DOI or title, source provenance, no bypass risk, and license for approval.",
    ]
    rejection_counts = Counter(str(row.get("rejection_reason") or "") for row in rejected)
    review_counts = Counter(str(row.get("review_reason") or "") for row in manual)
    report_lines.extend(["", "## Rejection Reasons", *(f"- {k}: {v}" for k, v in rejection_counts.most_common())])
    report_lines.extend(["", "## Manual Review Reasons", *(f"- {k}: {v}" for k, v in review_counts.most_common())])
    report_path = out / "pre_download_qa_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "approved_for_download_count": len(approved),
        "needs_manual_download_review_count": len(manual),
        "rejected_before_download_count": len(rejected),
        "actual_downloads": 0,
        "artifacts": {
            "approved_for_download": str((out / "approved_for_download.jsonl").resolve()),
            "needs_manual_download_review": str((out / "needs_manual_download_review.jsonl").resolve()),
            "rejected_before_download": str((out / "rejected_before_download.jsonl").resolve()),
            "pre_download_qa_report": str(report_path.resolve()),
        },
    }


def _download_candidate_risk(record: dict[str, Any]) -> str:
    pdf_url = _qa_pdf_url(record)
    evidence = _qa_evidence(record)
    provenance = record.get("source_provenance") or _source_provenance(record)
    if not pdf_url:
        return "missing_pdf_url"
    if not evidence:
        return "missing_oa_evidence"
    if not (record.get("doi") or record.get("title")):
        return "missing_doi_or_title"
    if not provenance:
        return "missing_source_provenance"
    if _qa_has_risk(record):
        return "forbidden_source_or_bypass_risk"
    return ""


def _candidate_url_from_location(location: Any) -> str:
    if isinstance(location, dict):
        return str(
            location.get("pdf_url")
            or location.get("url_for_pdf")
            or location.get("url")
            or location.get("landing_url")
            or ""
        )
    return str(location or "")


def _append_candidate(candidates: list[dict[str, Any]], record: dict[str, Any], pdf_url: str, source: str, evidence: Any, license_value: str, priority: int, reason: str) -> None:
    if not pdf_url:
        return
    candidate = {
        "pdf_url": pdf_url,
        "source": source,
        "evidence": evidence,
        "license": license_value,
        "priority": priority,
        "reason": reason,
    }
    text = json.dumps({**record, **candidate}, ensure_ascii=False).lower().replace("unpaywall", "")
    if is_forbidden_url(pdf_url) or is_forbidden_text(text) or any(term in text for term in DOWNLOAD_QA_RISK_TERMS):
        candidate["rejected"] = True
        candidate["rejection_reason"] = "forbidden_source_or_bypass_risk"
    candidates.append(candidate)


FULLTEXT_FORMAT_DIRS = {
    "pdf": "pdf",
    "pmc_xml": "xml",
    "europe_pmc_xml": "xml",
    "publisher_html": "html",
    "html": "html",
    "xml": "xml",
    "repository_file": "repository",
    "preprint_file": "preprints",
}

FULLTEXT_EXTENSIONS = {
    "pdf": ".pdf",
    "pmc_xml": ".xml",
    "europe_pmc_xml": ".xml",
    "publisher_html": ".html",
    "html": ".html",
    "xml": ".xml",
    "repository_file": "",
    "preprint_file": "",
}

FULLTEXT_TYPE_RANK = {
    "pdf": 1,
    "pmc_xml": 2,
    "europe_pmc_xml": 3,
    "publisher_html": 4,
    "html": 4,
    "repository_file": 5,
    "preprint_file": 6,
    "xml": 3,
}


def _fulltext_url_from_location(location: Any) -> str:
    if isinstance(location, dict):
        return str(
            location.get("pdf_url")
            or location.get("url_for_pdf")
            or location.get("fullTextUrl")
            or location.get("full_text_url")
            or location.get("url")
            or location.get("oa_url")
            or location.get("landing_page_url")
            or location.get("landing_url")
            or ""
        )
    return str(location or "")


def _infer_full_text_type(url: str, source: str = "", location: dict[str, Any] | None = None) -> str:
    location = location or {}
    explicit = str(location.get("full_text_type") or location.get("documentStyle") or location.get("availability") or "").lower()
    lowered = url.lower()
    source_l = source.lower()
    if "pmc/articles/pmc" in lowered and ("xml" in lowered or "oa_package" in lowered):
        return "pmc_xml"
    if "europepmc.org" in lowered and ("xml" in lowered or "fulltextxml" in lowered):
        return "europe_pmc_xml"
    if lowered.endswith(".pdf") or "/pdf" in lowered or "pdf=render" in lowered or explicit == "pdf":
        return "pdf"
    if lowered.endswith(".xml") or "format=xml" in lowered or explicit == "xml":
        return "xml"
    if any(host in lowered for host in ("biorxiv.org", "medrxiv.org", "arxiv.org")):
        return "preprint_file"
    if any(term in source_l for term in ("repository", "pmc", "europe pmc", "zenodo", "figshare", "institutional")):
        return "repository_file"
    if lowered.endswith(".html") or lowered.endswith(".htm") or explicit == "html":
        return "publisher_html"
    return "html"


def _append_fulltext_candidate(
    candidates: list[dict[str, Any]],
    record: dict[str, Any],
    url: str,
    source: str,
    evidence: Any,
    license_value: str,
    priority: int,
    reason: str,
    *,
    full_text_type: str | None = None,
    location: dict[str, Any] | None = None,
) -> None:
    if not url:
        return
    ftype = full_text_type or _infer_full_text_type(url, source, location)
    candidate = {
        "url": url,
        "full_text_type": ftype,
        "source": source,
        "license": license_value,
        "oa_evidence": evidence,
        "priority": priority,
        "reason": reason,
        "derived_from": reason,
        "requires_login": False,
        "is_paywalled": False,
        "is_legal_oa": bool(evidence),
        "source_policy": "",
        "safety_status": "pending",
    }
    lowered = url.lower()
    if "biorxiv.org" in lowered or "medrxiv.org" in lowered:
        candidate["source_policy"] = "preprint_public_endpoint_no_bypass"
    ok, reason_text = is_safe_legal_fulltext_candidate(candidate)
    if not ok:
        candidate["rejected"] = True
        candidate["rejection_reason"] = reason_text
        candidate["is_legal_oa"] = False
        candidate["safety_status"] = f"rejected:{reason_text}"
    else:
        candidate["safety_status"] = "safe_legal_oa_candidate"
    candidates.append(candidate)


def _preprint_fallback_urls(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    urls = [
        str(record.get("legal_pdf_url") or ""),
        str(record.get("pdf_url") or ""),
        str(record.get("url") or ""),
        str(record.get("landing_url") or ""),
    ]
    doi = str(record.get("doi") or "").strip()
    output: list[tuple[str, str, str]] = []
    for url in urls:
        lowered = url.lower()
        if "biorxiv.org" not in lowered and "medrxiv.org" not in lowered:
            continue
        base = "https://www.biorxiv.org" if "biorxiv.org" in lowered else "https://www.medrxiv.org"
        if doi:
            output.append((f"{base}/content/{doi}v1.full", "preprint_file", "preprint_standard_full_html"))
            output.append((f"{base}/content/{doi}v1.full.pdf", "pdf", "preprint_standard_pdf"))
        if ".full.pdf" in lowered:
            output.append((url.replace(".full.pdf", ".full"), "preprint_file", "preprint_full_html_from_pdf_url"))
    return output


def _with_network_fulltext_fallbacks(record: dict[str, Any], allow_network: bool) -> dict[str, Any]:
    if not allow_network:
        return record
    enriched = dict(record)
    doi = str(enriched.get("doi") or "").strip()
    pmid = str(enriched.get("pmid") or "").strip()
    try:
        if doi and not (enriched.get("unpaywall_best_oa_location") or enriched.get("unpaywall_oa_locations")):
            from .sources import unpaywall

            result = unpaywall.check_doi(
                doi,
                {"contact_email": "contact@example.org"},
                dry_run=False,
                allow_network=True,
                query_id=str(enriched.get("record_id") or "fulltext_retry"),
            )
            for key in ("unpaywall_best_oa_location", "unpaywall_oa_locations", "oa_locations", "license", "pmcid"):
                if result.get(key) and not enriched.get(key):
                    enriched[key] = result.get(key)
            if result.get("is_legal_oa_candidate") and not enriched.get("oa_evidence"):
                enriched["oa_evidence"] = result.get("oa_evidence") or result.get("evidence") or "Unpaywall confirmed OA"
    except Exception:
        pass
    try:
        if (doi or pmid) and not (enriched.get("fullTextXML") or enriched.get("fullTextUrlList")):
            spec = QuerySpec(
                input_id=str(enriched.get("record_id") or "fulltext_retry"),
                doi=doi,
                pmid=pmid,
                max_results=3,
            )
            records = europe_pmc.search(spec, {"max_results_per_source": 3, "require_keyword_match": False}, dry_run=False, allow_network=True)
            if records:
                epmc = records[0]
                for key in ("pmcid", "best_oa_location", "pdf_url_candidate", "license"):
                    if epmc.get(key) and not enriched.get(key):
                        enriched[key] = epmc.get(key)
                if epmc.get("pmcid") and not enriched.get("oa_evidence"):
                    enriched["oa_evidence"] = "Europe PMC/PMC open full text metadata"
    except Exception:
        pass
    return enriched


def _should_network_fulltext_enrich(record: dict[str, Any]) -> bool:
    if record.get("skip_network_fallback"):
        return False
    if record.get("force_network_fallback"):
        return True
    urls = [
        str(record.get("legal_pdf_url") or ""),
        str(record.get("pdf_url") or ""),
        str(record.get("url") or ""),
        str(record.get("landing_url") or ""),
    ]
    if not any(urls):
        return bool(record.get("doi") or record.get("pmid"))
    return any(url.lower().startswith(("http://", "https://")) for url in urls if url)


def resolve_legal_fulltext_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    base_evidence = _qa_evidence(record)
    base_license = str(record.get("license") or "")
    base_source = str(record.get("oa_source") or record.get("source") or "approved_record")
    for url_key in ("legal_pdf_url", "pdf_url"):
        _append_fulltext_candidate(candidates, record, str(record.get(url_key) or ""), base_source, base_evidence, base_license, 1, f"approved_record_{url_key}", full_text_type="pdf")
    open_access = record.get("open_access") or {}
    if isinstance(open_access, dict):
        _append_fulltext_candidate(candidates, record, str(open_access.get("oa_url") or ""), "openalex_open_access", base_evidence or "OpenAlex OA URL", base_license, 20, "openalex_open_access_oa_url")
    for loc_key, priority_base, source_name in [
        ("primary_location", 30, "openalex_primary_location"),
        ("best_oa_location", 35, "openalex_best_oa_location"),
        ("locations", 40, "openalex_locations"),
        ("oa_locations", 60, "oa_locations"),
        ("unpaywall_best_oa_location", 80, "unpaywall_best_oa_location"),
        ("unpaywall_oa_locations", 90, "unpaywall_oa_locations"),
    ]:
        values = record.get(loc_key)
        locs = values if isinstance(values, list) else ([values] if values else [])
        for index, location in enumerate(locs, 1):
            if isinstance(location, dict):
                evidence = location.get("evidence") or location.get("oa_evidence") or location.get("url_for_pdf") or base_evidence
                license_value = str(location.get("license") or base_license)
                source = str(location.get("source") or source_name)
                url = _fulltext_url_from_location(location)
            else:
                evidence = base_evidence
                license_value = base_license
                source = source_name
                url = _fulltext_url_from_location(location)
            _append_fulltext_candidate(candidates, record, url, source, evidence, license_value, priority_base + index, loc_key, location=location if isinstance(location, dict) else None)
    for index, item in enumerate(record.get("fullTextUrlList") or record.get("full_text_url_list") or [], 1):
        location = item if isinstance(item, dict) else {"url": item}
        _append_fulltext_candidate(candidates, record, _fulltext_url_from_location(location), "europe_pmc_fullTextUrlList", base_evidence or "Europe PMC full text URL", base_license, 120 + index, "europe_pmc_fullTextUrlList", location=location)
    for key, priority, ftype, source in [
        ("fullTextXML", 110, "europe_pmc_xml", "europe_pmc_fullTextXML"),
        ("full_text_xml", 111, "europe_pmc_xml", "europe_pmc_full_text_xml"),
        ("pmc_xml_url", 130, "pmc_xml", "pmc_xml"),
        ("publisher_html_url", 150, "publisher_html", "publisher_oa_html"),
        ("repository_fulltext_url", 170, "repository_file", "repository_fulltext"),
        ("preprint_fulltext_url", 180, "preprint_file", "preprint_fulltext"),
    ]:
        _append_fulltext_candidate(candidates, record, str(record.get(key) or ""), source, base_evidence, base_license, priority, key, full_text_type=ftype)
    pmcid = str(record.get("pmcid") or "").strip()
    if pmcid:
        normalized = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        _append_fulltext_candidate(candidates, record, f"https://www.ncbi.nlm.nih.gov/pmc/articles/{normalized}/?report=xml", "pmc", base_evidence or "PMC open full text candidate", base_license, 200, "pmc_article_xml", full_text_type="pmc_xml")
        _append_fulltext_candidate(candidates, record, f"https://www.ncbi.nlm.nih.gov/pmc/articles/{normalized}/pdf/", "pmc", base_evidence or "PMC open full text candidate", base_license, 201, "pmc_pdf_url", full_text_type="pdf")
        _append_fulltext_candidate(candidates, record, f"https://www.ebi.ac.uk/europepmc/webservices/rest/{normalized}/fullTextXML", "europe_pmc", base_evidence or "Europe PMC open full text candidate", base_license, 202, "europe_pmc_fullTextXML", full_text_type="europe_pmc_xml")
    for index, (url, ftype, reason) in enumerate(_preprint_fallback_urls(record), 1):
        _append_fulltext_candidate(candidates, record, url, "preprint_server", base_evidence or "Public preprint server full text candidate", base_license, 220 + index, reason, full_text_type=ftype)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (FULLTEXT_TYPE_RANK.get(str(item.get("full_text_type") or ""), 99), int(item.get("priority") or 999))):
        url = str(candidate.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(candidate)
    return unique


def is_safe_legal_fulltext_candidate(candidate: dict[str, Any]) -> tuple[bool, str]:
    url = str(candidate.get("url") or "")
    evidence = candidate.get("oa_evidence") or candidate.get("evidence")
    text = json.dumps(
        {
            "url": url,
            "source": candidate.get("source", ""),
            "reason": candidate.get("reason", ""),
            "oa_evidence": evidence,
        },
        ensure_ascii=False,
    ).lower().replace("unpaywall", "")
    risk_terms = tuple(DOWNLOAD_QA_RISK_TERMS) + (
        "login",
        "sign-in",
        "signin",
        "session-required",
        "captcha",
        "vpn",
        "bypass",
        "unauthorized mirror",
        "paywall",
    )
    if not url:
        return False, "missing_url"
    if is_forbidden_url(url) or is_forbidden_text(text) or any(term in text for term in risk_terms):
        return False, "forbidden_source_or_bypass_risk"
    if candidate.get("requires_login") is True:
        return False, "requires_login"
    if candidate.get("is_paywalled") is True:
        return False, "paywalled"
    if not evidence:
        return False, "missing_oa_evidence"
    if candidate.get("is_legal_oa") is False:
        return False, "uncertain_legality"
    return True, ""


def resolve_legal_oa_pdf_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    base_evidence = _qa_evidence(record)
    base_license = str(record.get("license") or "")
    _append_candidate(candidates, record, str(record.get("legal_pdf_url") or record.get("pdf_url") or ""), str(record.get("oa_source") or "approved_record"), base_evidence, base_license, 1, "approved_record_pdf_url")
    for index, location in enumerate(record.get("oa_locations") or [], 1):
        if isinstance(location, dict):
            evidence = location.get("evidence") or base_evidence
            license_value = str(location.get("license") or base_license)
            source = str(location.get("source") or "oa_locations")
        else:
            evidence = base_evidence
            license_value = base_license
            source = "oa_locations"
        _append_candidate(candidates, record, _candidate_url_from_location(location), source, evidence, license_value, 10 + index, "record_oa_locations")
    best = record.get("unpaywall_best_oa_location")
    if best:
        _append_candidate(candidates, record, _candidate_url_from_location(best), "unpaywall_best_oa_location", base_evidence or "Unpaywall best OA location", base_license, 30, "unpaywall_best_oa_location")
    for index, location in enumerate(record.get("unpaywall_oa_locations") or [], 1):
        evidence = location.get("evidence") if isinstance(location, dict) else base_evidence
        license_value = str((location.get("license") if isinstance(location, dict) else "") or base_license)
        _append_candidate(candidates, record, _candidate_url_from_location(location), "unpaywall_oa_locations", evidence or base_evidence or "Unpaywall OA location", license_value, 40 + index, "unpaywall_oa_locations")
    pmcid = str(record.get("pmcid") or "").strip()
    if pmcid:
        normalized = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        _append_candidate(candidates, record, f"https://www.ncbi.nlm.nih.gov/pmc/articles/{normalized}/pdf/", "pmc", base_evidence or "PMC open full text candidate", base_license, 60, "pmc_pdf_url")
        _append_candidate(candidates, record, f"https://europepmc.org/articles/{normalized}?pdf=render", "europe_pmc", base_evidence or "Europe PMC open full text candidate", base_license, 61, "europe_pmc_pdf_url")
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: int(item.get("priority") or 999)):
        url = str(candidate.get("pdf_url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(candidate)
    return unique


def _download_filename(record: dict[str, Any]) -> str:
    parts = [
        _filename_slug(record.get("record_id"), 45),
        _filename_slug(record.get("doi"), 45),
        _filename_slug(record.get("title"), 90),
    ]
    filename = "__".join(part for part in parts if part and part != "record")
    return f"{filename or 'legal_oa_pdf'}.pdf"


def _fulltext_filename(record: dict[str, Any], full_text_type: str, url: str) -> str:
    extension = FULLTEXT_EXTENSIONS.get(full_text_type, "")
    if not extension:
        parsed_ext = Path(urlparse(url).path).suffix
        extension = parsed_ext if parsed_ext and len(parsed_ext) <= 10 else ".dat"
    parts = [
        _filename_slug(record.get("record_id"), 45),
        _filename_slug(record.get("doi"), 45),
        _filename_slug(record.get("title"), 90),
    ]
    filename = "__".join(part for part in parts if part and part != "record")
    return f"{filename or 'legal_oa_fulltext'}{extension}"


def _format_allowed(candidate: dict[str, Any], prefer_formats: list[str] | None) -> bool:
    if not prefer_formats:
        return True
    return str(candidate.get("full_text_type") or "") in set(prefer_formats)


def _fulltext_url_has_risk(url: str) -> bool:
    text = str(url or "").lower()
    extra_risk_terms = ("misuse", "abuse.shtml", "/error/abuse", "error=cookies_not_supported", "needaccess=true", "needaccess")
    return is_forbidden_url(text) or is_forbidden_text(text) or any(term in text for term in DOWNLOAD_QA_RISK_TERMS) or any(term in text for term in extra_risk_terms)


def _existing_fulltext_row_safe(row: dict[str, Any]) -> bool:
    return not (_fulltext_url_has_risk(str(row.get("url") or "")) or _fulltext_url_has_risk(str(row.get("final_url") or "")))


def _recommended_next_action(final_error: str, rejected_candidates: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> str:
    text = " ".join([final_error, json.dumps(rejected_candidates, ensure_ascii=False), json.dumps(attempts, ensure_ascii=False)]).lower()
    if "403" in text or "forbidden" in text:
        return "blocked_by_server"
    if "missing_oa_evidence" in text or "uncertain" in text or "bypass" in text or "cookie" in text:
        return "manual_review"
    if not attempts:
        return "no_legal_fulltext_found"
    return "retry_later"


def _failure_retryable(final_error: str, action: str) -> bool:
    text = final_error.lower()
    return action in {"retry_later", "blocked_by_server"} or any(term in text for term in ("403", "404", "timeout", "connection", "temporarily"))


def collect_approved_legal_fulltexts(
    output_dir: str,
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
    prefer_formats: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    approved_path = out / approved_file
    approved = read_jsonl(approved_path)
    limit = len(approved) if max_items is None else max(0, int(max_items))
    records = approved[:limit]
    existing_before = read_jsonl(out / "collected_fulltexts_manifest.jsonl") if (out / "collected_fulltexts_manifest.jsonl").exists() else []
    already_collected_keys = {
        _record_lookup_key(row)
        for row in existing_before
        if _record_lookup_key(row) and _existing_fulltext_row_safe(row)
    }
    already_collected_skipped = [
        row
        for row in records
        if _record_lookup_key(row) and _record_lookup_key(row) in already_collected_keys
    ]
    records_to_process = [
        row
        for row in records
        if not (_record_lookup_key(row) and _record_lookup_key(row) in already_collected_keys)
    ]
    collected: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    now = _utc_now
    if not allow_download or not yes:
        skipped = [
            {
                "record_id": row.get("record_id", ""),
                "doi": row.get("doi", ""),
                "title": row.get("title", ""),
                "skip_reason": "allow_download_and_yes_required",
            }
            for row in records
        ]
    else:
        for record in records_to_process:
            if _should_network_fulltext_enrich(record):
                record = _with_network_fulltext_fallbacks(record, allow_network=True)
            all_candidates = [candidate for candidate in resolve_legal_fulltext_candidates(record) if _format_allowed(candidate, prefer_formats)]
            safe_candidates: list[dict[str, Any]] = []
            rejected_candidates: list[dict[str, Any]] = []
            for candidate in all_candidates:
                ok, reason = is_safe_legal_fulltext_candidate(candidate)
                if ok:
                    safe_candidates.append(candidate)
                else:
                    rejected = dict(candidate)
                    rejected["review_reason"] = reason
                    rejected_candidates.append(rejected)
            if rejected_candidates:
                manual_review.append(
                    {
                        "record_id": record.get("record_id", ""),
                        "doi": record.get("doi", ""),
                        "title": record.get("title", ""),
                        "review_reason": "one_or_more_candidates_need_manual_review",
                        "candidates": rejected_candidates,
                    }
                )
            if not safe_candidates:
                action = _recommended_next_action("no_safe_legal_fulltext_candidate", rejected_candidates, [])
                failed.append(
                    {
                        "record_id": record.get("record_id", ""),
                        "doi": record.get("doi", ""),
                        "title": record.get("title", ""),
                        "attempted_candidates": [],
                        "rejected_candidates": rejected_candidates,
                        "final_failure_reason": "no_safe_legal_fulltext_candidate",
                        "retryable": _failure_retryable("no_safe_legal_fulltext_candidate", action),
                        "recommended_next_action": action,
                    }
                )
                continue
            attempts: list[dict[str, Any]] = []
            success = False
            final_error = ""
            for rank, candidate in enumerate(safe_candidates, 1):
                url = str(candidate.get("url") or "")
                ftype = str(candidate.get("full_text_type") or "html")
                attempt = {"candidate_rank": rank, "url": url, "source": candidate.get("source", ""), "full_text_type": ftype, "status": "pending"}
                try:
                    request = Request(url, headers={"User-Agent": "LEGAL_LITERATURE_AGENT/0.1"})
                    with urlopen(request, timeout=30) as response:
                        final_url = str(getattr(response, "url", url) or url)
                        if _fulltext_url_has_risk(final_url):
                            raise RuntimeError("forbidden_or_bypass_risk_final_url")
                        payload = response.read()
                    if not payload:
                        raise RuntimeError("empty_fulltext_response")
                    checksum = hashlib.sha256(payload).hexdigest()
                    subdir = FULLTEXT_FORMAT_DIRS.get(ftype, "repository")
                    target_dir = out / "fulltexts" / subdir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / _fulltext_filename(record, ftype, final_url)
                    if target.exists():
                        target = target_dir / f"{target.stem}_{checksum[:10]}{target.suffix}"
                    target.write_bytes(payload)
                    attempt.update({"status": "collected", "final_url": final_url, "file_size": len(payload), "checksum_sha256": checksum})
                    attempts.append(attempt)
                    collected.append(
                        {
                            "record_id": record.get("record_id", ""),
                            "doi": record.get("doi", ""),
                            "title": record.get("title", ""),
                            "journal": record.get("journal", ""),
                            "year": record.get("year"),
                            "full_text_type": ftype,
                            "full_text_format": ftype.split("_")[-1] if "_" in ftype else ftype,
                            "source": candidate.get("source", ""),
                            "url": url,
                            "final_url": final_url,
                            "local_path": str(target.resolve()),
                            "license": candidate.get("license") or record.get("license", ""),
                            "oa_evidence": candidate.get("oa_evidence") or _qa_evidence(record),
                            "file_size": len(payload),
                            "checksum_sha256": checksum,
                            "collected_at": now(),
                            "candidate_rank": rank,
                            "attempts": attempts,
                            "access_mode": "legal_oa",
                        }
                    )
                    success = True
                    break
                except Exception as exc:
                    final_error = str(exc)
                    attempt.update({"status": "failed", "failure_reason": final_error})
                    attempts.append(attempt)
            if not success:
                action = _recommended_next_action(final_error or "all_candidates_failed", rejected_candidates, attempts)
                failed.append(
                    {
                        "record_id": record.get("record_id", ""),
                        "doi": record.get("doi", ""),
                        "title": record.get("title", ""),
                        "attempted_candidates": attempts,
                        "rejected_candidates": rejected_candidates,
                        "final_failure_reason": final_error or "all_candidates_failed",
                        "retryable": _failure_retryable(final_error or "all_candidates_failed", action),
                        "recommended_next_action": action,
                    }
                )
    unsafe_failed_keys = {
        _record_lookup_key(row)
        for row in failed
        if "forbidden" in str(row.get("final_failure_reason") or "") or "bypass" in str(row.get("final_failure_reason") or "")
    }
    existing_raw = read_jsonl(out / "collected_fulltexts_manifest.jsonl") if (out / "collected_fulltexts_manifest.jsonl").exists() else []
    existing: list[dict[str, Any]] = []
    for row in existing_raw:
        if _record_lookup_key(row) in unsafe_failed_keys:
            local_path = Path(str(row.get("local_path") or ""))
            if local_path.exists() and local_path.is_file():
                local_path.unlink()
        elif _existing_fulltext_row_safe(row):
            existing.append(row)
        else:
            local_path = Path(str(row.get("local_path") or ""))
            if local_path.exists() and local_path.is_file():
                local_path.unlink()
    by_key = {_record_lookup_key(row) or str(row.get("local_path") or ""): row for row in existing}
    for row in collected:
        by_key[_record_lookup_key(row) or str(row.get("local_path") or "")] = row
    merged_collected = list(by_key.values())
    _write_jsonl(out / "collected_fulltexts_manifest.jsonl", merged_collected)
    _write_jsonl(out / "failed_fulltext_fetches.jsonl", failed)
    _write_jsonl(out / "needs_manual_fulltext_review.jsonl", manual_review)
    type_counts = Counter(str(row.get("full_text_type") or "") for row in collected)
    report_path = out / "fulltext_collection_report.md"
    report_lines = [
        "# Legal OA Fulltext Collection Report",
        "",
        f"- allow_download: {allow_download}",
        f"- yes: {yes}",
        f"- attempted_items: {len(records_to_process) if allow_download and yes else 0}",
        f"- already_collected_skipped: {len(already_collected_skipped)}",
        f"- actual_fulltexts: {len(collected)}",
        f"- pdf_count: {type_counts.get('pdf', 0)}",
        f"- xml_count: {sum(count for key, count in type_counts.items() if 'xml' in key)}",
        f"- html_count: {sum(count for key, count in type_counts.items() if 'html' in key)}",
        f"- repository_count: {type_counts.get('repository_file', 0)}",
        f"- preprint_count: {type_counts.get('preprint_file', 0)}",
        f"- failed_fulltext_fetches: {len(failed)}",
        f"- needs_manual_fulltext_review: {len(manual_review)}",
        "",
        "Only records from the approved file are eligible. Manual-review and rejected records are not read for automatic fulltext collection.",
        "No paywall, login, cookie, CAPTCHA, VPN, or shadow-library bypass is used.",
    ]
    if failed:
        report_lines.extend(["", "## Failed Fulltext Fetches", *(f"- {row.get('record_id')}: {row.get('final_failure_reason')}" for row in failed)])
    if collected:
        report_lines.extend(["", "## Collected Fulltexts", *(f"- {row.get('record_id')}: {row.get('full_text_type')} size={row.get('file_size')} sha256={row.get('checksum_sha256')}" for row in collected)])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "ok" if not failed else "partial",
        "attempted_items": len(records_to_process) if allow_download and yes else 0,
        "already_collected_skipped": len(already_collected_skipped),
        "newly_attempted": len(records_to_process) if allow_download and yes else 0,
        "actual_fulltexts": len(collected),
        "pdf_count": type_counts.get("pdf", 0),
        "xml_count": sum(count for key, count in type_counts.items() if "xml" in key),
        "html_count": sum(count for key, count in type_counts.items() if "html" in key),
        "repository_count": type_counts.get("repository_file", 0),
        "preprint_count": type_counts.get("preprint_file", 0),
        "failed_fulltext_fetches": len(failed),
        "needs_manual_fulltext_review": len(manual_review),
        "skipped_items": len(skipped) + len(already_collected_skipped),
        "actual_downloads": len(collected),
        "collected": collected,
        "failed": failed,
        "manual_review": manual_review,
        "artifacts": {
            "collected_fulltexts_manifest": str((out / "collected_fulltexts_manifest.jsonl").resolve()),
            "failed_fulltext_fetches": str((out / "failed_fulltext_fetches.jsonl").resolve()),
            "needs_manual_fulltext_review": str((out / "needs_manual_fulltext_review.jsonl").resolve()),
            "fulltext_collection_report": str(report_path.resolve()),
            "fulltexts_dir": str((out / "fulltexts").resolve()),
        },
    }


def retry_failed_legal_fulltext_fetches(
    output_dir: str,
    failed_file: str = "failed_fulltext_fetches.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    failed_rows = [row for row in read_jsonl(out / failed_file) if row.get("retryable")]
    approved_rows = read_jsonl(out / "approved_for_download.jsonl")
    approved_by_key = {_record_lookup_key(row): row for row in approved_rows if _record_lookup_key(row)}
    retry_rows: list[dict[str, Any]] = []
    for failure in failed_rows:
        approved = approved_by_key.get(_record_lookup_key(failure))
        if approved:
            retry_rows.append(approved)
    if max_items is not None:
        retry_rows = retry_rows[: max(0, int(max_items))]
    if not allow_download or not yes:
        return {
            "status": "ok",
            "attempted_items": 0,
            "actual_fulltexts": 0,
            "pdf_count": 0,
            "xml_count": 0,
            "html_count": 0,
            "repository_count": 0,
            "preprint_count": 0,
            "failed_fulltext_fetches": 0,
            "needs_manual_fulltext_review": 0,
            "skipped_items": len(retry_rows),
            "actual_downloads": 0,
            "retry_attempted": 0,
            "retry_success": 0,
            "retry_failed": 0,
            "new_pdf_count": 0,
            "new_xml_count": 0,
            "new_html_count": 0,
            "rejected_candidates": 0,
            "blocked_by_server_count": 0,
            "artifacts": {
                "collected_fulltexts_manifest": str((out / "collected_fulltexts_manifest.jsonl").resolve()),
                "failed_fulltext_fetches": str((out / failed_file).resolve()),
                "retry_failed_fulltext_fetches": str((out / "retry_failed_fulltext_fetches.jsonl").resolve()),
            },
        }
    retry_file = out / "_retry_approved_for_fulltext.jsonl"
    _write_jsonl(retry_file, retry_rows)
    before = read_jsonl(out / "collected_fulltexts_manifest.jsonl")
    result = collect_approved_legal_fulltexts(
        output_dir=output_dir,
        approved_file=retry_file.name,
        allow_download=allow_download,
        yes=yes,
        max_items=len(retry_rows),
        prefer_formats=["pdf", "pmc_xml", "europe_pmc_xml", "publisher_html", "repository_file", "preprint_file", "html", "xml"],
    )
    after = read_jsonl(out / "collected_fulltexts_manifest.jsonl")
    before_keys = {_record_lookup_key(row) for row in before if _record_lookup_key(row)}
    new_rows = [row for row in after if _record_lookup_key(row) and _record_lookup_key(row) not in before_keys]
    retry_failed = list(result.get("failed") or [])
    _write_jsonl(out / "retry_failed_fulltext_fetches.jsonl", retry_failed)
    type_counts = Counter(str(row.get("full_text_type") or "") for row in new_rows)
    blocked_by_server_count = sum(1 for row in retry_failed if row.get("recommended_next_action") == "blocked_by_server")
    rejected_count = sum(len(row.get("rejected_candidates") or []) for row in retry_failed)
    result.update(
        {
            "retry_attempted": len(retry_rows) if allow_download and yes else 0,
            "retry_success": len(new_rows),
            "retry_failed": len(retry_failed),
            "new_pdf_count": type_counts.get("pdf", 0),
            "new_xml_count": sum(count for key, count in type_counts.items() if "xml" in key),
            "new_html_count": sum(count for key, count in type_counts.items() if "html" in key),
            "rejected_candidates": rejected_count,
            "blocked_by_server_count": blocked_by_server_count,
            "retry_failed_fulltext_fetches": len(retry_failed),
        }
    )
    artifacts = dict(result.get("artifacts") or {})
    artifacts["retry_failed_fulltext_fetches"] = str((out / "retry_failed_fulltext_fetches.jsonl").resolve())
    result["artifacts"] = artifacts
    return result


def download_approved_legal_oa_pdfs(
    output_dir: str,
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    approved_path = out / approved_file
    downloads_dir = out / "downloads"
    approved = read_jsonl(approved_path)
    limit = len(approved) if max_downloads is None else max(0, int(max_downloads))
    candidates = approved[:limit]
    downloaded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    now = _utc_now
    if not allow_download or not yes:
        reason = "allow_download_and_yes_required"
        skipped = [
            {
                "record_id": row.get("record_id", ""),
                "doi": row.get("doi", ""),
                "title": row.get("title", ""),
                "pdf_url": _qa_pdf_url(row),
                "skip_reason": reason,
            }
            for row in candidates
        ]
    else:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        for record in candidates:
            pdf_url = _qa_pdf_url(record)
            risk = _download_candidate_risk(record)
            if risk:
                failed.append(
                    {
                        "record_id": record.get("record_id", ""),
                        "doi": record.get("doi", ""),
                        "title": record.get("title", ""),
                        "pdf_url": pdf_url,
                        "failure_reason": risk,
                        "retryable": False,
                        "failed_at": now(),
                    }
                )
                continue
            candidate_list = resolve_legal_oa_pdf_candidates(record)
            if not candidate_list:
                candidate_list = [{"pdf_url": pdf_url, "source": record.get("oa_source") or "approved_record", "evidence": _qa_evidence(record), "license": record.get("license", ""), "priority": 1, "reason": "approved_record_pdf_url"}]
            attempts: list[dict[str, Any]] = []
            success = False
            try:
                final_error = ""
                for rank, candidate in enumerate(candidate_list, 1):
                    candidate_url = str(candidate.get("pdf_url") or "")
                    attempt = {"candidate_rank": rank, "pdf_url": candidate_url, "source": candidate.get("source", ""), "status": "pending"}
                    if candidate.get("rejected"):
                        attempt.update({"status": "rejected", "failure_reason": candidate.get("rejection_reason")})
                        attempts.append(attempt)
                        final_error = str(candidate.get("rejection_reason") or "candidate_rejected")
                        continue
                    try:
                        request = Request(candidate_url, headers={"User-Agent": "LEGAL_LITERATURE_AGENT/0.1"})
                        with urlopen(request, timeout=30) as response:
                            final_url = str(getattr(response, "url", candidate_url) or candidate_url)
                            if is_forbidden_url(final_url):
                                raise RuntimeError("forbidden_final_url")
                            payload = response.read()
                        if not payload:
                            raise RuntimeError("empty_pdf_response")
                        checksum = hashlib.sha256(payload).hexdigest()
                        target = downloads_dir / _download_filename(record)
                        if target.exists():
                            stem = target.stem
                            target = downloads_dir / f"{stem}_{checksum[:10]}.pdf"
                        target.write_bytes(payload)
                        attempt.update({"status": "downloaded", "final_pdf_url": final_url, "file_size": len(payload), "checksum_sha256": checksum})
                        attempts.append(attempt)
                        downloaded.append(
                            {
                                "record_id": record.get("record_id", ""),
                                "doi": record.get("doi", ""),
                                "title": record.get("title", ""),
                                "journal": record.get("journal", ""),
                                "year": record.get("year"),
                                "pdf_url": candidate_url,
                                "final_pdf_url": final_url,
                                "pdf_path": str(target.resolve()),
                                "oa_source": record.get("oa_source", ""),
                                "license": record.get("license", ""),
                                "oa_evidence": _qa_evidence(record),
                                "source_provenance": record.get("source_provenance") or [],
                                "file_size": len(payload),
                                "checksum_sha256": checksum,
                                "downloaded_at": now(),
                                "download_status": "downloaded",
                                "selected_candidate_source": candidate.get("source", ""),
                                "candidate_rank": rank,
                                "download_attempts": attempts,
                            }
                        )
                        success = True
                        break
                    except Exception as candidate_exc:
                        final_error = str(candidate_exc)
                        attempt.update({"status": "failed", "failure_reason": final_error})
                        attempts.append(attempt)
                if success:
                    continue
                raise RuntimeError(final_error or "all_candidates_failed")
            except Exception as exc:
                failed.append(
                    {
                        "record_id": record.get("record_id", ""),
                        "doi": record.get("doi", ""),
                        "title": record.get("title", ""),
                        "pdf_url": pdf_url,
                        "failure_reason": str(exc),
                        "final_failure_reason": str(exc),
                        "attempted_candidates": attempts,
                        "retryable": True,
                        "failed_at": now(),
                    }
                )
    existing_downloads = read_jsonl(out / "downloaded_pdfs_manifest.jsonl") if (out / "downloaded_pdfs_manifest.jsonl").exists() else []
    by_record = {_record_lookup_key(row): row for row in existing_downloads if _record_lookup_key(row)}
    for row in downloaded:
        by_record[_record_lookup_key(row)] = row
    _write_jsonl(out / "downloaded_pdfs_manifest.jsonl", list(by_record.values()))
    _write_jsonl(out / "failed_downloads.jsonl", failed)
    report_lines = [
        "# Approved Legal OA PDF Download Report",
        "",
        f"- allow_download: {allow_download}",
        f"- yes: {yes}",
        f"- attempted_downloads: {len(candidates) if allow_download and yes else 0}",
        f"- actual_downloads: {len(downloaded)}",
        f"- failed_downloads: {len(failed)}",
        f"- skipped_downloads: {len(skipped)}",
        f"- approved_file: {approved_file}",
        f"- downloads_dir: {downloads_dir.resolve()}",
        "",
        "Only records from approved_for_download.jsonl are eligible. Manual-review and rejected candidates are not read for download.",
    ]
    if failed:
        report_lines.extend(["", "## Failed Downloads", *(f"- {row.get('record_id')}: {row.get('failure_reason')}" for row in failed)])
    if downloaded:
        report_lines.extend(["", "## Downloaded PDFs", *(f"- {row.get('record_id')}: size={row.get('file_size')} sha256={row.get('checksum_sha256')}" for row in downloaded)])
    report_path = out / "download_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "ok" if not failed else "partial",
        "attempted_downloads": len(candidates) if allow_download and yes else 0,
        "actual_downloads": len(downloaded),
        "failed_downloads": len(failed),
        "skipped_downloads": len(skipped),
        "downloads_dir": str(downloads_dir.resolve()),
        "downloaded": downloaded,
        "failed": failed,
        "artifacts": {
            "downloaded_pdfs_manifest": str((out / "downloaded_pdfs_manifest.jsonl").resolve()),
            "failed_downloads": str((out / "failed_downloads.jsonl").resolve()),
            "download_report": str(report_path.resolve()),
            "downloads_dir": str(downloads_dir.resolve()),
        },
    }


def retry_failed_legal_oa_downloads(
    output_dir: str,
    failed_file: str = "failed_downloads.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    failed_rows = [row for row in read_jsonl(out / failed_file) if row.get("retryable")]
    approved_rows = read_jsonl(out / "approved_for_download.jsonl")
    approved_by_key = {_record_lookup_key(row): row for row in approved_rows if _record_lookup_key(row)}
    retry_rows: list[dict[str, Any]] = []
    for failure in failed_rows:
        key = _record_lookup_key(failure)
        approved = approved_by_key.get(key)
        if approved:
            retry_rows.append(approved)
    if max_downloads is not None:
        retry_rows = retry_rows[: max(0, int(max_downloads))]
    retry_file = out / "_retry_approved_for_download.jsonl"
    _write_jsonl(retry_file, retry_rows)
    result = download_approved_legal_oa_pdfs(
        output_dir,
        approved_file=retry_file.name,
        allow_download=allow_download,
        yes=yes,
        max_downloads=len(retry_rows),
    )
    result["retry_attempted"] = result.get("attempted_downloads", 0)
    result["retry_success"] = result.get("actual_downloads", 0)
    result["retry_failed"] = result.get("failed_downloads", 0)
    return result


def _make_oa_audit(record: dict[str, Any]) -> dict[str, Any]:
    decision = assess_legal_oa_candidate(record).to_dict()
    legal_pdf_url = record.get("legal_pdf_url") or record.get("pdf_url_candidate") or decision.get("pdf_url_candidate") or ""
    evidence = decision.get("evidence_summary") or record.get("oa_evidence") or record.get("evidence") or ""
    can_download = bool(decision.get("is_legal_oa") and legal_pdf_url and evidence)
    return {
        **record,
        "is_oa": bool(decision.get("is_legal_oa")),
        "legal_pdf_url": legal_pdf_url,
        "license": record.get("license") or decision.get("license") or "",
        "evidence": evidence,
        "can_download": can_download,
        "needs_manual_review": not can_download,
        "failure_reason": "" if can_download else decision.get("reason", "not_confirmed_legal_oa"),
    }


def _evaluate_collection_scope(record: dict[str, Any], scope_profile: str = "crispr_broad", include_terms: list[str] | None = None, exclude_terms: list[str] | None = None) -> dict[str, Any]:
    if scope_profile.startswith("crispr"):
        return evaluate_crispr_scope(record, include_reviews=True, conservative=True)
    text = " ".join(str(record.get(field) or "") for field in ("title", "abstract", "journal", "publication_type")).lower()
    for term in exclude_terms or []:
        if term and str(term).lower() in text:
            return {"scope_status": "excluded", "scope_reason": f"excluded_term:{term}"}
    terms = [str(term).lower() for term in include_terms or [] if str(term).strip()]
    if terms and not any(term in text for term in terms):
        return {"scope_status": "needs_manual_review", "scope_reason": "include_terms_not_confirmed"}
    return {"scope_status": "included", "scope_reason": "generic_scope_included"}


def merge_batch_outputs(output_dir: str, scope_profile: str = "crispr_broad", include_terms: list[str] | None = None, exclude_terms: list[str] | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    raw_dir = out / "raw_records_by_batch"
    raw_records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []:
        raw_records.extend(read_jsonl(path))
    unique_records = deduplicate_records(raw_records)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for record in unique_records:
        scoped = dict(record)
        scoped.update(_evaluate_collection_scope(scoped, scope_profile=scope_profile, include_terms=include_terms, exclude_terms=exclude_terms))
        if scoped["scope_status"] == "included":
            included.append(scoped)
        elif scoped["scope_status"] == "excluded":
            excluded.append(scoped)
        else:
            manual.append(scoped)
    topic_records = included + manual
    oa_audit = [_make_oa_audit(record) for record in topic_records]
    download_plan = [
        {
            "doi": record.get("doi", ""),
            "title": record.get("title", ""),
            "legal_pdf_url": record.get("legal_pdf_url", ""),
            "license": record.get("license", ""),
            "evidence": record.get("evidence", ""),
            "planned_status": "planned_legal_oa_download",
        }
        for record in oa_audit
        if record.get("can_download")
    ]
    failures = [
        {"stage": "scope_guard", "title": record.get("title", ""), "failure_reason": record.get("scope_reason", "scope_excluded")}
        for record in excluded
    ] + [
        {"stage": "oa_audit", "title": record.get("title", ""), "doi": record.get("doi", ""), "failure_reason": record.get("failure_reason", "not_confirmed_legal_oa")}
        for record in oa_audit
        if not record.get("can_download")
    ]
    artifacts = {
        "raw_records_merged": str((out / "raw_records_merged.jsonl").resolve()),
        "metadata_all": str((out / "metadata_all.jsonl").resolve()),
        "unique_records": str((out / "unique_records.jsonl").resolve()),
        "scope_included_records": str((out / "scope_included_records.jsonl").resolve()),
        "scope_excluded_records": str((out / "scope_excluded_records.jsonl").resolve()),
        "needs_manual_scope_review": str((out / "needs_manual_scope_review.jsonl").resolve()),
        "oa_audit": str((out / "oa_audit.jsonl").resolve()),
        "download_manifest": str((out / "download_manifest.jsonl").resolve()),
        "failures": str((out / "failures.jsonl").resolve()),
        "coverage_report": str((out / "coverage_report.md").resolve()),
        "batch_report": str((out / "batch_report.md").resolve()),
    }
    _write_jsonl(out / "raw_records_merged.jsonl", raw_records)
    _write_jsonl(out / "metadata_all.jsonl", unique_records)
    _write_jsonl(out / "unique_records.jsonl", unique_records)
    _write_jsonl(out / "scope_included_records.jsonl", included)
    _write_jsonl(out / "scope_excluded_records.jsonl", excluded)
    _write_jsonl(out / "needs_manual_scope_review.jsonl", manual)
    _write_jsonl(out / "oa_audit.jsonl", oa_audit)
    _write_jsonl(out / "download_manifest.jsonl", download_plan)
    _write_jsonl(out / "failures.jsonl", failures)
    manifest_rows = _rewrite_manifest_latest(out / "batch_manifest.jsonl") if (out / "batch_manifest.jsonl").exists() else []
    per_source = Counter()
    per_query = Counter()
    errors = []
    has_more = []
    for row in manifest_rows:
        per_source[str(row.get("source"))] += int(row.get("retrieved_count") or 0)
        per_query[str(row.get("query"))] += int(row.get("retrieved_count") or 0)
        if row.get("error"):
            errors.append(row)
        if row.get("has_more"):
            has_more.append(row)
    report_lines = [
        "# Batch Search Report",
        "",
        f"- total_batches: {len(manifest_rows)}",
        f"- completed_batches: {sum(1 for row in manifest_rows if row.get('status') == 'completed')}",
        f"- failed_batches: {sum(1 for row in manifest_rows if row.get('status') == 'failed')}",
        f"- partial_batches: {sum(1 for row in manifest_rows if row.get('status') == 'partial')}",
        f"- total_raw_records: {len(raw_records)}",
        f"- batches_with_has_more: {len(has_more)}",
        "",
        "## Per Source Retrieved Count",
        *(f"- {key}: {value}" for key, value in per_source.most_common()),
        "",
        "## Per Query Retrieved Count",
        *(f"- {key}: {value}" for key, value in per_query.most_common()),
        "",
        "## API Errors",
        *(f"- {row.get('batch_id')}: {row.get('error')}" for row in errors),
        "",
        "## Suggested Next Steps",
        "- Run with retry_failed=True for failed batches.",
        "- Increase max_results_per_batch for partial batches with has_more=True.",
    ]
    (out / "batch_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out / "coverage_report.md").write_text(
        "\n".join(
            [
                "# Coverage Report",
                "",
                f"- total_batches: {len(manifest_rows)}",
                f"- total_raw_records: {len(raw_records)}",
                f"- unique_records: {len(unique_records)}",
                f"- scope_included_records: {len(included)}",
                f"- scope_excluded_records: {len(excluded)}",
                f"- needs_manual_scope_review: {len(manual)}",
                f"- oa_audit_records: {len(oa_audit)}",
                f"- can_download: {len(download_plan)}",
                f"- actual_downloads: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "total_raw_records": len(raw_records),
        "unique_records": len(unique_records),
        "scope_included_records": len(included),
        "scope_excluded_records": len(excluded),
        "needs_manual_scope_review": len(manual),
        "oa_audit_records": len(oa_audit),
        "can_download": len(download_plan),
        "actual_downloads": 0,
        "per_source": dict(per_source),
        "per_query": dict(per_query),
        "artifacts": artifacts,
    }
