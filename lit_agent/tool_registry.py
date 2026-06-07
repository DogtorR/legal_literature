"""Standardized tool layer shared by MCP tools and LangGraph nodes."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .downloader import DownloadRequest, execute_download_plan, request_from_legality_decision
from .crispr_scope import evaluate_crispr_scope
from .legality import assess_legal_oa_candidate, is_forbidden_text, is_forbidden_url
from .manifest import append_jsonl, read_jsonl, write_download_plan, write_failure, write_legality_audits, write_manifest
from .query import QuerySpec
from .report import write_reports
from .sources import crossref, europe_pmc, openalex, pubmed, unpaywall


class _FastMetadataResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, url: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url

    def json(self) -> dict[str, Any]:
        return self._payload


class _FastMetadataSession:
    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int | None = None, **_kwargs: Any) -> _FastMetadataResponse:
        full_url = url
        if params:
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{urlencode(params, doseq=True)}"
        request = Request(full_url, headers=headers or {})
        with urlopen(request, timeout=min(int(timeout or self.timeout), self.timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = int(getattr(response, "status", 200) or 200)
            final_url = str(getattr(response, "url", full_url) or full_url)
        return _FastMetadataResponse(payload, status_code=status, url=final_url)


FAST_METADATA_SESSION = _FastMetadataSession(timeout=5)


def _result(
    status: str,
    tool: str,
    data: Any = None,
    artifacts: dict[str, Any] | None = None,
    error: str = "",
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "tool": tool,
        "data": data,
        "artifacts": artifacts or {},
        "error": error,
        "retryable": retryable,
    }


def _query_spec(query: str, year_from: int | None, year_to: int | None, max_results: int, **kwargs: Any) -> QuerySpec:
    keywords = kwargs.get("keywords")
    if isinstance(keywords, str):
        keyword_values = [part.strip() for part in keywords.replace("|", ";").split(";") if part.strip()]
    elif isinstance(keywords, list):
        keyword_values = [str(item) for item in keywords if str(item).strip()]
    else:
        keyword_values = [query] if query else []
    return QuerySpec(
        input_id=str(kwargs.get("query_id") or "tool_query"),
        title=str(kwargs.get("title") or ""),
        doi=str(kwargs.get("doi") or ""),
        pmid=str(kwargs.get("pmid") or ""),
        keywords=keyword_values,
        year_from=year_from,
        year_to=year_to,
        publication_date_from=str(kwargs.get("publication_date_from") or ""),
        publication_date_to=str(kwargs.get("publication_date_to") or ""),
        journal=str(kwargs.get("journal") or ""),
        max_results=max_results,
        include_terms=list(kwargs.get("include_terms") or []),
        exclude_terms=list(kwargs.get("exclude_terms") or []),
    )


def _config(max_results: int, **kwargs: Any) -> dict[str, Any]:
    return {
        "max_results_per_source": max_results,
        "contact_email": kwargs.get("contact_email") or "contact@example.org",
        "require_keyword_match": kwargs.get("require_keyword_match", False),
    }


def _allow_network(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("allow_network") or kwargs.get("allow_network_metadata"))


def _search_tool(tool: str, module: Any, query: str, year_from: int | None, year_to: int | None, max_results: int, **kwargs: Any) -> dict[str, Any]:
    try:
        spec = _query_spec(query, year_from, year_to, max_results, **kwargs)
        allow_network = _allow_network(kwargs)
        records = module.search(spec, _config(max_results, **kwargs), dry_run=not allow_network, allow_network=allow_network, session=FAST_METADATA_SESSION if allow_network else None)
        info = dict(getattr(module, "LAST_SEARCH_INFO", {}) or {})
        data = {
            "records": records,
            "total_count": info.get("total_count"),
            "retrieved_count": int(info.get("retrieved_count", len(records)) or 0),
            "page_count": int(info.get("page_count", 0) or 0),
            "has_more": bool(info.get("has_more", False)),
            "next_cursor": info.get("next_cursor"),
            "query_used": query,
            "date_filter": {"year_from": year_from, "year_to": year_to},
            "source_info": info,
        }
        status = "partial" if data["has_more"] or (data["total_count"] is not None and data["retrieved_count"] < int(data["total_count"])) else "ok"
        return {
            "status": status,
            "tool": tool,
            "query": query,
            "source": tool.replace("search_", ""),
            "data": data,
            "artifacts": {"count": len(records), "source_info": info},
            "error": "",
            "retryable": bool(info.get("errors")),
        }
    except Exception as exc:
        return {
            "status": "error",
            "tool": tool,
            "query": query,
            "source": tool.replace("search_", ""),
            "data": {
                "records": [],
                "total_count": None,
                "retrieved_count": 0,
                "page_count": 0,
                "has_more": False,
                "next_cursor": None,
                "query_used": query,
                "date_filter": {"year_from": year_from, "year_to": year_to},
            },
            "artifacts": {},
            "error": str(exc),
            "retryable": True,
        }


def search_openalex_tool(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, **kwargs: Any) -> dict[str, Any]:
    return _search_tool("search_openalex", openalex, query, year_from, year_to, max_results, **kwargs)


def search_pubmed_tool(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, **kwargs: Any) -> dict[str, Any]:
    return _search_tool("search_pubmed", pubmed, query, year_from, year_to, max_results, **kwargs)


def diagnose_pubmed_query_recall(query: str, year_from: int, year_to: int, max_results: int = 5000) -> dict[str, Any]:
    result = search_pubmed_tool(query, year_from=year_from, year_to=year_to, max_results=max_results, allow_network=True)
    data = result.get("data") or {}
    records = list(data.get("records") or [])
    return {
        "query": query,
        "total_count": data.get("total_count"),
        "retrieved_count": data.get("retrieved_count", len(records)),
        "page_count": data.get("page_count", 0),
        "has_more": data.get("has_more", False),
        "first_5_titles": [record.get("title", "") for record in records[:5]],
        "last_5_titles": [record.get("title", "") for record in records[-5:]],
        "errors": (data.get("source_info") or {}).get("errors") or ([result.get("error")] if result.get("error") else []),
    }


def search_crossref_tool(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, **kwargs: Any) -> dict[str, Any]:
    return _search_tool("search_crossref", crossref, query, year_from, year_to, max_results, **kwargs)


def search_europe_pmc_tool(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, **kwargs: Any) -> dict[str, Any]:
    return _search_tool("search_europe_pmc", europe_pmc, query, year_from, year_to, max_results, **kwargs)


def check_unpaywall_tool(doi: str, **kwargs: Any) -> dict[str, Any]:
    try:
        allow_network = _allow_network(kwargs)
        data = unpaywall.check_doi(
            doi,
            _config(int(kwargs.get("max_results") or 1), **kwargs),
            dry_run=not allow_network,
            allow_network=allow_network,
            session=FAST_METADATA_SESSION if allow_network else None,
            query_id=str(kwargs.get("query_id") or "oa_check"),
        )
        status = "ok" if data.get("is_legal_oa_candidate") else "partial"
        return _result(status, "check_unpaywall", data=data)
    except Exception as exc:
        return _result("error", "check_unpaywall", data={}, error=str(exc), retryable=True)


def check_europe_pmc_oa_tool(record: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    try:
        payload = dict(record or {})
        payload.update({key: value for key, value in kwargs.items() if key in {"doi", "pmid", "pmcid", "title", "source"}})
        payload["source"] = payload.get("source") or "europe_pmc"
        decision = assess_legal_oa_candidate(payload).to_dict()
        status = "ok" if decision.get("is_legal_oa") else "partial"
        return _result(status, "check_europe_pmc_oa", data=decision)
    except Exception as exc:
        return _result("error", "check_europe_pmc_oa", data={}, error=str(exc), retryable=True)


def plan_legal_downloads_tool(records: list[dict[str, Any]], max_downloads: int = 5, **kwargs: Any) -> dict[str, Any]:
    try:
        decisions = []
        requests = []
        for record in records:
            decision = dict(record)
            if decision.get("decision") != "allowed_for_future_download":
                decision = assess_legal_oa_candidate(record).to_dict()
            decisions.append(decision)
            if decision.get("decision") == "allowed_for_future_download":
                requests.append(request_from_legality_decision(decision).to_dict())
        planned = requests[: max_downloads or len(requests)]
        if kwargs.get("write_artifact", False):
            write_legality_audits(decisions)
            for item in planned:
                write_download_plan(item)
        return _result("ok" if planned else "partial", "plan_legal_downloads", data=planned, artifacts={"decisions": decisions, "count": len(planned)})
    except Exception as exc:
        return _result("error", "plan_legal_downloads", data=[], error=str(exc), retryable=True)


def _has_legal_oa_evidence(download_request: dict[str, Any]) -> bool:
    if download_request.get("legality_decision") == "allowed_for_future_download":
        return True
    if download_request.get("decision") == "allowed_for_future_download":
        return True
    if download_request.get("is_legal_oa") is True:
        return True
    if download_request.get("is_legal_oa_candidate") is True and (download_request.get("license") or download_request.get("evidence_sources")):
        return True
    return False


def download_legal_pdf_tool(download_request: dict[str, Any], allow_download: bool = False, yes: bool = False, **kwargs: Any) -> dict[str, Any]:
    if not allow_download:
        error = "allow_download must be True"
        write_failure("download blocked", {"reason": error, "download_request": download_request})
        return _result("blocked", "download_legal_pdf", data={}, error=error, retryable=False)
    if not yes:
        error = "yes must be True for real download"
        write_failure("download blocked", {"reason": error, "download_request": download_request})
        return _result("blocked", "download_legal_pdf", data={}, error=error, retryable=False)
    if not (download_request.get("legal_pdf_url") or download_request.get("pdf_url") or download_request.get("pdf_url_candidate")):
        error = "missing legal_pdf_url"
        write_failure("download blocked", {"reason": error, "download_request": download_request})
        return _result("blocked", "download_legal_pdf", data={}, error=error, retryable=False)
    if not _has_legal_oa_evidence(download_request):
        error = "missing confirmed legal OA evidence"
        write_failure("download blocked", {"reason": error, "download_request": download_request})
        return _result("blocked", "download_legal_pdf", data={}, error=error, retryable=False)
    url_values = [
        str(download_request.get("legal_pdf_url") or ""),
        str(download_request.get("pdf_url") or ""),
        str(download_request.get("pdf_url_candidate") or ""),
        str(download_request.get("landing_url") or ""),
    ]
    if any(is_forbidden_url(url) for url in url_values) or is_forbidden_text(json.dumps(download_request, ensure_ascii=False)):
        error = "forbidden source or bypass text detected"
        write_failure("download blocked", {"reason": error, "download_request": download_request})
        return _result("blocked", "download_legal_pdf", data={}, error=error, retryable=False)
    try:
        payload = dict(download_request)
        if payload.get("decision") == "allowed_for_future_download" and not payload.get("legality_decision"):
            payload["legality_decision"] = "allowed_for_future_download"
        if not payload.get("pdf_url") and payload.get("legal_pdf_url"):
            payload["pdf_url"] = payload.get("legal_pdf_url")
        if not payload.get("pdf_url") and payload.get("pdf_url_candidate"):
            payload["pdf_url"] = payload.get("pdf_url_candidate")
        if kwargs.get("output_dir"):
            payload["output_dir"] = str(kwargs["output_dir"])
        request = DownloadRequest(**{key: value for key, value in payload.items() if key in DownloadRequest.__dataclass_fields__})
        results = execute_download_plan([request], dry_run=False, allow_download=True)
        data = [result.to_dict() for result in results]
        status = "ok" if any(item.get("status") == "downloaded" for item in data) else "partial"
        return _result(status, "download_legal_pdf", data=data)
    except Exception as exc:
        return _result("error", "download_legal_pdf", data={}, error=str(exc), retryable=True)


def write_manifest_tool(records: list[dict[str, Any]] | dict[str, Any], path: str = "manifest.jsonl", **_kwargs: Any) -> dict[str, Any]:
    try:
        payloads = records if isinstance(records, list) else [records]
        for record in payloads:
            write_manifest(record, path=path)
        return _result("ok", "write_manifest", data={"written": len(payloads)}, artifacts={"path": str(Path(path).resolve())})
    except Exception as exc:
        return _result("error", "write_manifest", data={}, error=str(exc), retryable=True)


def write_report_tool(summary: dict[str, Any], output_dir: str = "agent_runs", **kwargs: Any) -> dict[str, Any]:
    try:
        en_path, zh_path = write_reports(summary, output_dir=output_dir, base_dir=kwargs.get("base_dir", "."))
        return _result("ok", "write_report", data={"report_en": str(en_path), "report_zh": str(zh_path)}, artifacts={"report_en": str(en_path.resolve()), "report_zh": str(zh_path.resolve())})
    except Exception as exc:
        return _result("error", "write_report", data={}, error=str(exc), retryable=True)


def read_artifact_tool(path: str, max_lines: int = 80) -> dict[str, Any]:
    try:
        records = read_jsonl(path)
        if records:
            return _result("ok", "read_artifact", data=records[-max_lines:])
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace") if file_path.exists() else ""
        lines = text.splitlines()
        return _result("ok" if text else "partial", "read_artifact", data="\n".join(lines[-max_lines:]))
    except Exception as exc:
        return _result("error", "read_artifact", data="", error=str(exc), retryable=True)


def build_corpus_search_batches_tool(
    queries: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int,
) -> dict[str, Any]:
    try:
        from .batch_runner import build_corpus_search_batches

        batches = build_corpus_search_batches(queries, sources, year_from, year_to, max_results_per_batch)
        return _result("ok", "build_corpus_search_batches", data={"batches": batches, "count": len(batches)})
    except Exception as exc:
        return _result("error", "build_corpus_search_batches", data={}, error=str(exc), retryable=False)


def run_corpus_batch_search_tool(
    queries: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int = 500,
    output_dir: str = "agent_runs/batch_search",
    resume: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    try:
        from .batch_runner import run_corpus_batch_search

        data = run_corpus_batch_search(
            queries,
            sources,
            year_from,
            year_to,
            max_results_per_batch=max_results_per_batch,
            output_dir=output_dir,
            resume=resume,
            retry_failed=retry_failed,
        )
        status = "partial" if data.get("failed_batches") or data.get("partial_batches") else "ok"
        return _result(status, "run_corpus_batch_search", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "run_corpus_batch_search", data={}, error=str(exc), retryable=True)


def merge_batch_outputs_tool(output_dir: str, scope_profile: str = "crispr_broad", include_terms: list[str] | None = None, exclude_terms: list[str] | None = None) -> dict[str, Any]:
    try:
        from .batch_runner import merge_batch_outputs

        data = merge_batch_outputs(output_dir, scope_profile=scope_profile, include_terms=include_terms, exclude_terms=exclude_terms)
        return _result("ok", "merge_batch_outputs", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "merge_batch_outputs", data={}, error=str(exc), retryable=True)


def continue_partial_batches_tool(
    output_dir: str,
    max_additional_results_per_batch: int = 500,
    max_batches: int | None = None,
    sources: list[str] | None = None,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    try:
        from .batch_runner import continue_partial_batches

        data = continue_partial_batches(
            output_dir=output_dir,
            max_additional_results_per_batch=max_additional_results_per_batch,
            max_batches=max_batches,
            sources=sources,
            queries=queries,
        )
        return _result(data.get("status", "ok"), "continue_partial_batches", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "continue_partial_batches", data={}, error=str(exc), retryable=True)


def build_final_corpus_manifest_tool(output_dir: str) -> dict[str, Any]:
    try:
        from .batch_runner import build_final_corpus_manifests

        data = build_final_corpus_manifests(output_dir)
        return _result(data.get("status", "ok"), "build_final_corpus_manifest", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "build_final_corpus_manifest", data={}, error=str(exc), retryable=True)


def export_metadata_only_manifest_tool(output_dir: str) -> dict[str, Any]:
    try:
        from .batch_runner import build_final_corpus_manifests

        data = build_final_corpus_manifests(output_dir)
        artifacts = data.get("artifacts") or {}
        return _result("ok", "export_metadata_only_manifest", data={"count": data.get("metadata_only_manifest_count", 0)}, artifacts={"metadata_only_manifest": artifacts.get("metadata_only_manifest", "")})
    except Exception as exc:
        return _result("error", "export_metadata_only_manifest", data={}, error=str(exc), retryable=True)


def export_downloaded_pdfs_manifest_tool(output_dir: str) -> dict[str, Any]:
    try:
        from .batch_runner import build_final_corpus_manifests

        data = build_final_corpus_manifests(output_dir)
        artifacts = data.get("artifacts") or {}
        return _result("ok", "export_downloaded_pdfs_manifest", data={"count": data.get("downloaded_pdfs_manifest_count", 0)}, artifacts={"downloaded_pdfs_manifest": artifacts.get("downloaded_pdfs_manifest", "")})
    except Exception as exc:
        return _result("error", "export_downloaded_pdfs_manifest", data={}, error=str(exc), retryable=True)


def write_dataset_card_tool(output_dir: str) -> dict[str, Any]:
    try:
        from .batch_runner import build_final_corpus_manifests

        data = build_final_corpus_manifests(output_dir)
        artifacts = data.get("artifacts") or {}
        return _result("ok", "write_dataset_card", data={"dataset_card": artifacts.get("dataset_card", "")}, artifacts={"dataset_card": artifacts.get("dataset_card", "")})
    except Exception as exc:
        return _result("error", "write_dataset_card", data={}, error=str(exc), retryable=True)


def pre_download_qa_for_legal_oa_candidates_tool(output_dir: str) -> dict[str, Any]:
    try:
        from .batch_runner import pre_download_qa_for_legal_oa_candidates

        data = pre_download_qa_for_legal_oa_candidates(output_dir)
        return _result(data.get("status", "ok"), "pre_download_qa_for_legal_oa_candidates", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "pre_download_qa_for_legal_oa_candidates", data={}, error=str(exc), retryable=True)


def download_approved_legal_oa_pdfs_tool(
    output_dir: str,
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    try:
        from .batch_runner import download_approved_legal_oa_pdfs

        data = download_approved_legal_oa_pdfs(
            output_dir=output_dir,
            approved_file=approved_file,
            allow_download=allow_download,
            yes=yes,
            max_downloads=max_downloads,
        )
        return _result(data.get("status", "ok"), "download_approved_legal_oa_pdfs", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "download_approved_legal_oa_pdfs", data={}, error=str(exc), retryable=True)


def collect_approved_legal_fulltexts_tool(
    output_dir: str,
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
    prefer_formats: list[str] | None = None,
) -> dict[str, Any]:
    try:
        from .batch_runner import collect_approved_legal_fulltexts

        data = collect_approved_legal_fulltexts(
            output_dir=output_dir,
            approved_file=approved_file,
            allow_download=allow_download,
            yes=yes,
            max_items=max_items,
            prefer_formats=prefer_formats,
        )
        return _result(data.get("status", "ok"), "collect_approved_legal_fulltexts", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "collect_approved_legal_fulltexts", data={}, error=str(exc), retryable=True)


def retry_failed_legal_oa_downloads_tool(
    output_dir: str,
    failed_file: str = "failed_downloads.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    try:
        from .batch_runner import retry_failed_legal_oa_downloads

        data = retry_failed_legal_oa_downloads(
            output_dir=output_dir,
            failed_file=failed_file,
            allow_download=allow_download,
            yes=yes,
            max_downloads=max_downloads,
        )
        return _result(data.get("status", "ok"), "retry_failed_legal_oa_downloads", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "retry_failed_legal_oa_downloads", data={}, error=str(exc), retryable=True)


def retry_failed_legal_fulltext_fetches_tool(
    output_dir: str,
    failed_file: str = "failed_fulltext_fetches.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    try:
        from .batch_runner import retry_failed_legal_fulltext_fetches

        data = retry_failed_legal_fulltext_fetches(
            output_dir=output_dir,
            failed_file=failed_file,
            allow_download=allow_download,
            yes=yes,
            max_items=max_items,
        )
        return _result(data.get("status", "ok"), "retry_failed_legal_fulltext_fetches", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "retry_failed_legal_fulltext_fetches", data={}, error=str(exc), retryable=True)


def run_full_collection_loop_tool(
    query_list: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int = 500,
    max_additional_results_per_batch: int = 500,
    max_rounds: int = 20,
    max_batches_per_round: int = 20,
    stop_if_no_growth_rounds: int = 2,
    min_new_unique_records_per_round: int = 10,
    retry_failed: bool = False,
    time_budget_seconds: int | None = 3300,
    graceful_stop_buffer_seconds: int = 180,
    resume: bool = True,
    checkpoint_every_batch: bool = True,
    finalize_on_stop: bool = True,
    scope_profile: str = "crispr_broad",
    include_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    output_dir: str = "agent_runs/crispr_broad_full_collection",
) -> dict[str, Any]:
    try:
        from .collection_orchestrator import run_full_collection_loop

        data = run_full_collection_loop(
            query_list=query_list,
            sources=sources,
            year_from=year_from,
            year_to=year_to,
            max_results_per_batch=max_results_per_batch,
            max_additional_results_per_batch=max_additional_results_per_batch,
            max_rounds=max_rounds,
            max_batches_per_round=max_batches_per_round,
            stop_if_no_growth_rounds=stop_if_no_growth_rounds,
            min_new_unique_records_per_round=min_new_unique_records_per_round,
            retry_failed=retry_failed,
            time_budget_seconds=time_budget_seconds,
            graceful_stop_buffer_seconds=graceful_stop_buffer_seconds,
            resume=resume,
            checkpoint_every_batch=checkpoint_every_batch,
            finalize_on_stop=finalize_on_stop,
            scope_profile=scope_profile,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
            output_dir=output_dir,
        )
        return _result(data.get("status", "ok"), "run_full_collection_loop", data=data, artifacts=data.get("artifacts") or {})
    except Exception as exc:
        return _result("error", "run_full_collection_loop", data={}, error=str(exc), retryable=True)


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _norm_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def normalize_record_tool(record: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = dict(record)
        doi = _norm_text(normalized.get("doi")).removeprefix("https://doi.org/")
        normalized["doi"] = doi
        normalized["pmid"] = str(normalized.get("pmid") or "").strip()
        normalized["pmcid"] = str(normalized.get("pmcid") or "").strip().upper()
        normalized["title"] = str(normalized.get("title") or "").strip()
        normalized["title_normalized"] = _norm_title(normalized.get("title"))
        normalized.setdefault("seen_sources", [normalized.get("source")] if normalized.get("source") else [])
        normalized.setdefault("matched_queries", [normalized.get("query") or normalized.get("query_id")] if (normalized.get("query") or normalized.get("query_id")) else [])
        return _result("ok", "normalize_record", data=normalized)
    except Exception as exc:
        return _result("error", "normalize_record", data={}, error=str(exc), retryable=False)


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("doi"):
        return ("doi", str(record["doi"]).lower())
    if record.get("pmid"):
        return ("pmid", str(record["pmid"]))
    if record.get("pmcid"):
        return ("pmcid", str(record["pmcid"]).upper())
    return ("title", _norm_title(record.get("title")))


def deduplicate_records_tool(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        duplicates: list[dict[str, Any]] = []
        for record in records:
            normalized_result = normalize_record_tool(record)
            normalized = normalized_result.get("data") or dict(record)
            key = _record_key(normalized)
            if not key[1]:
                key = ("object", str(len(unique)))
            if key not in unique:
                unique[key] = normalized
                continue
            existing = unique[key]
            existing_sources = set(existing.get("seen_sources") or [])
            existing_sources.update(source for source in [normalized.get("source"), *(normalized.get("seen_sources") or [])] if source)
            existing["seen_sources"] = sorted(existing_sources)
            existing_queries = set(existing.get("matched_queries") or [])
            existing_queries.update(query for query in [normalized.get("query"), normalized.get("query_id"), *(normalized.get("matched_queries") or [])] if query)
            existing["matched_queries"] = sorted(existing_queries)
            for field in ("abstract", "journal", "publication_year", "publication_date", "license", "oa_status", "pdf_url_candidate", "landing_url"):
                if not existing.get(field) and normalized.get(field):
                    existing[field] = normalized[field]
            duplicates.append(normalized)
        data = {"unique_records": list(unique.values()), "duplicates": duplicates}
        return _result("ok", "deduplicate_records", data=data, artifacts={"unique_count": len(unique), "duplicate_count": len(duplicates)})
    except Exception as exc:
        return _result("error", "deduplicate_records", data={"unique_records": [], "duplicates": []}, error=str(exc), retryable=False)


CRISPR_TERMS = ("crispr", "cas12", "cas12a", "cas13", "cas13a", "sherlock", "detectr", "holmes")
DETECTION_TERMS = ("detect", "detection", "diagnostic", "diagnostics", "biosensor", "biosensing", "sensor", "assay", "nucleic acid")
EXCLUDED_TYPES = ("editorial", "comment", "letter")


def topic_guard_records_tool(records: list[dict[str, Any]], include_reviews: bool = False) -> dict[str, Any]:
    try:
        kept: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        excluded_types = EXCLUDED_TYPES if include_reviews else (*EXCLUDED_TYPES, "review")
        for record in records:
            haystack = " ".join(str(record.get(key) or "") for key in ("title", "abstract", "publication_type", "journal", "keywords_matched")).lower()
            pub_type = str(record.get("publication_type") or "").lower()
            if any(kind in pub_type for kind in excluded_types):
                item = dict(record)
                item["excluded_reason"] = "excluded_publication_type"
                excluded.append(item)
                continue
            has_crispr = any(term in haystack for term in CRISPR_TERMS)
            has_detection = any(term in haystack for term in DETECTION_TERMS)
            if has_crispr and has_detection:
                kept.append(record)
            else:
                item = dict(record)
                item["excluded_reason"] = "topic_guard_failed"
                excluded.append(item)
        return _result("ok", "topic_guard_records", data={"topic_records": kept, "excluded_records": excluded}, artifacts={"kept_count": len(kept), "excluded_count": len(excluded)})
    except Exception as exc:
        return _result("error", "topic_guard_records", data={"topic_records": [], "excluded_records": []}, error=str(exc), retryable=False)


def scope_guard_records_tool(
    records: list[dict[str, Any]],
    mode: str = "crispr_broad_corpus",
    include_reviews: bool = True,
    conservative: bool = True,
) -> dict[str, Any]:
    try:
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        manual: list[dict[str, Any]] = []
        exclude_reasons: dict[str, int] = {}
        include_reasons: dict[str, int] = {}
        for record in records:
            scoped = dict(record)
            scoped.update(evaluate_crispr_scope(scoped, include_reviews=include_reviews, conservative=conservative))
            reason = str(scoped.get("scope_reason") or "unknown")
            if scoped["scope_status"] == "included":
                included.append(scoped)
                include_reasons[reason] = include_reasons.get(reason, 0) + 1
            elif scoped["scope_status"] == "excluded":
                excluded.append(scoped)
                exclude_reasons[reason] = exclude_reasons.get(reason, 0) + 1
            else:
                manual.append(scoped)
                include_reasons[reason] = include_reasons.get(reason, 0) + 1
        data = {
            "included_records": included,
            "excluded_records": excluded,
            "needs_manual_scope_review": manual,
            "stats": {
                "input_count": len(records),
                "included_count": len(included),
                "excluded_count": len(excluded),
                "manual_review_count": len(manual),
                "exclude_reasons": exclude_reasons,
                "include_reasons": include_reasons,
                "mode": mode,
            },
        }
        return _result("ok", "scope_guard_records_tool", data=data)
    except Exception as exc:
        return _result("error", "scope_guard_records_tool", data={}, error=str(exc), retryable=False)


def _oa_schema(record: dict[str, Any], evidence_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    evidence_records = evidence_records or []
    merged = dict(record)
    for evidence in evidence_records:
        for key, value in evidence.items():
            if value not in (None, "", []) and not merged.get(key):
                merged[key] = value
    decision = assess_legal_oa_candidate(merged).to_dict()
    legal_pdf_url = str(merged.get("legal_pdf_url") or merged.get("pdf_url_candidate") or merged.get("url_for_pdf") or decision.get("pdf_url_candidate") or "")
    evidence_summary = decision.get("evidence_summary") or merged.get("evidence") or merged.get("oa_evidence") or merged.get("reason") or ""
    is_oa = bool(decision.get("is_legal_oa") or merged.get("is_legal_oa_candidate") or merged.get("unpaywall_is_oa"))
    can_download = bool(is_oa and legal_pdf_url and evidence_summary and not decision.get("is_forbidden"))
    return {
        **merged,
        "is_oa": is_oa,
        "legal_pdf_url": legal_pdf_url,
        "license": merged.get("license") or decision.get("license") or "",
        "oa_source": merged.get("source") or decision.get("source") or "",
        "evidence": evidence_summary,
        "can_download": can_download,
        "needs_manual_review": not can_download,
        "decision": "allowed_for_future_download" if can_download else decision.get("decision", "candidate_needs_confirmation"),
        "legality_decision": "allowed_for_future_download" if can_download else decision.get("decision", ""),
        "failure_reason": "" if can_download else decision.get("reason", "insufficient_legal_oa_evidence"),
    }


def audit_oa_records_tool(records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    try:
        audited: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        unpaywall_cache: dict[str, dict[str, Any]] = {}
        for record in records:
            evidences: list[dict[str, Any]] = []
            doi = str(record.get("doi") or "")
            if doi:
                cache_key = doi.lower()
                if cache_key in unpaywall_cache:
                    unpaywall_result = unpaywall_cache[cache_key]
                else:
                    unpaywall_result = check_unpaywall_tool(doi, allow_network=_allow_network(kwargs), query_id=record.get("query_id") or "oa_audit")
                    unpaywall_cache[cache_key] = unpaywall_result
                if unpaywall_result.get("status") in {"ok", "partial"}:
                    evidences.append(unpaywall_result.get("data") or {})
                else:
                    failures.append({"doi": doi, "reason": unpaywall_result.get("error") or "unpaywall_check_failed"})
            europe_result = check_europe_pmc_oa_tool(record)
            if europe_result.get("status") in {"ok", "partial"}:
                evidences.append(europe_result.get("data") or {})
            audited.append(_oa_schema(record, evidences))
        return _result("ok", "audit_oa_records", data={"oa_audit_records": audited, "failures": failures}, artifacts={"audited_count": len(audited), "failure_count": len(failures)})
    except Exception as exc:
        return _result("error", "audit_oa_records", data={"oa_audit_records": [], "failures": []}, error=str(exc), retryable=True)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(path, record)


def export_corpus_manifest_tool(state_or_records: dict[str, Any], output_dir: str) -> dict[str, Any]:
    try:
        out = Path(output_dir)
        records = list(state_or_records.get("oa_audit_records") or state_or_records.get("topic_records") or state_or_records.get("records") or [])
        download_results = state_or_records.get("download_results") or []
        pdf_by_doi: dict[str, dict[str, Any]] = {}
        for item in download_results:
            data = item.get("data") if isinstance(item, dict) else None
            rows = data if isinstance(data, list) else []
            for row in rows:
                pdf_by_doi[str(row.get("doi") or "").lower()] = row
        corpus: list[dict[str, Any]] = []
        for idx, record in enumerate(records, 1):
            doi = str(record.get("doi") or "").lower()
            downloaded = pdf_by_doi.get(doi, {})
            sources = record.get("seen_sources") or record.get("sources") or ([record.get("source")] if record.get("source") else [])
            corpus.append(
                {
                    "record_id": record.get("record_id") or doi or record.get("pmid") or record.get("pmcid") or f"record_{idx}",
                    "doi": record.get("doi", ""),
                    "pmid": record.get("pmid", ""),
                    "pmcid": record.get("pmcid", ""),
                    "title": record.get("title", ""),
                    "journal": record.get("journal", ""),
                    "year": record.get("publication_year") or record.get("year"),
                    "abstract": record.get("abstract", ""),
                    "keywords": record.get("keywords") or record.get("keywords_matched") or [],
                    "matched_queries": record.get("matched_queries") or [],
                    "sources": sources,
                    "is_oa": bool(record.get("is_oa")),
                    "license": record.get("license", ""),
                    "pdf_path": downloaded.get("local_path", ""),
                    "download_status": downloaded.get("status") or ("not_requested" if not record.get("can_download") else "planned"),
                    "failure_reason": record.get("failure_reason", ""),
                    "text_extraction_status": "not_started",
                }
            )
        failure_rows = list(state_or_records.get("failures") or [])
        for record in state_or_records.get("excluded_records") or []:
            failure_rows.append(
                {
                    "stage": "topic_guard",
                    "doi": record.get("doi", ""),
                    "pmid": record.get("pmid", ""),
                    "pmcid": record.get("pmcid", ""),
                    "title": record.get("title", ""),
                    "failure_reason": record.get("excluded_reason") or "topic_excluded",
                }
            )
        for record in records:
            if record.get("can_download"):
                continue
            reason = record.get("failure_reason") or ""
            if not reason:
                if not record.get("doi"):
                    reason = "missing_doi"
                elif not record.get("is_oa"):
                    reason = "not_confirmed_oa"
                elif not record.get("legal_pdf_url"):
                    reason = "missing_legal_pdf_url"
                elif not record.get("evidence"):
                    reason = "missing_oa_evidence"
                else:
                    reason = "needs_manual_review"
            failure_rows.append(
                {
                    "stage": "oa_audit",
                    "doi": record.get("doi", ""),
                    "pmid": record.get("pmid", ""),
                    "pmcid": record.get("pmcid", ""),
                    "title": record.get("title", ""),
                    "failure_reason": reason,
                }
            )
        artifacts = {
            "corpus_manifest": str((out / "corpus_manifest.jsonl").resolve()),
            "metadata_all": str((out / "metadata_all.jsonl").resolve()),
            "oa_audit": str((out / "oa_audit.jsonl").resolve()),
            "download_manifest": str((out / "download_manifest.jsonl").resolve()),
            "failures": str((out / "failures.jsonl").resolve()),
            "excluded_records": str((out / "excluded_records.jsonl").resolve()),
        }
        _write_jsonl(out / "corpus_manifest.jsonl", corpus)
        _write_jsonl(out / "metadata_all.jsonl", list(state_or_records.get("unique_records") or state_or_records.get("raw_records") or []))
        _write_jsonl(out / "oa_audit.jsonl", records)
        _write_jsonl(out / "download_manifest.jsonl", list(state_or_records.get("download_plan") or []))
        _write_jsonl(out / "failures.jsonl", failure_rows)
        _write_jsonl(out / "excluded_records.jsonl", list(state_or_records.get("excluded_records") or []))
        return _result("ok", "export_corpus_manifest", data={"corpus_manifest": corpus}, artifacts=artifacts)
    except Exception as exc:
        return _result("error", "export_corpus_manifest", data={"corpus_manifest": []}, error=str(exc), retryable=True)


def write_coverage_report_tool(state_or_summary: dict[str, Any], output_dir: str) -> dict[str, Any]:
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        coverage = state_or_summary.get("coverage") or {}
        search_log = list(state_or_summary.get("search_log") or [])
        broad_rows = [row for row in search_log if str(row.get("query") or "").startswith("(") or "CRISPR-Cas" in str(row.get("query") or "")]
        broad_total = sum(int(row.get("total_count") or 0) for row in broad_rows if row.get("total_count") not in (None, ""))
        broad_retrieved = sum(int(row.get("retrieved_count") or row.get("result_count") or 0) for row in broad_rows)
        lines = [
            "# Coverage Report",
            "",
            f"- Query count: {len(state_or_summary.get('query_list') or [])}",
            f"- Raw records: {len(state_or_summary.get('raw_records') or [])}",
            f"- Unique records: {len(state_or_summary.get('unique_records') or [])}",
            f"- Topic records: {len(state_or_summary.get('topic_records') or [])}",
            f"- Broad query total_count: {broad_total if broad_rows else 'not_available'}",
            f"- Broad query retrieved_count: {broad_retrieved if broad_rows else 'not_available'}",
            f"- Included scope count: {len(state_or_summary.get('scope_included_records') or [])}",
            f"- Excluded pure editing count: {len(state_or_summary.get('scope_excluded_records') or [])}",
            f"- Manual scope review count: {len(state_or_summary.get('needs_manual_scope_review') or [])}",
            f"- OA audited records: {len(state_or_summary.get('oa_audit_records') or [])}",
            f"- Planned downloads: {len(state_or_summary.get('download_plan') or [])}",
            f"- Failures: {len(state_or_summary.get('failures') or [])}",
            f"- Coverage status: {coverage.get('status', 'unknown')}",
            f"- Needs more search: {coverage.get('needs_more_search', False)}",
            f"- Needs more OA audit: {coverage.get('needs_more_oa_audit', False)}",
        ]
        path = out / "coverage_report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _result("ok", "write_coverage_report", data={"coverage_report": str(path)}, artifacts={"coverage_report": str(path.resolve())})
    except Exception as exc:
        return _result("error", "write_coverage_report", data={}, error=str(exc), retryable=True)


def write_recall_diagnostics_tool(state_or_summary: dict[str, Any], output_dir: str) -> dict[str, Any]:
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        search_log = list(state_or_summary.get("search_log") or [])
        manual_counts = dict(state_or_summary.get("manual_reference_counts") or {})
        diagnostics: list[dict[str, Any]] = []
        for item in search_log:
            total = item.get("total_count")
            retrieved = int(item.get("retrieved_count") or item.get("result_count") or 0)
            ratio = None if total in (None, 0, "") else retrieved / int(total)
            reason = ""
            if item.get("status") == "error":
                reason = "source_error"
            elif item.get("has_more") or (total not in (None, "") and retrieved < int(total or 0)):
                reason = "pagination_or_limit_insufficient"
            diagnostics.append({**item, "retrieval_ratio": ratio, "reason": reason})
        agent_retrieved = {
            "raw_records": len(state_or_summary.get("raw_records") or []),
            "unique_records": len(state_or_summary.get("unique_records") or []),
            "topic_records": len(state_or_summary.get("topic_records") or []),
            "oa_audit_records": len(state_or_summary.get("oa_audit_records") or []),
        }
        payload = {
            "manual_reference_counts": manual_counts,
            "agent_retrieved_counts": agent_retrieved,
            "source_query_diagnostics": diagnostics,
            "coverage": state_or_summary.get("coverage") or {},
        }
        json_path = out / "recall_diagnostics.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# Recall Diagnostics Report",
            "",
            "## Manual Reference Counts",
            f"- PubMed manual reference count: {manual_counts.get('pubmed_reference_count', 'not_provided')}",
            f"- Google Scholar manual reference count: {manual_counts.get('google_scholar_reference_count', 'not_provided')}",
            "",
            "## Agent Retrieved Counts",
            *(f"- {key}: {value}" for key, value in agent_retrieved.items()),
            "",
            "## Gap Analysis",
            f"- PubMed gap: manual {manual_counts.get('pubmed_reference_count', 'not_provided')} vs agent raw {agent_retrieved['raw_records']}",
            f"- Google Scholar is manual reference only and is not scraped or used as an automated source.",
            "",
            "## Source Query Diagnostics",
        ]
        for item in diagnostics:
            lines.append(
                f"- {item.get('source')} | {item.get('query')}: total={item.get('total_count')} retrieved={item.get('retrieved_count')} "
                f"pages={item.get('page_count')} has_more={item.get('has_more')} reason={item.get('reason') or 'ok'}"
            )
        report_path = out / "recall_diagnostics_report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _result("ok", "write_recall_diagnostics", data=payload, artifacts={"recall_diagnostics": str(json_path.resolve()), "recall_diagnostics_report": str(report_path.resolve())})
    except Exception as exc:
        return _result("error", "write_recall_diagnostics", data={}, error=str(exc), retryable=True)


def write_topic_guard_diagnostics_tool(state_or_summary: dict[str, Any], output_dir: str, sample_size: int = 50) -> dict[str, Any]:
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        excluded = list(state_or_summary.get("excluded_records") or [])
        reason_counts: dict[str, int] = {}
        rows: list[dict[str, Any]] = []
        for record in excluded:
            reason = str(record.get("excluded_reason") or record.get("failure_reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            haystack = " ".join(str(record.get(key) or "") for key in ("title", "abstract")).lower()
            possible_false_negative = any(term in haystack for term in CRISPR_TERMS) and any(term in haystack for term in DETECTION_TERMS)
            rows.append(
                {
                    "title": record.get("title", ""),
                    "abstract_snippet": str(record.get("abstract") or "")[:300],
                    "matched_queries": record.get("matched_queries") or [],
                    "exclusion_reason": reason,
                    "seen_sources": record.get("seen_sources") or record.get("sources") or [],
                    "possible_false_negative": possible_false_negative,
                }
            )
        sampled = rows[:sample_size]
        jsonl_path = out / "topic_guard_diagnostics.jsonl"
        _write_jsonl(jsonl_path, sampled)
        lines = [
            "# Topic Guard Diagnostics Report",
            "",
            "## Exclusion Reasons",
            *(f"- {reason}: {count}" for reason, count in sorted(reason_counts.items())),
            "",
            f"## Sampled Excluded Records ({len(sampled)})",
        ]
        for row in sampled:
            flag = " possible_false_negative" if row["possible_false_negative"] else ""
            lines.append(f"- {row['title']} [{row['exclusion_reason']}]{flag}")
        report_path = out / "topic_guard_diagnostics_report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _result("ok", "write_topic_guard_diagnostics", data={"reason_counts": reason_counts, "sample": sampled}, artifacts={"topic_guard_diagnostics": str(jsonl_path.resolve()), "topic_guard_diagnostics_report": str(report_path.resolve())})
    except Exception as exc:
        return _result("error", "write_topic_guard_diagnostics", data={}, error=str(exc), retryable=True)


def write_scope_guard_outputs_tool(state_or_summary: dict[str, Any], output_dir: str) -> dict[str, Any]:
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        included = list(state_or_summary.get("scope_included_records") or [])
        excluded = list(state_or_summary.get("scope_excluded_records") or state_or_summary.get("excluded_records") or [])
        manual = list(state_or_summary.get("needs_manual_scope_review") or [])
        stats = dict(state_or_summary.get("scope_stats") or {})
        included_path = out / "scope_included_records.jsonl"
        excluded_path = out / "scope_excluded_records.jsonl"
        manual_path = out / "needs_manual_scope_review.jsonl"
        _write_jsonl(included_path, included)
        _write_jsonl(excluded_path, excluded)
        _write_jsonl(manual_path, manual)
        lines = [
            "# Scope Guard Report",
            "",
            f"- Input count: {stats.get('input_count', len(included) + len(excluded) + len(manual))}",
            f"- Included count: {stats.get('included_count', len(included))}",
            f"- Excluded pure editing count: {stats.get('excluded_count', len(excluded))}",
            f"- Manual scope review count: {stats.get('manual_review_count', len(manual))}",
            "",
            "## Include Reasons",
        ]
        for reason, count in sorted((stats.get("include_reasons") or {}).items()):
            lines.append(f"- {reason}: {count}")
        lines.append("")
        lines.append("## Exclude Reasons")
        for reason, count in sorted((stats.get("exclude_reasons") or {}).items()):
            lines.append(f"- {reason}: {count}")
        report_path = out / "scope_guard_report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _result(
            "ok",
            "write_scope_guard_outputs",
            data={"scope_guard_report": str(report_path)},
            artifacts={
                "scope_included_records": str(included_path.resolve()),
                "scope_excluded_records": str(excluded_path.resolve()),
                "needs_manual_scope_review": str(manual_path.resolve()),
                "scope_guard_report": str(report_path.resolve()),
            },
        )
    except Exception as exc:
        return _result("error", "write_scope_guard_outputs", data={}, error=str(exc), retryable=True)


def calibrate_openalex_oa_counts_tool(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    output_dir: str = "agent_runs/openalex_oa_calibration",
) -> dict[str, Any]:
    try:
        from .openalex_calibration import calibrate_openalex_oa_counts

        data = calibrate_openalex_oa_counts(
            query=query,
            year_from=year_from,
            year_to=year_to,
            output_dir=output_dir,
        )
        status = data.get("status", "ok")
        return _result(
            status,
            "calibrate_openalex_oa_counts",
            data=data,
            artifacts=data.get("artifacts") or {},
            error=str(data.get("failure_reason") or ""),
            retryable=status == "error",
        )
    except Exception as exc:
        return _result("error", "calibrate_openalex_oa_counts", data={}, error=str(exc), retryable=True)
