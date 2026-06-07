"""LangGraph主体 for the legal literature agent."""

from __future__ import annotations

from typing import Any
import re
from pathlib import Path
import shutil

from langgraph.graph import END, StateGraph

from .agent_state import LiteratureAgentState
from .agent_task import CRISPR_SEARCH_KEYWORDS
from .acquisition_request import (
    LEGAL_FULLTEXT_FORMATS,
    LiteratureAcquisitionRequest,
    parse_acquisition_request_with_llm,
    write_acquisition_plan,
)
from .count_summary import summarize_literature_counts
from .crispr_scope import BROAD_CRISPR_CORPUS_QUERIES, PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY
from .llm_nodes import diagnose_with_llm, understand_request_with_llm
from . import tool_registry as tr
from .collection_orchestrator import _write_full_collection_report, check_collection_completion, plan_collection_batches, run_batch_scheduler
from .manifest import append_jsonl, read_jsonl


CRISPR_CORPUS_QUERIES = [
    "CRISPR detection",
    "CRISPR diagnostics",
    "CRISPR-based detection",
    "CRISPR-based diagnostics",
    "CRISPR biosensor",
    "CRISPR biosensing",
    "CRISPR sensor",
    "CRISPR assay",
    "Cas12 detection",
    "Cas12a detection",
    "Cas13 detection",
    "Cas13a detection",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
    "CRISPR nucleic acid detection",
    "CRISPR molecular diagnostics",
    "CRISPR point-of-care detection",
]


def initialize_state_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    result.setdefault("options", {})
    result.setdefault("metadata_results", [])
    result.setdefault("oa_results", [])
    result.setdefault("download_plan", [])
    result.setdefault("download_results", [])
    result.setdefault("artifacts", {})
    result.setdefault("reports", {})
    result.setdefault("failures", [])
    result.setdefault("messages", [])
    result.setdefault("diagnostics", {})
    result.setdefault("blocked", False)
    result.setdefault("finished", False)
    options = result.setdefault("options", {})
    result.setdefault("query_list", [])
    result.setdefault("search_log", [])
    result.setdefault("raw_records", [])
    result.setdefault("unique_records", [])
    result.setdefault("excluded_records", [])
    result.setdefault("scope_included_records", [])
    result.setdefault("scope_excluded_records", [])
    result.setdefault("needs_manual_scope_review", [])
    result.setdefault("scope_stats", {})
    result.setdefault("topic_records", [])
    result.setdefault("oa_audit_records", [])
    result.setdefault("corpus_manifest", [])
    result.setdefault("coverage", {})
    result.setdefault("search_attempt_count", 0)
    result.setdefault("oa_attempt_count", 0)
    result.setdefault("max_search_attempts", int(options.get("max_search_attempts") or (2 if options.get("exhaustive_mode") else 1)))
    result.setdefault("max_oa_attempts", int(options.get("max_oa_attempts") or 1))
    result.setdefault("year_from", options.get("year_from"))
    result.setdefault("year_to", options.get("year_to"))
    result.setdefault("max_results_per_query", int(options.get("max_results_per_query") or options.get("max_results") or 30))
    result.setdefault("exhaustive_mode", bool(options.get("exhaustive_mode", False)))
    return result


def understand_request_node(state: LiteratureAgentState) -> LiteratureAgentState:
    return understand_request_with_llm(state)


def route_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    if result.get("finished"):
        route = "finish"
    elif not result.get("task"):
        route = "report"
    else:
        route = "search_metadata"
    result["route"] = route
    result["next_action"] = route
    return result


def _route_after_route_node(state: LiteratureAgentState) -> str:
    return str(state.get("route") or "report")


def query_expansion_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    task = result.get("task") or {}
    options = result.get("options") or {}
    user_request = result.get("user_request", "")
    mode = str(options.get("mode") or task.get("mode") or "crispr_detection_corpus")
    base_queries = list(BROAD_CRISPR_CORPUS_QUERIES if mode == "crispr_broad_corpus" else CRISPR_CORPUS_QUERIES)
    for keyword in CRISPR_SEARCH_KEYWORDS:
        if keyword not in base_queries:
            base_queries.append(keyword)
    for keyword in task.get("keywords") or []:
        text = str(keyword).strip()
        if text and text not in base_queries:
            base_queries.append(text)
    query_list = []
    seen = set()
    for query in base_queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            query_list.append(query)
    result["query_list"] = query_list
    messages = list(result.get("messages") or [])
    messages.append(
        {
            "node": "query_expansion_node",
            "method": "rules",
            "mode": mode,
            "source": "BROAD_CRISPR_CORPUS_QUERIES + task.keywords" if mode == "crispr_broad_corpus" else "CRISPR_CORPUS_QUERIES + task.keywords",
            "query_count": len(query_list),
            "request": user_request,
        }
    )
    result["messages"] = messages
    return result


def multi_source_search_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    query_list = list(result.get("query_list") or [result.get("user_request", "")])
    year_from = result.get("year_from") or options.get("year_from") or (result.get("task") or {}).get("year_from")
    year_to = result.get("year_to") or options.get("year_to") or (result.get("task") or {}).get("year_to")
    max_results = int(result.get("max_results_per_query") or options.get("max_results_per_query") or options.get("max_results") or 30)
    recall_only = bool(options.get("recall_calibration_only", False))
    broad_probe = bool(options.get("mode") == "crispr_broad_corpus" and not options.get("download") and int(options.get("max_downloads") or 0) <= 0)
    allow_network = bool(options.get("allow_network_metadata", True))
    sources = {
        "openalex": tr.search_openalex_tool,
        "pubmed": tr.search_pubmed_tool,
        "crossref": tr.search_crossref_tool,
        "europe_pmc": tr.search_europe_pmc_tool,
    }
    if recall_only:
        sources = {"pubmed": tr.search_pubmed_tool}
    raw_records = list(result.get("raw_records") or [])
    search_log = list(result.get("search_log") or [])
    failures = list(result.get("failures") or [])
    attempt = int(result.get("search_attempt_count") or 0) + 1
    active_queries = ["CRISPR detection"] if recall_only else (query_list[:5] if broad_probe else query_list)
    for query in active_queries:
        for source, tool in sources.items():
            source_max_results = max_results
            if recall_only and not (source == "pubmed" and query == "CRISPR detection"):
                source_max_results = 1
            if broad_probe:
                source_max_results = min(source_max_results, 50)
            tool_result = tool(
                query,
                year_from=year_from,
                year_to=year_to,
                max_results=source_max_results,
                allow_network=allow_network,
                query_id=f"{re.sub(r'[^A-Za-z0-9]+', '_', query).strip('_').lower()}_{source}",
                keywords=[query],
            )
            payload = tool_result.get("data") or {}
            data = list(payload.get("records") if isinstance(payload, dict) else payload or [])
            search_log.append(
                {
                    "attempt": attempt,
                    "query": query,
                    "source": source,
                    "status": tool_result.get("status"),
                    "result_count": len(data),
                    "total_count": payload.get("total_count") if isinstance(payload, dict) else None,
                    "retrieved_count": payload.get("retrieved_count", len(data)) if isinstance(payload, dict) else len(data),
                    "page_count": payload.get("page_count", 0) if isinstance(payload, dict) else 0,
                    "has_more": bool(payload.get("has_more", False)) if isinstance(payload, dict) else False,
                    "next_cursor": payload.get("next_cursor") if isinstance(payload, dict) else None,
                    "max_results_per_query": source_max_results,
                    "page": 1,
                    "pagination_supported": True,
                    "error": tool_result.get("error", ""),
                }
            )
            if tool_result.get("status") in {"ok", "partial"}:
                for record in data:
                    item = dict(record)
                    item["source"] = item.get("source") or source
                    item["matched_queries"] = sorted(set([*(item.get("matched_queries") or []), query]))
                    item["seen_sources"] = sorted(set([*(item.get("seen_sources") or []), source]))
                    raw_records.append(item)
            else:
                failures.append({"stage": "multi_source_search_node", "query": query, "source": source, "error": tool_result.get("error")})
    result["raw_records"] = raw_records
    result["metadata_results"] = raw_records
    result["search_log"] = search_log
    result["failures"] = failures
    result["search_attempt_count"] = attempt
    return result


def deduplicate_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    tool_result = tr.deduplicate_records_tool(list(result.get("raw_records") or []))
    if tool_result.get("status") == "ok":
        data = tool_result.get("data") or {}
        result["unique_records"] = list(data.get("unique_records") or [])
        result["metadata_results"] = list(data.get("unique_records") or [])
    else:
        failures = list(result.get("failures") or [])
        failures.append({"stage": "deduplicate_node", "error": tool_result.get("error")})
        result["failures"] = failures
    return result


RECALL_CALIBRATION_QUERIES = [
    "CRISPR detection",
    "CRISPR diagnostics",
    "CRISPR-based detection",
    "Cas12 detection",
    "Cas12a detection",
    "Cas13 detection",
    "Cas13a detection",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
]


def recall_calibration_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    diagnostics: list[dict[str, Any]] = []
    for item in result.get("search_log") or []:
        if item.get("query") not in RECALL_CALIBRATION_QUERIES:
            continue
        total = item.get("total_count")
        retrieved = int(item.get("retrieved_count") or item.get("result_count") or 0)
        retrieval_ratio = None if total in (None, 0, "") else retrieved / int(total)
        reason = ""
        if item.get("status") == "error":
            reason = "source_error"
        elif item.get("has_more") or (total not in (None, "") and retrieved < int(total or 0)):
            reason = "pagination_or_limit_insufficient"
        diagnostics.append({**item, "retrieval_ratio": retrieval_ratio, "reason": reason})
    coverage = dict(result.get("coverage") or {})
    if any(item.get("reason") == "pagination_or_limit_insufficient" for item in diagnostics):
        coverage["needs_more_search"] = True
        coverage["status"] = "partial"
        coverage["reason"] = "pagination_or_limit_insufficient"
    result["coverage"] = coverage
    result["recall_diagnostics"] = diagnostics
    return result


def topic_guard_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    mode = str(options.get("mode") or "crispr_detection_corpus")
    if mode == "crispr_broad_corpus":
        tool_result = tr.scope_guard_records_tool(
            list(result.get("unique_records") or []),
            mode=mode,
            include_reviews=bool(options.get("include_reviews", True)),
            conservative=bool(options.get("conservative_scope_guard", True)),
        )
        if tool_result.get("status") == "ok":
            data = tool_result.get("data") or {}
            included = list(data.get("included_records") or [])
            manual = list(data.get("needs_manual_scope_review") or [])
            excluded = list(data.get("excluded_records") or [])
            result["scope_included_records"] = included
            result["needs_manual_scope_review"] = manual
            result["scope_excluded_records"] = excluded
            result["excluded_records"] = excluded
            result["scope_stats"] = dict(data.get("stats") or {})
            result["topic_records"] = included + manual
            result["metadata_results"] = included + manual
        else:
            failures = list(result.get("failures") or [])
            failures.append({"stage": "topic_guard_node", "tool": "scope_guard_records_tool", "error": tool_result.get("error")})
            result["failures"] = failures
        return result
    tool_result = tr.topic_guard_records_tool(list(result.get("unique_records") or []), include_reviews=bool(options.get("include_reviews", False)))
    if tool_result.get("status") == "ok":
        data = tool_result.get("data") or {}
        result["topic_records"] = list(data.get("topic_records") or [])
        result["excluded_records"] = list(data.get("excluded_records") or [])
        result["metadata_results"] = list(data.get("topic_records") or [])
    else:
        failures = list(result.get("failures") or [])
        failures.append({"stage": "topic_guard_node", "error": tool_result.get("error")})
        result["failures"] = failures
    return result


def oa_audit_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    if options.get("recall_calibration_only"):
        result["oa_audit_records"] = []
        result["oa_results"] = []
        return result
    tool_result = tr.audit_oa_records_tool(list(result.get("topic_records") or []), allow_network=bool(options.get("allow_network_metadata", True)))
    failures = list(result.get("failures") or [])
    if tool_result.get("status") in {"ok", "partial"}:
        data = tool_result.get("data") or {}
        result["oa_audit_records"] = list(data.get("oa_audit_records") or [])
        result["oa_results"] = list(data.get("oa_audit_records") or [])
        failures.extend({"stage": "oa_audit_node", **item} for item in data.get("failures") or [])
    else:
        failures.append({"stage": "oa_audit_node", "error": tool_result.get("error")})
    result["failures"] = failures
    result["oa_attempt_count"] = int(result.get("oa_attempt_count") or 0) + 1
    return result


def search_metadata_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    task = result.get("task") or {}
    intent = result.get("search_intent") or {}
    options = result.get("options") or {}
    query = str(intent.get("query") or task.get("topic") or result.get("user_request") or "")
    sources = list(task.get("sources") or intent.get("sources") or ["openalex", "crossref", "europe_pmc", "pubmed"])
    year_from = task.get("year_from") or intent.get("year_from")
    year_to = task.get("year_to") or intent.get("year_to")
    max_results = int(options.get("max_results") or task.get("max_results") or intent.get("max_results") or 30)
    common = {
        "allow_network": bool(options.get("allow_network_metadata", True)),
        "keywords": task.get("keywords") or intent.get("keywords") or [query],
        "include_terms": task.get("include_terms") or [],
        "exclude_terms": task.get("exclude_terms") or [],
        "publication_date_from": task.get("publication_date_from") or "",
        "publication_date_to": task.get("publication_date_to") or "",
    }
    tool_map = {
        "openalex": tr.search_openalex_tool,
        "pubmed": tr.search_pubmed_tool,
        "crossref": tr.search_crossref_tool,
        "europe_pmc": tr.search_europe_pmc_tool,
    }
    metadata: list[dict[str, Any]] = []
    failures = list(result.get("failures") or [])
    messages = list(result.get("messages") or [])
    for source in sources:
        tool = tool_map.get(str(source))
        if tool is None:
            continue
        tool_result = tool(query, year_from=year_from, year_to=year_to, max_results=max_results, **common)
        messages.append({"node": "search_metadata_node", "tool": tool_result.get("tool"), "status": tool_result.get("status")})
        if tool_result.get("status") in {"ok", "partial"}:
            metadata.extend(list(tool_result.get("data") or []))
        else:
            failures.append({"stage": "search_metadata_node", "tool": tool_result.get("tool"), "error": tool_result.get("error")})
    result["metadata_results"] = metadata
    result["failures"] = failures
    result["messages"] = messages
    return result


def audit_oa_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    oa_results: list[dict[str, Any]] = []
    failures = list(result.get("failures") or [])
    for record in result.get("metadata_results") or []:
        doi = str(record.get("doi") or "")
        if doi:
            checked = tr.check_unpaywall_tool(doi, allow_network=bool(options.get("allow_network_metadata", True)), query_id=record.get("query_id") or "oa_check")
            if checked.get("status") in {"ok", "partial"}:
                merged = dict(record)
                merged.update(checked.get("data") or {})
                oa_results.append(merged)
            else:
                failures.append({"stage": "audit_oa_node", "tool": "check_unpaywall", "doi": doi, "error": checked.get("error")})
        if record.get("pmcid") or str(record.get("source") or "") == "europe_pmc":
            europe = tr.check_europe_pmc_oa_tool(record)
            if europe.get("status") in {"ok", "partial"}:
                oa_results.append(europe.get("data") or {})
            else:
                failures.append({"stage": "audit_oa_node", "tool": "check_europe_pmc_oa", "error": europe.get("error")})
    result["oa_results"] = oa_results
    result["failures"] = failures
    return result


def plan_download_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    if options.get("recall_calibration_only") or int(options.get("max_downloads") or 0) <= 0:
        result["download_plan"] = []
        return result
    records = [record for record in list(result.get("oa_audit_records") or []) if record.get("can_download") is True]
    max_downloads = int(options.get("max_downloads") or 5)
    plan: list[dict[str, Any]] = []
    for record in records[:max_downloads]:
        doi = str(record.get("doi") or "")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", doi or record.get("title") or "paper").strip("._")[:80] or "paper"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        plan.append(
            {
                "doi": doi,
                "title": record.get("title", ""),
                "pdf_url": record.get("legal_pdf_url", ""),
                "legal_pdf_url": record.get("legal_pdf_url", ""),
                "oa_source": record.get("oa_source", ""),
                "license": record.get("license", ""),
                "evidence": record.get("evidence", ""),
                "target_filename": safe_name,
                "planned_status": "planned_legal_oa_download",
                "legality_decision": "allowed_for_future_download",
                "decision": "allowed_for_future_download",
                "source": record.get("oa_source") or record.get("source") or "",
                "evidence_sources": [record.get("oa_source") or record.get("source") or "oa_audit"],
            }
        )
    if plan or result.get("oa_audit_records") is not None:
        result["download_plan"] = plan
        return result
    records = list(result.get("oa_results") or [])
    if not records:
        records = list(result.get("metadata_results") or [])
    planned = tr.plan_legal_downloads_tool(records, max_downloads=max_downloads)
    if planned.get("status") in {"ok", "partial"}:
        result["download_plan"] = list(planned.get("data") or [])
    else:
        failures = list(result.get("failures") or [])
        failures.append({"stage": "plan_download_node", "tool": "plan_legal_downloads", "error": planned.get("error")})
        result["failures"] = failures
    return result


def coverage_check_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    query_count = 1 if options.get("recall_calibration_only") else len(result.get("query_list") or [])
    searched_pairs = {(item.get("query"), item.get("source")) for item in result.get("search_log") or []}
    expected_pairs = query_count * (1 if options.get("recall_calibration_only") else 4)
    unaudited = max(0, len(result.get("topic_records") or []) - len(result.get("oa_audit_records") or []))
    retryable_download_failures = [
        failure for failure in result.get("failures") or [] if failure.get("stage") == "download_node" and failure.get("retryable")
    ]
    search_logs = list(result.get("search_log") or [])
    maybe_more = any(item.get("has_more") or item.get("next_cursor") for item in search_logs)
    insufficient = []
    max_limited = []
    source_errors = []
    low_recall = []
    for item in search_logs:
        total = item.get("total_count")
        retrieved = int(item.get("retrieved_count") or item.get("result_count") or 0)
        if item.get("status") == "error":
            source_errors.append(item)
        if total not in (None, "") and retrieved < int(total or 0):
            insufficient.append(item)
        if retrieved and retrieved >= int(item.get("max_results_per_query") or 0):
            max_limited.append(item)
        if total not in (None, 0, "") and retrieved / int(total) < 0.5:
            low_recall.append(item)
    needs_more_search = bool(expected_pairs and len(searched_pairs) < expected_pairs and int(result.get("search_attempt_count") or 0) < int(result.get("max_search_attempts") or 1))
    needs_more_oa = bool(unaudited and int(result.get("oa_attempt_count") or 0) < int(result.get("max_oa_attempts") or 1))
    reasons = sorted(
        {
            *({"pagination_or_limit_insufficient"} if insufficient or maybe_more or max_limited else set()),
            *({"source_error"} if source_errors else set()),
            *({"low_recall_ratio"} if low_recall else set()),
        }
    )
    coverage = {
        "status": "partial" if (needs_more_search or needs_more_oa or retryable_download_failures or maybe_more or insufficient or max_limited or source_errors or low_recall) else "ok",
        "query_count": query_count,
        "searched_query_source_pairs": len(searched_pairs),
        "expected_query_source_pairs": expected_pairs,
        "unaudited_records": unaudited,
        "retryable_download_failures": len(retryable_download_failures),
        "pagination_maybe_more": maybe_more,
        "insufficient_retrieval_count": len(insufficient),
        "max_limited_count": len(max_limited),
        "source_error_count": len(source_errors),
        "low_recall_count": len(low_recall),
        "reasons": reasons,
        "needs_more_search": needs_more_search,
        "needs_more_oa_audit": needs_more_oa,
    }
    result["coverage"] = coverage
    if needs_more_search:
        result["next_action"] = "multi_source_search"
    elif needs_more_oa:
        result["next_action"] = "oa_audit"
    else:
        result["next_action"] = "gate"
    return result


def _route_after_coverage(state: LiteratureAgentState) -> str:
    if (state.get("options") or {}).get("recall_calibration_only"):
        return "human_or_policy_gate_node"
    action = state.get("next_action")
    if action == "multi_source_search":
        return "multi_source_search_node"
    if action == "oa_audit":
        return "oa_audit_node"
    return "human_or_policy_gate_node"


def corpus_export_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    output_dir = str((result.get("options") or {}).get("output_dir") or "agent_runs")
    manifest_result = tr.export_corpus_manifest_tool(result, output_dir=output_dir)
    coverage_result = tr.write_coverage_report_tool(result, output_dir=output_dir)
    recall_result = tr.write_recall_diagnostics_tool(result, output_dir=output_dir)
    topic_diag_result = tr.write_topic_guard_diagnostics_tool(result, output_dir=output_dir)
    scope_result = tr.write_scope_guard_outputs_tool(result, output_dir=output_dir)
    artifacts = dict(result.get("artifacts") or {})
    reports = dict(result.get("reports") or {})
    failures = list(result.get("failures") or [])
    if manifest_result.get("status") == "ok":
        artifacts.update(manifest_result.get("artifacts") or {})
        result["corpus_manifest"] = list((manifest_result.get("data") or {}).get("corpus_manifest") or [])
    else:
        failures.append({"stage": "corpus_export_node", "tool": "export_corpus_manifest", "error": manifest_result.get("error")})
    if coverage_result.get("status") == "ok":
        artifacts.update(coverage_result.get("artifacts") or {})
        reports.update(coverage_result.get("data") or {})
    else:
        failures.append({"stage": "corpus_export_node", "tool": "write_coverage_report", "error": coverage_result.get("error")})
    if recall_result.get("status") == "ok":
        artifacts.update(recall_result.get("artifacts") or {})
    else:
        failures.append({"stage": "corpus_export_node", "tool": "write_recall_diagnostics", "error": recall_result.get("error")})
    if topic_diag_result.get("status") == "ok":
        artifacts.update(topic_diag_result.get("artifacts") or {})
    else:
        failures.append({"stage": "corpus_export_node", "tool": "write_topic_guard_diagnostics", "error": topic_diag_result.get("error")})
    if scope_result.get("status") == "ok":
        artifacts.update(scope_result.get("artifacts") or {})
        reports.update(scope_result.get("data") or {})
    else:
        failures.append({"stage": "corpus_export_node", "tool": "write_scope_guard_outputs", "error": scope_result.get("error")})
    result["artifacts"] = artifacts
    result["reports"] = reports
    result["failures"] = failures
    return result


def human_or_policy_gate_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    failures = list(result.get("failures") or [])
    if not options.get("download", False):
        result["next_action"] = "diagnose"
        return result
    if not options.get("allow_download", False):
        result["blocked"] = True
        result["next_action"] = "diagnose"
        failures.append({"stage": "human_or_policy_gate_node", "reason": "allow_download must be True"})
    elif not options.get("yes", False):
        result["blocked"] = True
        result["next_action"] = "diagnose"
        failures.append({"stage": "human_or_policy_gate_node", "reason": "yes must be True"})
    else:
        result["next_action"] = "download"
    result["failures"] = failures
    return result


def _route_after_gate(state: LiteratureAgentState) -> str:
    return "download_node" if state.get("next_action") == "download" else "diagnose_node"


def download_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    download_results: list[dict[str, Any]] = []
    failures = list(result.get("failures") or [])
    for request in result.get("download_plan") or []:
        tool_result = tr.download_legal_pdf_tool(
            request,
            allow_download=bool(options.get("allow_download", False)),
            yes=bool(options.get("yes", False)),
            output_dir=str(options.get("download_dir") or "downloads"),
        )
        download_results.append(tool_result)
        if tool_result.get("status") not in {"ok", "partial"}:
            failures.append({"stage": "download_node", "tool": "download_legal_pdf", "error": tool_result.get("error")})
    result["download_results"] = download_results
    result["failures"] = failures
    return result


def diagnose_node(state: LiteratureAgentState) -> LiteratureAgentState:
    return diagnose_with_llm(state)


def _summary_from_state(state: LiteratureAgentState) -> dict[str, Any]:
    downloads = state.get("download_results") or []
    actual_downloads = 0
    for item in downloads:
        data = item.get("data") if isinstance(item, dict) else None
        if isinstance(data, list):
            actual_downloads += sum(1 for row in data if row.get("status") == "downloaded")
    return {
        "request": state.get("user_request", ""),
        "task": state.get("task", {}),
        "metadata_count": len(state.get("metadata_results") or []),
        "raw_count": len(state.get("raw_records") or []),
        "unique_count": len(state.get("unique_records") or []),
        "topic_count": len(state.get("topic_records") or []),
        "scope_included_count": len(state.get("scope_included_records") or []),
        "scope_excluded_count": len(state.get("scope_excluded_records") or []),
        "manual_scope_review_count": len(state.get("needs_manual_scope_review") or []),
        "oa_count": len(state.get("oa_results") or []),
        "planned_downloads": len(state.get("download_plan") or []),
        "actual_downloads": actual_downloads,
        "blocked": bool(state.get("blocked", False)),
        "diagnostics": state.get("diagnostics", {}),
        "artifacts": state.get("artifacts", {}),
        "reports": state.get("reports", {}),
        "failures": state.get("failures", []),
        "query_list": state.get("query_list", []),
        "coverage": state.get("coverage", {}),
        "corpus_manifest_count": len(state.get("corpus_manifest") or []),
        "steps": [
            {"step": "metadata_rank", "summary": {"metadata_results_written": len(state.get("metadata_results") or [])}},
            {"step": "oa_check", "summary": {"oa_checked": len(state.get("oa_results") or [])}},
            {"step": "download_plan", "summary": {"planned_downloads": len(state.get("download_plan") or [])}},
            {"step": "download", "summary": {"actual_downloads": actual_downloads}},
        ],
    }


def report_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    summary = _summary_from_state(result)
    report = tr.write_report_tool(summary, output_dir=str(options.get("output_dir") or "agent_runs"))
    reports = dict(result.get("reports") or {})
    artifacts = dict(result.get("artifacts") or {})
    if report.get("status") == "ok":
        reports.update(report.get("data") or {})
        artifacts.update(report.get("artifacts") or {})
    else:
        failures = list(result.get("failures") or [])
        failures.append({"stage": "report_node", "tool": "write_report", "error": report.get("error")})
        result["failures"] = failures
    result["reports"] = reports
    result["artifacts"] = artifacts
    return result


def finish_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    result["finished"] = True
    return result


def understand_acquisition_request_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    request = str(result.get("user_request") or result.get("request") or "")
    parsed = parse_acquisition_request_with_llm(request)
    result["acquisition_request"] = parsed.to_dict()
    result["parsed_request"] = parsed.to_dict()
    result["query_list"] = [parsed.primary_query, *parsed.expansion_queries]
    result["year_from"] = parsed.year_from
    result["year_to"] = parsed.year_to
    result["sources"] = list(parsed.sources)
    result.setdefault("artifacts", {})
    return result


def plan_acquisition_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    parsed = LiteratureAcquisitionRequest(**(result.get("acquisition_request") or {}))
    if parsed.intent == "count_summary":
        result["next_action"] = "run_count_summary_node"
        return result
    if parsed.needs_clarification:
        result["status"] = "needs_clarification"
        result["blocked"] = True
        result["next_action"] = "finalize_acquisition_node"
        return result
    output_dir = str(((result.get("options") or {}).get("output_dir")) or parsed.output_dir)
    parsed.output_dir = output_dir
    artifacts = dict(result.get("artifacts") or {})
    artifacts["acquisition_request"] = write_acquisition_plan(parsed, output_dir)
    result["artifacts"] = artifacts
    result["acquisition_request"] = parsed.to_dict()
    result["parsed_request"] = parsed.to_dict()
    result["query_list"] = [parsed.primary_query, *parsed.expansion_queries]
    result["next_action"] = "run_collection_node"
    return result


def run_count_summary_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    parsed = LiteratureAcquisitionRequest(**(result.get("acquisition_request") or {}))
    output_dir = str(((result.get("options") or {}).get("output_dir")) or parsed.output_dir or "")
    summary = summarize_literature_counts(
        topic=parsed.topic or parsed.original_request,
        query=parsed.primary_query,
        year_from=parsed.year_from,
        year_to=parsed.year_to,
        output_dir=output_dir or None,
        include_openalex=True,
        include_pubmed=True,
        include_local=True,
    )
    result["count_summary"] = summary
    result["summary"] = {**summary, "parsed_request": parsed.to_dict()}
    result["artifacts"] = {**(result.get("artifacts") or {}), **(summary.get("artifacts") or {})}
    result["finished"] = True
    result["next_action"] = "finish_node"
    return result


def run_collection_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    parsed = LiteratureAcquisitionRequest(**(result.get("acquisition_request") or {}))
    collection = tr.run_full_collection_loop_tool(
        query_list=list(result.get("query_list") or [parsed.primary_query]),
        sources=list(parsed.sources or ["pubmed", "openalex", "crossref", "europe_pmc"]),
        year_from=parsed.year_from,
        year_to=parsed.year_to,
        max_results_per_batch=int(options.get("max_results_per_batch") or 300),
        max_additional_results_per_batch=int(options.get("max_additional_results_per_batch") or 300),
        max_rounds=int(options.get("max_rounds") or 20),
        max_batches_per_round=int(options.get("max_batches_per_round") or 12),
        stop_if_no_growth_rounds=int(options.get("stop_if_no_growth_rounds") or 3),
        min_new_unique_records_per_round=int(options.get("min_new_unique_records_per_round") or 5),
        retry_failed=bool(options.get("retry_failed", False)),
        time_budget_seconds=options.get("time_budget_seconds", 3300),
        graceful_stop_buffer_seconds=int(options.get("graceful_stop_buffer_seconds") or 180),
        resume=True,
        checkpoint_every_batch=True,
        finalize_on_stop=True,
        scope_profile=parsed.query_profile,
        include_terms=list(parsed.include_terms),
        exclude_terms=list(parsed.exclude_terms),
        output_dir=parsed.output_dir,
    )
    result["collection_result"] = collection.get("data") or {}
    artifacts = dict(result.get("artifacts") or {})
    artifacts.update(collection.get("artifacts") or {})
    result["artifacts"] = artifacts
    result["next_action"] = "run_pre_download_qa_node"
    return result


def run_pre_download_qa_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    parsed = LiteratureAcquisitionRequest(**(result.get("acquisition_request") or {}))
    qa = tr.pre_download_qa_for_legal_oa_candidates_tool(parsed.output_dir)
    result["pre_download_qa"] = qa.get("data") or {}
    artifacts = dict(result.get("artifacts") or {})
    artifacts.update(qa.get("artifacts") or {})
    result["artifacts"] = artifacts
    options = result.get("options") or {}
    result["next_action"] = "run_fulltext_collection_node" if bool(options.get("allow_download")) and bool(options.get("yes")) else "finalize_acquisition_node"
    return result


def run_fulltext_collection_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    parsed = LiteratureAcquisitionRequest(**(result.get("acquisition_request") or {}))
    fulltext = tr.collect_approved_legal_fulltexts_tool(
        output_dir=parsed.output_dir,
        approved_file="approved_for_download.jsonl",
        allow_download=bool(options.get("allow_download")),
        yes=bool(options.get("yes")),
        max_items=options.get("max_fulltext_items"),
        prefer_formats=list(parsed.fulltext_formats or LEGAL_FULLTEXT_FORMATS),
    )
    result["fulltext_collection"] = fulltext.get("data") or {}
    artifacts = dict(result.get("artifacts") or {})
    artifacts.update(fulltext.get("artifacts") or {})
    result["artifacts"] = artifacts
    result["next_action"] = "finalize_acquisition_node"
    return result


def finalize_acquisition_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    parsed = result.get("parsed_request") or result.get("acquisition_request") or {}
    if result.get("status") == "needs_clarification" or parsed.get("needs_clarification"):
        result["summary"] = {
            "status": "needs_clarification",
            "request": parsed.get("original_request", result.get("user_request", "")),
            "parsed_request": parsed,
            "clarification_question": parsed.get("clarification_question"),
            "suggested_profiles": parsed.get("suggested_profiles") or [],
            "actual_downloads": 0,
            "actual_fulltexts": 0,
            "artifacts": result.get("artifacts") or {},
        }
        result["finished"] = True
        return result
    collection = result.get("collection_result") or {}
    qa = result.get("pre_download_qa") or {}
    fulltext = result.get("fulltext_collection") or {}
    result["summary"] = {
        "status": collection.get("status", "ok"),
        "request": parsed.get("original_request", result.get("user_request", "")),
        "parsed_request": parsed,
        "collection": collection,
        "pre_download_qa": qa,
        "fulltext_collection": fulltext,
        "fulltext_collection_note": "" if fulltext else "Fulltext collection requires allow_download=True and yes=True.",
        "actual_downloads": int(fulltext.get("actual_fulltexts") or fulltext.get("actual_downloads") or 0),
        "actual_fulltexts": int(fulltext.get("actual_fulltexts") or 0),
        "artifacts": result.get("artifacts") or {},
        "warnings": parsed.get("warnings") or [],
    }
    result["finished"] = True
    return result


def plan_collection_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    result["query_list"] = list(options.get("query_list") or BROAD_CRISPR_CORPUS_QUERIES)
    result["sources"] = list(options.get("sources") or ["pubmed", "openalex", "crossref", "europe_pmc"])
    result.setdefault("collection_round", 0)
    result.setdefault("low_growth_rounds", 0)
    result.setdefault("previous_unique_records", 0)
    return result


def build_batches_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    output_dir = Path(str(options.get("output_dir") or "agent_runs/crispr_broad_full_collection"))
    manifest_path = output_dir / "batch_manifest.jsonl"
    batches = plan_collection_batches(
        list(result.get("query_list") or []),
        list(result.get("sources") or []),
        options.get("year_from") if options.get("year_from") is not None else result.get("year_from"),
        options.get("year_to") if options.get("year_to") is not None else result.get("year_to"),
        int(options.get("max_results_per_batch") or 500),
    )
    if manifest_path.exists() and read_jsonl(manifest_path):
        result["batch_manifest"] = read_jsonl(manifest_path)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("", encoding="utf-8")
        for batch in batches:
            batch["output_file"] = str((output_dir / "raw_records_by_batch" / f"{batch['batch_id']}.jsonl").resolve())
            append_jsonl(manifest_path, batch)
        result["batch_manifest"] = batches
    return result


def batch_scheduler_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    output_dir = str(options.get("output_dir") or "agent_runs/crispr_broad_full_collection")
    round_index = int(result.get("collection_round") or 0) + 1
    scheduled = run_batch_scheduler(
        list(result.get("batch_manifest") or []),
        output_dir=output_dir,
        max_batches_per_round=int(options.get("max_batches_per_round") or 20),
        max_additional_results_per_batch=int(options.get("max_additional_results_per_batch") or 500),
        retry_failed=bool(options.get("retry_failed", False)),
        strategy=str(options.get("scheduler_strategy") or "coverage_first"),
    )
    unique_records = int((scheduled.get("merge") or {}).get("unique_records") or 0)
    previous = int(result.get("previous_unique_records") or 0)
    new_unique = max(0, unique_records - previous)
    min_growth = int(options.get("min_new_unique_records_per_round") or 10)
    low_growth = int(result.get("low_growth_rounds") or 0)
    low_growth = low_growth + 1 if new_unique < min_growth else 0
    progress = list(result.get("collection_progress") or [])
    progress_row = {
        "round": round_index,
        "scheduled_batches": scheduled.get("scheduled_batches", 0),
        "selected_batches": scheduled.get("selected_batches", []),
        "new_unique_records": new_unique,
        "unique_records": unique_records,
    }
    progress.append(progress_row)
    append_jsonl(Path(output_dir) / "full_collection_progress.jsonl", progress_row)
    result["collection_round"] = round_index
    result["batch_manifest"] = list(scheduled.get("batch_manifest") or [])
    result["previous_unique_records"] = unique_records
    result["low_growth_rounds"] = low_growth
    result["collection_progress"] = progress
    result["merge"] = scheduled.get("merge") or {}
    return result


def collection_completion_check_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    options = result.get("options") or {}
    decision = check_collection_completion(
        list(result.get("batch_manifest") or []),
        round_index=int(result.get("collection_round") or 0),
        low_growth_rounds=int(result.get("low_growth_rounds") or 0),
        max_rounds=int(options.get("max_rounds") or 20),
        stop_if_no_growth_rounds=int(options.get("stop_if_no_growth_rounds") or 2),
    )
    result["collection_completion"] = decision
    result["next_action"] = "batch_scheduler_node" if decision.get("should_continue") else ("diagnose_collection_node" if decision.get("needs_llm_diagnosis") else "merge_dedup_node")
    return result


def diagnose_collection_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    result["diagnostics"] = {**(result.get("diagnostics") or {}), "collection": result.get("collection_completion") or {}}
    result["next_action"] = "merge_dedup_node"
    return result


def merge_dedup_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    output_dir = str((result.get("options") or {}).get("output_dir") or "agent_runs/crispr_broad_full_collection")
    merged = tr.merge_batch_outputs_tool(output_dir)
    result["merge"] = merged.get("data") or {}
    result["next_action"] = "finalize_corpus_node"
    return result


def finalize_corpus_node(state: LiteratureAgentState) -> LiteratureAgentState:
    result: LiteratureAgentState = dict(state)
    output_dir = str((result.get("options") or {}).get("output_dir") or "agent_runs/crispr_broad_full_collection")
    final = tr.build_final_corpus_manifest_tool(output_dir)
    result["final_corpus"] = final.get("data") or {}
    result["artifacts"] = {**(result.get("artifacts") or {}), **(final.get("artifacts") or {})}
    latest_manifest = list(read_jsonl(Path(output_dir) / "batch_manifest.jsonl")) if (Path(output_dir) / "batch_manifest.jsonl").exists() else list(result.get("batch_manifest") or [])
    status_counts = {status: sum(1 for row in latest_manifest if row.get("status") == status) for status in ["completed", "partial", "failed", "pending"]}
    merge = result.get("merge") or {}
    final_data = final.get("data") or {}
    summary = {
        "rounds_run": int(result.get("collection_round") or 0),
        "stop_reason": (result.get("collection_completion") or {}).get("stop_reason", ""),
        "total_batches": len(latest_manifest),
        "completed_batches": status_counts.get("completed", 0),
        "partial_batches": status_counts.get("partial", 0),
        "failed_batches": status_counts.get("failed", 0),
        "total_raw_records": merge.get("total_raw_records", 0),
        "unique_records": merge.get("unique_records", 0),
        "corpus_manifest_records": final_data.get("corpus_manifest_count", 0),
        "metadata_only_records": final_data.get("metadata_only_manifest_count", 0),
        "downloaded_pdf_records": final_data.get("downloaded_pdfs_manifest_count", 0),
        "planned_legal_oa_count": final_data.get("planned_legal_oa_count", 0),
        "actual_downloads": 0,
        "scheduler_strategy": str((result.get("options") or {}).get("scheduler_strategy") or "coverage_first"),
    }
    report_path = _write_full_collection_report(output_dir, summary, list(result.get("collection_progress") or []))
    artifacts = dict(result.get("artifacts") or {})
    artifacts["full_collection_report"] = report_path
    artifacts["full_collection_progress"] = str((Path(output_dir) / "full_collection_progress.jsonl").resolve())
    result["artifacts"] = artifacts
    result["next_action"] = "finish_node"
    return result


def _route_after_collection_check(state: LiteratureAgentState) -> str:
    return str(state.get("next_action") or "merge_dedup_node")


def _route_after_plan_acquisition(state: LiteratureAgentState) -> str:
    return str(state.get("next_action") or "run_collection_node")


def _route_after_pre_download_qa(state: LiteratureAgentState) -> str:
    return str(state.get("next_action") or "finalize_acquisition_node")


def build_acquisition_graph():
    graph = StateGraph(LiteratureAgentState)
    graph.add_node("initialize_state_node", initialize_state_node)
    graph.add_node("understand_acquisition_request_node", understand_acquisition_request_node)
    graph.add_node("plan_acquisition_node", plan_acquisition_node)
    graph.add_node("run_count_summary_node", run_count_summary_node)
    graph.add_node("run_collection_node", run_collection_node)
    graph.add_node("run_pre_download_qa_node", run_pre_download_qa_node)
    graph.add_node("run_fulltext_collection_node", run_fulltext_collection_node)
    graph.add_node("finalize_acquisition_node", finalize_acquisition_node)
    graph.add_node("finish_node", finish_node)
    graph.set_entry_point("initialize_state_node")
    graph.add_edge("initialize_state_node", "understand_acquisition_request_node")
    graph.add_edge("understand_acquisition_request_node", "plan_acquisition_node")
    graph.add_conditional_edges(
        "plan_acquisition_node",
        _route_after_plan_acquisition,
        {
            "run_count_summary_node": "run_count_summary_node",
            "run_collection_node": "run_collection_node",
            "finalize_acquisition_node": "finalize_acquisition_node",
        },
    )
    graph.add_edge("run_count_summary_node", "finish_node")
    graph.add_edge("run_collection_node", "run_pre_download_qa_node")
    graph.add_conditional_edges(
        "run_pre_download_qa_node",
        _route_after_pre_download_qa,
        {
            "run_fulltext_collection_node": "run_fulltext_collection_node",
            "finalize_acquisition_node": "finalize_acquisition_node",
        },
    )
    graph.add_edge("run_fulltext_collection_node", "finalize_acquisition_node")
    graph.add_edge("finalize_acquisition_node", "finish_node")
    graph.add_edge("finish_node", END)
    return graph.compile()


def build_full_collection_graph():
    graph = StateGraph(LiteratureAgentState)
    graph.add_node("initialize_state_node", initialize_state_node)
    graph.add_node("understand_request_node", understand_request_node)
    graph.add_node("plan_collection_node", plan_collection_node)
    graph.add_node("build_batches_node", build_batches_node)
    graph.add_node("batch_scheduler_node", batch_scheduler_node)
    graph.add_node("collection_completion_check_node", collection_completion_check_node)
    graph.add_node("diagnose_collection_node", diagnose_collection_node)
    graph.add_node("merge_dedup_node", merge_dedup_node)
    graph.add_node("scope_guard_node", lambda state: state)
    graph.add_node("oa_audit_node", lambda state: state)
    graph.add_node("finalize_corpus_node", finalize_corpus_node)
    graph.add_node("finish_node", finish_node)
    graph.set_entry_point("initialize_state_node")
    graph.add_edge("initialize_state_node", "understand_request_node")
    graph.add_edge("understand_request_node", "plan_collection_node")
    graph.add_edge("plan_collection_node", "build_batches_node")
    graph.add_edge("build_batches_node", "batch_scheduler_node")
    graph.add_edge("batch_scheduler_node", "collection_completion_check_node")
    graph.add_conditional_edges(
        "collection_completion_check_node",
        _route_after_collection_check,
        {
            "batch_scheduler_node": "batch_scheduler_node",
            "diagnose_collection_node": "diagnose_collection_node",
            "merge_dedup_node": "merge_dedup_node",
        },
    )
    graph.add_edge("diagnose_collection_node", "merge_dedup_node")
    graph.add_edge("merge_dedup_node", "scope_guard_node")
    graph.add_edge("scope_guard_node", "oa_audit_node")
    graph.add_edge("oa_audit_node", "finalize_corpus_node")
    graph.add_edge("finalize_corpus_node", "finish_node")
    graph.add_edge("finish_node", END)
    return graph.compile()


def build_agent_graph():
    graph = StateGraph(LiteratureAgentState)
    graph.add_node("initialize_state_node", initialize_state_node)
    graph.add_node("understand_request_node", understand_request_node)
    graph.add_node("route_node", route_node)
    graph.add_node("query_expansion_node", query_expansion_node)
    graph.add_node("multi_source_search_node", multi_source_search_node)
    graph.add_node("deduplicate_node", deduplicate_node)
    graph.add_node("recall_calibration_node", recall_calibration_node)
    graph.add_node("topic_guard_node", topic_guard_node)
    graph.add_node("oa_audit_node", oa_audit_node)
    graph.add_node("search_metadata_node", search_metadata_node)
    graph.add_node("audit_oa_node", audit_oa_node)
    graph.add_node("plan_download_node", plan_download_node)
    graph.add_node("coverage_check_node", coverage_check_node)
    graph.add_node("human_or_policy_gate_node", human_or_policy_gate_node)
    graph.add_node("download_node", download_node)
    graph.add_node("diagnose_node", diagnose_node)
    graph.add_node("corpus_export_node", corpus_export_node)
    graph.add_node("report_node", report_node)
    graph.add_node("finish_node", finish_node)
    graph.set_entry_point("initialize_state_node")
    graph.add_edge("initialize_state_node", "understand_request_node")
    graph.add_edge("understand_request_node", "route_node")
    graph.add_conditional_edges(
        "route_node",
        _route_after_route_node,
        {
            "search_metadata": "search_metadata_node",
            "query_expansion": "query_expansion_node",
            "audit_oa": "audit_oa_node",
            "plan_download": "plan_download_node",
            "download": "human_or_policy_gate_node",
            "report": "report_node",
            "finish": "finish_node",
        },
    )
    graph.add_edge("search_metadata_node", "query_expansion_node")
    graph.add_edge("query_expansion_node", "multi_source_search_node")
    graph.add_edge("multi_source_search_node", "deduplicate_node")
    graph.add_edge("deduplicate_node", "recall_calibration_node")
    graph.add_edge("recall_calibration_node", "topic_guard_node")
    graph.add_edge("topic_guard_node", "oa_audit_node")
    graph.add_edge("audit_oa_node", "plan_download_node")
    graph.add_edge("oa_audit_node", "plan_download_node")
    graph.add_edge("plan_download_node", "coverage_check_node")
    graph.add_conditional_edges(
        "coverage_check_node",
        _route_after_coverage,
        {
            "multi_source_search_node": "multi_source_search_node",
            "oa_audit_node": "oa_audit_node",
            "human_or_policy_gate_node": "human_or_policy_gate_node",
        },
    )
    graph.add_conditional_edges("human_or_policy_gate_node", _route_after_gate, {"download_node": "download_node", "diagnose_node": "diagnose_node"})
    graph.add_edge("download_node", "diagnose_node")
    graph.add_edge("diagnose_node", "corpus_export_node")
    graph.add_edge("corpus_export_node", "report_node")
    graph.add_edge("report_node", "finish_node")
    graph.add_edge("finish_node", END)
    return graph.compile()


def run_literature_graph_agent(
    user_request: str,
    *,
    allow_network_metadata: bool = True,
    allow_network_rank: bool = True,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results: int = 30,
    max_downloads: int = 5,
    use_llm_diagnosis: bool = True,
    output_dir: str = "agent_runs",
    year_from: int | None = None,
    year_to: int | None = None,
    max_results_per_query: int | None = None,
    exhaustive_mode: bool = False,
    include_reviews: bool = False,
    google_scholar_reference_count: int | None = None,
    pubmed_reference_count: int | None = None,
    mode: str = "crispr_detection_corpus",
    conservative_scope_guard: bool = True,
    recall_calibration_only: bool = False,
) -> dict[str, Any]:
    initial: LiteratureAgentState = {
        "user_request": user_request,
        "options": {
            "allow_network_metadata": allow_network_metadata,
            "allow_network_rank": allow_network_rank,
            "download": download,
            "allow_download": allow_download,
            "yes": yes,
            "max_results": max_results,
            "max_downloads": max_downloads,
            "year_from": year_from,
            "year_to": year_to,
            "max_results_per_query": max_results_per_query or max_results,
            "exhaustive_mode": exhaustive_mode,
            "include_reviews": include_reviews,
            "recall_calibration_only": recall_calibration_only,
            "max_search_attempts": 1 if recall_calibration_only else None,
            "mode": mode,
            "conservative_scope_guard": conservative_scope_guard,
            "google_scholar_reference_count": google_scholar_reference_count,
            "pubmed_reference_count": pubmed_reference_count,
            "use_llm_diagnosis": use_llm_diagnosis,
            "output_dir": output_dir,
        },
        "manual_reference_counts": {
            "google_scholar_reference_count": google_scholar_reference_count,
            "pubmed_reference_count": pubmed_reference_count,
        },
    }
    final_state = build_agent_graph().invoke(initial)
    summary = _summary_from_state(final_state)
    if final_state.get("blocked"):
        status = "blocked"
    elif final_state.get("error"):
        status = "error"
    elif final_state.get("failures"):
        status = "partial"
    else:
        status = "ok"
    summary.update(
        {
            "status": status,
            "request": user_request,
            "task": final_state.get("task", {}),
            "metadata_count": len(final_state.get("metadata_results") or []),
            "raw_count": len(final_state.get("raw_records") or []),
            "unique_count": len(final_state.get("unique_records") or []),
            "topic_count": len(final_state.get("topic_records") or []),
            "scope_included_count": len(final_state.get("scope_included_records") or []),
            "scope_excluded_count": len(final_state.get("scope_excluded_records") or []),
            "manual_scope_review_count": len(final_state.get("needs_manual_scope_review") or []),
            "oa_count": len(final_state.get("oa_results") or []),
            "planned_downloads": len(final_state.get("download_plan") or []),
            "blocked": bool(final_state.get("blocked", False)),
            "diagnostics": final_state.get("diagnostics", {}),
            "artifacts": final_state.get("artifacts", {}),
            "reports": final_state.get("reports", {}),
            "failures": final_state.get("failures", []),
            "query_list": final_state.get("query_list", []),
            "coverage": final_state.get("coverage", {}),
            "corpus_manifest_count": len(final_state.get("corpus_manifest") or []),
        }
    )
    return summary


def run_crispr_detection_corpus_agent(
    *,
    year_from: int,
    year_to: int,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results_per_query: int = 200,
    max_downloads: int = 1000,
    exhaustive_mode: bool = True,
    include_reviews: bool = False,
    output_dir: str = "agent_runs/crispr_detection_corpus",
    google_scholar_reference_count: int | None = 23200,
    pubmed_reference_count: int | None = 2700,
) -> dict[str, Any]:
    return run_literature_graph_agent(
        f"Collect legal open-access CRISPR detection corpus from {year_from} to {year_to}.",
        allow_network_metadata=bool(exhaustive_mode),
        allow_network_rank=False,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results=max_results_per_query,
        max_downloads=max_downloads,
        use_llm_diagnosis=False,
        output_dir=output_dir,
        year_from=year_from,
        year_to=year_to,
        max_results_per_query=max_results_per_query,
        exhaustive_mode=exhaustive_mode,
        include_reviews=include_reviews,
        google_scholar_reference_count=google_scholar_reference_count,
        pubmed_reference_count=pubmed_reference_count,
        mode="crispr_detection_corpus",
    )


def run_crispr_broad_corpus_agent(
    *,
    year_from: int,
    year_to: int,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results_per_query: int = 5000,
    max_downloads: int = 2000,
    exhaustive_mode: bool = True,
    include_reviews: bool = True,
    conservative_scope_guard: bool = True,
    output_dir: str = "agent_runs/crispr_broad_corpus",
) -> dict[str, Any]:
    return run_literature_graph_agent(
        f"Collect broad legal open-access CRISPR corpus from {year_from} to {year_to}.",
        allow_network_metadata=bool(exhaustive_mode),
        allow_network_rank=False,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results=max_results_per_query,
        max_downloads=max_downloads,
        use_llm_diagnosis=False,
        output_dir=output_dir,
        year_from=year_from,
        year_to=year_to,
        max_results_per_query=max_results_per_query,
        exhaustive_mode=exhaustive_mode,
        include_reviews=include_reviews,
        mode="crispr_broad_corpus",
        conservative_scope_guard=conservative_scope_guard,
    )


def run_crispr_broad_recall_calibration(
    *,
    year_from: int,
    year_to: int,
    max_results_per_source: int = 5000,
    output_dir: str = "agent_runs/crispr_broad_recall_calibration",
) -> dict[str, Any]:
    result = run_literature_graph_agent(
        f"Calibrate broad CRISPR corpus recall from {year_from} to {year_to}.",
        allow_network_metadata=True,
        allow_network_rank=False,
        download=False,
        allow_download=False,
        yes=False,
        max_results=max_results_per_source,
        max_downloads=0,
        use_llm_diagnosis=False,
        output_dir=output_dir,
        year_from=year_from,
        year_to=year_to,
        max_results_per_query=max_results_per_source,
        exhaustive_mode=True,
        include_reviews=True,
        mode="crispr_broad_corpus",
        recall_calibration_only=True,
    )
    out = Path(output_dir)
    recall_json = out / "recall_diagnostics.json"
    recall_md = out / "recall_diagnostics_report.md"
    if recall_json.exists():
        shutil.copyfile(recall_json, out / "broad_recall_diagnostics.json")
    if recall_md.exists():
        shutil.copyfile(recall_md, out / "broad_recall_diagnostics_report.md")
    metadata = out / "metadata_all.jsonl"
    if metadata.exists():
        shutil.copyfile(metadata, out / "raw_primary_broad_query_records.jsonl")
    return result


def run_crispr_broad_full_collection(
    *,
    year_from: int | None,
    year_to: int | None,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
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
    output_dir: str = "agent_runs/crispr_broad_full_collection",
) -> dict[str, Any]:
    _ = (download, allow_download, yes)
    result = tr.run_full_collection_loop_tool(
        query_list=BROAD_CRISPR_CORPUS_QUERIES,
        sources=["pubmed", "openalex", "crossref", "europe_pmc"],
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
        output_dir=output_dir,
    )
    return dict(result.get("data") or {"status": result.get("status"), "error": result.get("error"), "actual_downloads": 0})


def run_literature_acquisition_agent(
    request: str,
    allow_download: bool = False,
    yes: bool = False,
    output_dir: str | None = None,
    max_rounds: int = 20,
    max_batches_per_round: int = 12,
    max_results_per_batch: int = 300,
    max_additional_results_per_batch: int = 300,
    time_budget_seconds: int | None = 3300,
    graceful_stop_buffer_seconds: int = 180,
) -> dict[str, Any]:
    graph = build_acquisition_graph()
    initial: LiteratureAgentState = {
        "user_request": request,
        "options": {
            "allow_download": bool(allow_download),
            "yes": bool(yes),
            "output_dir": output_dir,
            "max_rounds": max_rounds,
            "max_batches_per_round": max_batches_per_round,
            "max_results_per_batch": max_results_per_batch,
            "max_additional_results_per_batch": max_additional_results_per_batch,
            "time_budget_seconds": time_budget_seconds,
            "graceful_stop_buffer_seconds": graceful_stop_buffer_seconds,
            "stop_if_no_growth_rounds": 3,
            "min_new_unique_records_per_round": 5,
            "retry_failed": False,
        },
    }
    final_state = graph.invoke(initial)
    summary = dict(final_state.get("summary") or {})
    summary.setdefault("request", request)
    summary.setdefault("parsed_request", final_state.get("parsed_request") or {})
    summary.setdefault("actual_downloads", 0)
    summary.setdefault("actual_fulltexts", 0)
    summary.setdefault("artifacts", final_state.get("artifacts") or {})
    return summary
