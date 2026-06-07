"""Count-only literature summary helpers.

This module performs metadata count queries and local manifest counting only.
It does not start batch collection and does not download or save full text.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode

from .manifest import read_jsonl
from .openalex_calibration import _params as openalex_params
from .openalex_calibration import _get_json as openalex_get_json
from .query import QuerySpec
from .sources.common import safe_get_json
from .sources.openalex import OPENALEX_WORKS_ENDPOINT
from .sources.pubmed import PUBMED_ESEARCH_ENDPOINT, build_pubmed_esearch_params


CRISPR_COUNT_QUERY = '(CRISPR OR "CRISPR-Cas" OR Cas9 OR Cas12 OR Cas12a OR Cas13 OR Cas13a OR Cas14 OR Cpf1 OR C2c2 OR SHERLOCK OR DETECTR OR HOLMES)'
CRISPR_DETECTION_COUNT_QUERY = "(CRISPR OR Cas12 OR Cas12a OR Cas13 OR Cas13a) AND (detection OR diagnosis OR diagnostic OR diagnostics OR biosensor OR biosensing OR assay OR sensor)"
DEFAULT_LOCAL_DIRS = [
    "agent_runs/crispr_broad_all_years",
    "agent_runs/crispr_broad_full_collection_2025_2026_medium",
    "agent_runs/test_crispr_broad_full_collection_scheduler_2025_2026",
]


def default_count_query(topic: str, query: str | None = None) -> str:
    if query:
        return query
    text = topic.lower()
    if "crispr" in text and any(term in text for term in ["detect", "diagnos", "biosensor", "assay", "sensor"]):
        return CRISPR_DETECTION_COUNT_QUERY
    if "crispr" in text or "cas12" in text or "cas13" in text:
        return CRISPR_COUNT_QUERY
    return topic


def _status_from_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError) and exc.code == 429:
        return "rate_limited"
    text = str(exc).lower()
    return "rate_limited" if "429" in text or "rate" in text else "error"


def _api_url(params: dict[str, str]) -> str:
    return f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(params)}"


def _openalex_params_for_mode(
    query: str,
    year_from: int | None,
    year_to: int | None,
    *,
    is_oa: bool | None = None,
    mode: str = "title_and_abstract_filter",
    sort: str | None = None,
    per_page: int = 1,
) -> dict[str, str]:
    if mode == "auto":
        params = openalex_params(query, year_from, year_to, is_oa=is_oa, per_page=per_page)
        if sort:
            params["sort"] = sort
        return params
    filters: list[str] = []
    params: dict[str, str] = {"per-page": str(max(1, min(int(per_page), 200))), "mailto": "contact@example.org"}
    if mode == "search":
        params["search"] = query
    elif mode == "title_and_abstract_filter":
        filters.append(f"title_and_abstract.search:{query}")
    else:
        raise ValueError(f"Unsupported OpenAlex count mode: {mode}")
    if year_from is not None:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if is_oa is not None:
        filters.append(f"open_access.is_oa:{str(is_oa).lower()}")
    if filters:
        params["filter"] = ",".join(filters)
    if sort:
        params["sort"] = sort
    return params


def _count_from_params(params: dict[str, str], session: Any = None) -> int:
    payload = openalex_get_json(params, session=session)
    return int((payload.get("meta") or {}).get("count") or 0)


def debug_openalex_count_query(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    *,
    mode: str = "title_and_abstract_filter",
    sort: str | None = None,
    session: Any = None,
) -> dict[str, Any]:
    total_params = _openalex_params_for_mode(query, year_from, year_to, mode=mode, sort=sort, per_page=1)
    oa_params = _openalex_params_for_mode(query, year_from, year_to, is_oa=True, mode=mode, sort=sort, per_page=1)
    try:
        total_count = _count_from_params(total_params, session=session)
        oa_count = _count_from_params(oa_params, session=session)
        return {
            "status": "ok",
            "mode": mode,
            "endpoint": OPENALEX_WORKS_ENDPOINT,
            "api_url": _api_url(total_params),
            "oa_api_url": _api_url(oa_params),
            "params": total_params,
            "oa_params": oa_params,
            "total_count": total_count,
            "oa_count": oa_count,
            "oa_ratio": (oa_count / total_count) if total_count else 0.0,
            "query_used": query,
            "search": total_params.get("search", ""),
            "filter": total_params.get("filter", ""),
            "sort": total_params.get("sort", ""),
            "per_page": total_params.get("per-page", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": _status_from_error(exc),
            "mode": mode,
            "endpoint": OPENALEX_WORKS_ENDPOINT,
            "api_url": _api_url(total_params),
            "oa_api_url": _api_url(oa_params),
            "params": total_params,
            "oa_params": oa_params,
            "total_count": None,
            "oa_count": None,
            "oa_ratio": None,
            "query_used": query,
            "search": total_params.get("search", ""),
            "filter": total_params.get("filter", ""),
            "sort": total_params.get("sort", ""),
            "per_page": total_params.get("per-page", ""),
            "error": str(exc),
        }


def compare_openalex_count_modes(query: str, year_from: int | None, year_to: int | None, *, session: Any = None) -> dict[str, Any]:
    return {
        "search_parameter_mode": debug_openalex_count_query(query, year_from, year_to, mode="search", session=session),
        "title_and_abstract_filter_mode": debug_openalex_count_query(query, year_from, year_to, mode="title_and_abstract_filter", session=session),
    }


def _openalex_summary(query: str, year_from: int | None, year_to: int | None, session: Any = None) -> dict[str, Any]:
    summary = debug_openalex_count_query(query, year_from, year_to, mode="title_and_abstract_filter", session=session)
    summary["debug_modes"] = compare_openalex_count_modes(query, year_from, year_to, session=session)
    return summary


def _pubmed_summary(query: str, year_from: int | None, year_to: int | None, session: Any = None) -> dict[str, Any]:
    spec = QuerySpec(keywords=[query], year_from=year_from, year_to=year_to, max_results=1)
    params = build_pubmed_esearch_params(spec, max_results=1, retstart=0)
    try:
        payload, _calls = safe_get_json(PUBMED_ESEARCH_ENDPOINT, params=params, source="pubmed", session=session, rate_limit_seconds=0.34)
        result = payload.get("esearchresult") or {}
        return {
            "status": "ok",
            "total_count": int(result.get("count") or 0),
            "query_used": query,
            "params": params,
            "error": "",
        }
    except Exception as exc:
        return {
            "status": _status_from_error(exc),
            "total_count": None,
            "query_used": query,
            "params": params,
            "error": str(exc),
        }


def _jsonl_count(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def _choose_local_dir(output_dir: str | None) -> Path | None:
    candidates = [output_dir] if output_dir else []
    candidates.extend(DEFAULT_LOCAL_DIRS)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(output_dir) if output_dir else None


def _local_summary(output_dir: str | None) -> dict[str, Any]:
    path = _choose_local_dir(output_dir)
    if path is None or not path.exists():
        return {"status": "missing", "output_dir": str(path) if path else "", "error": "No local corpus directory found."}
    collected = read_jsonl(path / "collected_fulltexts_manifest.jsonl")
    type_counts = Counter(str(row.get("full_text_type") or row.get("full_text_format") or "").lower() for row in collected)
    return {
        "status": "ok",
        "output_dir": str(path),
        "corpus_manifest_records": _jsonl_count(path / "corpus_manifest.jsonl"),
        "metadata_only_records": _jsonl_count(path / "metadata_only_manifest.jsonl"),
        "approved_for_download_count": _jsonl_count(path / "approved_for_download.jsonl"),
        "collected_fulltexts_count": len(collected),
        "pdf_count": type_counts.get("pdf", 0),
        "xml_count": type_counts.get("pmc_xml", 0) + type_counts.get("europe_pmc_xml", 0) + type_counts.get("xml", 0),
        "html_count": type_counts.get("publisher_html", 0) + type_counts.get("html", 0),
        "failed_fulltext_fetches_count": _jsonl_count(path / "failed_fulltext_fetches.jsonl"),
        "error": "",
    }


def _write_report(output_dir: str | None, summary: dict[str, Any]) -> str:
    base = Path(output_dir or (summary.get("local") or {}).get("output_dir") or "agent_runs/literature_count_summary")
    base.mkdir(parents=True, exist_ok=True)
    openalex = summary.get("openalex") or {}
    debug_modes = summary.get("openalex_debug_modes") or openalex.get("debug_modes") or {}
    pubmed = summary.get("pubmed") or {}
    local = summary.get("local") or {}
    search_mode = debug_modes.get("search_parameter_mode") or {}
    filter_mode = debug_modes.get("title_and_abstract_filter_mode") or {}
    lines = [
        "# Literature Count Summary Report",
        "",
        f"- topic: {summary.get('topic')}",
        f"- query: {summary.get('query_used')}",
        f"- year_filter: {summary.get('year_filter')}",
        "",
        "## OpenAlex",
        f"- status: {openalex.get('status')}",
        f"- total_count: {openalex.get('total_count')}",
        f"- oa_count: {openalex.get('oa_count')}",
        f"- oa_ratio: {openalex.get('oa_ratio')}",
        f"- endpoint: {openalex.get('endpoint')}",
        f"- query_used: {openalex.get('query_used')}",
        f"- search parameter: {openalex.get('search', '')}",
        f"- filter parameter: {openalex.get('filter', '')}",
        f"- sort parameter: {openalex.get('sort', '')}",
        f"- per-page: {openalex.get('per_page', '')}",
        f"- api_url: {openalex.get('api_url', '')}",
        f"- oa_api_url: {openalex.get('oa_api_url', '')}",
        f"- params: {json.dumps(openalex.get('params') or {}, ensure_ascii=False)}",
        f"- oa_params: {json.dumps(openalex.get('oa_params') or {}, ensure_ascii=False)}",
        f"- error: {openalex.get('error', '')}",
        "",
        "### OpenAlex Query Alignment Debug",
        "#### A. search parameter mode",
        f"- status: {search_mode.get('status')}",
        f"- total_count: {search_mode.get('total_count')}",
        f"- oa_count: {search_mode.get('oa_count')}",
        f"- oa_ratio: {search_mode.get('oa_ratio')}",
        f"- api_url: {search_mode.get('api_url', '')}",
        f"- oa_api_url: {search_mode.get('oa_api_url', '')}",
        f"- params: {json.dumps(search_mode.get('params') or {}, ensure_ascii=False)}",
        f"- oa_params: {json.dumps(search_mode.get('oa_params') or {}, ensure_ascii=False)}",
        f"- error: {search_mode.get('error', '')}",
        "",
        "#### B. title_and_abstract.search filter mode",
        f"- status: {filter_mode.get('status')}",
        f"- total_count: {filter_mode.get('total_count')}",
        f"- oa_count: {filter_mode.get('oa_count')}",
        f"- oa_ratio: {filter_mode.get('oa_ratio')}",
        f"- api_url: {filter_mode.get('api_url', '')}",
        f"- oa_api_url: {filter_mode.get('oa_api_url', '')}",
        f"- params: {json.dumps(filter_mode.get('params') or {}, ensure_ascii=False)}",
        f"- oa_params: {json.dumps(filter_mode.get('oa_params') or {}, ensure_ascii=False)}",
        f"- error: {filter_mode.get('error', '')}",
        "",
        "## PubMed",
        f"- status: {pubmed.get('status')}",
        f"- total_count: {pubmed.get('total_count')}",
        f"- error: {pubmed.get('error', '')}",
        "",
        "## Local Corpus",
        f"- status: {local.get('status')}",
        f"- output_dir: {local.get('output_dir')}",
        f"- corpus_manifest_records: {local.get('corpus_manifest_records')}",
        f"- metadata_only_records: {local.get('metadata_only_records')}",
        f"- approved_for_download_count: {local.get('approved_for_download_count')}",
        f"- collected_fulltexts_count: {local.get('collected_fulltexts_count')}",
        f"- pdf_count: {local.get('pdf_count')}",
        f"- xml_count: {local.get('xml_count')}",
        f"- html_count: {local.get('html_count')}",
        f"- failed_fulltext_fetches_count: {local.get('failed_fulltext_fetches_count')}",
        "",
        "## Interpretation",
        "- Source-level counts are search counts and may include duplicates or broader matches.",
        "- OA count is not the same as successful legal full-text collection count.",
        "- Local corpus counts are after the Agent's filtering, deduplication, OA audit, and QA.",
        "- OpenAlex UI about counts are approximate and can differ from API meta counts.",
        "- OpenAlex UI boolean search and API title_and_abstract.search filters are not always identical.",
        "- Boolean operators, quoted terms, field selection, and NOT conditions can shift counts between UI and API.",
    ]
    report = base / "literature_count_summary_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report.resolve())


def summarize_literature_counts(
    topic: str,
    query: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    output_dir: str | None = None,
    include_openalex: bool = True,
    include_pubmed: bool = True,
    include_local: bool = True,
    *,
    openalex_session: Any = None,
    pubmed_session: Any = None,
) -> dict[str, Any]:
    query_used = default_count_query(topic, query)
    year_filter = "all_years" if year_from is None and year_to is None else f"{year_from or 'any'}-{year_to or 'any'}"
    openalex = _openalex_summary(query_used, year_from, year_to, session=openalex_session) if include_openalex else {"status": "skipped"}
    pubmed = _pubmed_summary(query_used, year_from, year_to, session=pubmed_session) if include_pubmed else {"status": "skipped"}
    local = _local_summary(output_dir) if include_local else {"status": "skipped"}
    statuses = [item.get("status") for item in (openalex, pubmed, local) if item.get("status") != "skipped"]
    status = "ok" if all(item == "ok" for item in statuses) else ("error" if all(item == "error" for item in statuses) else "partial")
    summary = {
        "status": status,
        "intent": "count_summary",
        "topic": topic,
        "query_used": query_used,
        "year_filter": year_filter,
        "openalex": openalex,
        "openalex_debug_modes": openalex.get("debug_modes") or {},
        "pubmed": pubmed,
        "local": local,
        "notes": [
            "OpenAlex/PubMed counts are source-level search counts and may include duplicates or broader matches.",
            "Local corpus counts are after your Agent's filtering, deduplication, and QA.",
        ],
        "actual_downloads": 0,
        "actual_fulltexts": 0,
    }
    report = _write_report(output_dir, summary)
    summary["artifacts"] = {"literature_count_summary_report": report}
    return summary


def write_count_summary_json(output_dir: str, summary: dict[str, Any]) -> str:
    path = Path(output_dir) / "literature_count_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())
