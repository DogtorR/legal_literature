"""Crossref metadata adapter."""

from __future__ import annotations

import time
from typing import Any

from ..filters import apply_query_filters
from ..manifest import write_search_log
from ..metadata import MetadataRecord, deduplicate_records, normalize_doi
from ..query import QuerySpec
from .common import dry_run_mock_records, require_keyword_match, safe_get_json


CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/works"
LAST_SEARCH_INFO: dict[str, Any] = {}


def build_crossref_request(query_spec: QuerySpec, max_results: int | None = None) -> tuple[str, dict[str, str]]:
    if query_spec.doi:
        return f"{CROSSREF_WORKS_ENDPOINT}/{normalize_doi(query_spec.doi)}", {}
    params: dict[str, str] = {"rows": str(max(1, min(int(max_results or query_spec.max_results or 25), 1000)))}
    if query_spec.title:
        params["query.title"] = query_spec.title
    elif query_spec.keywords:
        params["query.bibliographic"] = " ".join(query_spec.keywords)
    if query_spec.journal:
        params["query.container-title"] = query_spec.journal
    filters: list[str] = []
    if query_spec.publication_date_from:
        filters.append(f"from-pub-date:{query_spec.publication_date_from}")
    elif query_spec.year_from is not None:
        filters.append(f"from-pub-date:{query_spec.year_from}-01-01")
    if query_spec.publication_date_to:
        filters.append(f"until-pub-date:{query_spec.publication_date_to}")
    elif query_spec.year_to is not None:
        filters.append(f"until-pub-date:{query_spec.year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    return CROSSREF_WORKS_ENDPOINT, params


def _date_parts(item: dict[str, Any]) -> list[int]:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return [int(part) for part in parts[0] if str(part).isdigit()]
    return []


def _date_year(item: dict[str, Any]) -> int | None:
    parts = _date_parts(item)
    if parts:
        return int(parts[0])
    return None


def _date_string(item: dict[str, Any]) -> str:
    parts = _date_parts(item)
    if not parts:
        return ""
    year = parts[0]
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _title(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _authors(item: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return authors


def _license(item: dict[str, Any]) -> str:
    licenses = item.get("license") or []
    urls = [str(entry.get("URL")) for entry in licenses if entry.get("URL")]
    return ";".join(urls)


def _pdf_link(item: dict[str, Any]) -> str:
    for link in item.get("link") or []:
        url = str(link.get("URL") or "")
        content_type = str(link.get("content-type") or "").lower()
        if url and ("pdf" in content_type or url.lower().endswith(".pdf")):
            return url
    return ""


def crossref_item_to_metadata(item: dict[str, Any], query_spec: QuerySpec) -> dict[str, Any]:
    doi = normalize_doi(str(item.get("DOI") or ""))
    record = MetadataRecord(
        query_id=query_spec.input_id,
        title=_title(item.get("title")),
        doi=doi,
        publication_year=_date_year(item),
        publication_date=_date_string(item),
        publication_type=str(item.get("type") or ""),
        journal=_title(item.get("container-title")),
        authors=_authors(item),
        abstract=str(item.get("abstract") or ""),
        source="crossref",
        landing_url=str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
        oa_status="",
        best_oa_location="",
        pdf_url_candidate=_pdf_link(item),
        license=_license(item),
        score=float(item.get("score") or 0.0),
        sources=["crossref"],
        oa_evidence=["crossref metadata only"],
    ).to_dict()
    record["raw_source_summary"] = {"publisher": item.get("publisher"), "reference_count": item.get("reference-count")}
    return record


def candidates_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("pdf_url_candidate"):
            candidates.append(
                {
                    "query_id": record.get("query_id", ""),
                    "title": record.get("title", ""),
                    "doi": record.get("doi", ""),
                    "pmid": record.get("pmid", ""),
                    "pmcid": record.get("pmcid", ""),
                    "journal": record.get("journal", ""),
                    "publication_year": record.get("publication_year"),
                    "publication_date": record.get("publication_date", ""),
                    "source": "crossref",
                    "candidate_url": record.get("pdf_url_candidate", ""),
                    "landing_url": record.get("landing_url", ""),
                    "reason": "crossref_pdf_link_requires_confirmation",
                    "suggested_manual_check": "Confirm legal OA through publisher explicit OA, Unpaywall confirmed OA, PMC, or another allowed source before download.",
                }
            )
    return candidates


def search(query_spec: QuerySpec, config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> list[dict[str, Any]]:
    global LAST_SEARCH_INFO
    max_results = int(query_spec.max_results or config.get("max_results_per_source") or 25)
    url, params = build_crossref_request(query_spec, max_results)
    network_calls = 0
    total_count: int | None = None
    page_count = 0
    has_more = False
    next_cursor: str | None = None
    errors: list[str] = []
    if dry_run or not allow_network:
        raw_records = dry_run_mock_records("crossref", query_spec.input_id)
        total_count = len(raw_records)
        page_count = 1 if raw_records else 0
    else:
        try:
            raw_records = []
            if query_spec.doi:
                payload, network_calls = safe_get_json(url, params=params, source="crossref", session=session)
                message = payload.get("message") or {}
                raw_records = [crossref_item_to_metadata(message, query_spec)] if message else []
                total_count = len(raw_records)
                page_count = 1 if raw_records else 0
            else:
                cursor = str(config.get("next_cursor") or config.get("start_cursor") or "*")
                while len(raw_records) < max_results:
                    page_params = dict(params)
                    page_params["rows"] = str(min(1000, max_results - len(raw_records)))
                    page_params["cursor"] = cursor
                    payload, calls = safe_get_json(url, params=page_params, source="crossref", session=session)
                    network_calls += calls
                    message = payload.get("message") or {}
                    if total_count is None:
                        total_count = int(message.get("total-results") or 0)
                    items = message.get("items") or []
                    page_count += 1
                    raw_records.extend(crossref_item_to_metadata(item, query_spec) for item in items)
                    next_cursor = message.get("next-cursor")
                    if not items or not next_cursor or len(raw_records) >= int(total_count or 0):
                        break
                    cursor = str(next_cursor)
                raw_records = raw_records[:max_results]
                has_more = bool(total_count is not None and len(raw_records) < total_count and next_cursor)
        except Exception as exc:
            errors.append(str(exc))
            raw_records = []
    filtered = apply_query_filters(raw_records, query_spec, require_keyword_match=require_keyword_match(config))
    deduped = deduplicate_records(filtered)
    LAST_SEARCH_INFO = {
        "source": "crossref",
        "query_id": query_spec.input_id,
        "query_summary": {"url": url, "params": params},
        "raw_count": len(raw_records),
        "total_count": total_count,
        "retrieved_count": len(raw_records),
        "page_count": page_count,
        "has_more": has_more,
        "next_cursor": next_cursor if has_more else None,
        "filtered_count": len(filtered),
        "deduped_count": len(deduped),
        "network_calls": network_calls,
        "dry_run": dry_run or not allow_network,
        "errors": errors,
        "year_from": query_spec.year_from,
        "year_to": query_spec.year_to,
        "keywords": query_spec.keywords,
        "include_terms": query_spec.include_terms,
        "exclude_terms": query_spec.exclude_terms,
    }
    write_search_log(dict(event="crossref_search", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **LAST_SEARCH_INFO))
    return deduped
