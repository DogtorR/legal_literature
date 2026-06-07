"""OpenAlex metadata adapter.

OpenAlex is used here only for public metadata and OA-location clues. PDF URLs
from OpenAlex are treated as candidates requiring confirmation by a stronger
legal OA source before any future download.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..filters import apply_query_filters
from ..manifest import read_jsonl, write_failure, write_search_log
from ..metadata import MetadataRecord, deduplicate_records, normalize_doi
from ..query import QuerySpec
from .common import require_keyword_match


OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_USER_AGENT = "LEGAL_LITERATURE_AGENT/0.1 (mailto:contact@example.org)"
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
LAST_SEARCH_INFO: dict[str, Any] = {}


def build_openalex_params(query_spec: QuerySpec, max_results: int | None = None) -> dict[str, str]:
    params: dict[str, str] = {}
    filters: list[str] = []
    per_page = max_results or query_spec.max_results or 25
    params["per-page"] = str(max(1, min(int(per_page), 200)))

    if query_spec.doi:
        filters.append(f"doi:{normalize_doi(query_spec.doi)}")
    elif query_spec.title:
        params["search"] = query_spec.title
    elif query_spec.keywords:
        params["search"] = " ".join(query_spec.keywords)

    if query_spec.publication_date_from:
        filters.append(f"from_publication_date:{query_spec.publication_date_from}")
    elif query_spec.year_from is not None:
        filters.append(f"from_publication_date:{query_spec.year_from}-01-01")
    if query_spec.publication_date_to:
        filters.append(f"to_publication_date:{query_spec.publication_date_to}")
    elif query_spec.year_to is not None:
        filters.append(f"to_publication_date:{query_spec.year_to}-12-31")

    if query_spec.publication_type:
        mapped = _map_publication_type(query_spec.publication_type)
        if mapped:
            filters.append(f"type:{mapped}")

    if filters:
        params["filter"] = ",".join(filters)
    return params


def _map_publication_type(publication_type: str) -> str:
    text = publication_type.lower()
    if "journal" in text or "article" in text:
        return "article"
    if "preprint" in text:
        return "preprint"
    return ""


def restore_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned.append((int(position), word))
    return " ".join(word for _, word in sorted(positioned))


def _first_author_names(work: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(str(name))
    return names


def _location_url(location: dict[str, Any] | None) -> str:
    if not location:
        return ""
    return str(location.get("pdf_url") or location.get("landing_page_url") or "")


def _source_name(location: dict[str, Any] | None) -> str:
    if not location:
        return ""
    source = location.get("source") or {}
    return str(source.get("display_name") or "")


def _pmid_from_ids(ids: dict[str, Any]) -> str:
    pmid = str(ids.get("pmid") or "")
    return pmid.rsplit("/", 1)[-1] if pmid else ""


def openalex_work_to_metadata(work: dict[str, Any], query_spec: QuerySpec) -> dict[str, Any]:
    ids = work.get("ids") or {}
    primary_location = work.get("primary_location") or {}
    best_oa_location = work.get("best_oa_location") or {}
    open_access = work.get("open_access") or {}
    doi = normalize_doi(str(work.get("doi") or ids.get("doi") or ""))
    record = MetadataRecord(
        query_id=query_spec.input_id,
        title=str(work.get("title") or work.get("display_name") or ""),
        doi=doi,
        pmid=_pmid_from_ids(ids),
        openalex_id=str(work.get("id") or ids.get("openalex") or ""),
        publication_year=work.get("publication_year"),
        publication_date=str(work.get("publication_date") or ""),
        publication_type=str(work.get("type") or ""),
        journal=_source_name(primary_location),
        authors=_first_author_names(work),
        abstract=restore_abstract(work.get("abstract_inverted_index")),
        source="openalex",
        landing_url=str(primary_location.get("landing_page_url") or work.get("id") or ""),
        oa_status=str(open_access.get("oa_status") or ""),
        best_oa_location=_location_url(best_oa_location),
        pdf_url_candidate=str(best_oa_location.get("pdf_url") or primary_location.get("pdf_url") or ""),
        license=str(best_oa_location.get("license") or primary_location.get("license") or ""),
        score=float(work.get("relevance_score") or 0.0),
        sources=["openalex"],
        oa_evidence=["openalex metadata only"],
    ).to_dict()
    record["primary_location"] = {
        "landing_page_url": primary_location.get("landing_page_url"),
        "pdf_url": primary_location.get("pdf_url"),
        "license": primary_location.get("license"),
        "source": _source_name(primary_location),
    }
    record["open_access"] = {
        "is_oa": open_access.get("is_oa"),
        "oa_status": open_access.get("oa_status"),
        "oa_url": open_access.get("oa_url"),
    }
    record["raw_source_summary"] = {
        "cited_by_count": work.get("cited_by_count"),
        "host_venue": _source_name(primary_location),
    }
    return record


def candidate_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    candidate_url = record.get("pdf_url_candidate") or record.get("best_oa_location")
    if not candidate_url:
        return None
    return {
        "query_id": record.get("query_id", ""),
        "title": record.get("title", ""),
        "doi": record.get("doi", ""),
        "openalex_id": record.get("openalex_id", ""),
        "journal": record.get("journal", ""),
        "publication_year": record.get("publication_year"),
        "publication_date": record.get("publication_date", ""),
        "candidate_url": candidate_url,
        "landing_url": record.get("landing_url", ""),
        "source": "openalex",
        "reason": "openalex_oa_location_requires_confirmation",
        "suggested_manual_check": "Confirm through PMC, publisher explicit OA, Unpaywall confirmed OA, or another allowed source before download.",
    }


def candidates_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        candidate = candidate_from_record(record)
        if candidate:
            candidates.append(candidate)
    return candidates


def _dry_run_records(query_spec: QuerySpec) -> list[dict[str, Any]]:
    path = Path("examples/mock_metadata.jsonl")
    if not path.exists():
        return []
    records = read_jsonl(path)
    return [
        dict(record, source="openalex", sources=["openalex"], query_id=query_spec.input_id)
        for record in records
    ]


def _http_get_json(params: dict[str, str], headers: dict[str, str], timeout: int, session: Any = None) -> tuple[dict[str, Any], int]:
    if session is not None:
        response = session.get(OPENALEX_WORKS_ENDPOINT, params=params, headers=headers, timeout=timeout)
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise RuntimeError(f"OpenAlex HTTP {status_code}")
        return response.json(), 1

    url = f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(params)}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, 1


def _fetch_with_retries(params: dict[str, str], headers: dict[str, str], timeout: int, session: Any = None) -> tuple[dict[str, Any], int]:
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        attempts += 1
        try:
            payload, calls = _http_get_json(params, headers, timeout, session=session)
            return payload, calls
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(0.25 * attempt, 1.0))
    assert last_error is not None
    raise last_error


def search(query_spec: QuerySpec, config: dict[str, Any], *, dry_run: bool = True, session: Any = None, allow_network: bool = False) -> list[dict[str, Any]]:
    """Search OpenAlex metadata or return dry-run mock metadata without networking."""

    global LAST_SEARCH_INFO
    max_results = int(query_spec.max_results or config.get("max_results_per_source") or 25)
    params = build_openalex_params(query_spec, max_results)
    raw_records: list[dict[str, Any]] = []
    network_calls = 0
    total_count: int | None = None
    page_count = 0
    has_more = False
    next_cursor: str | None = None
    errors: list[str] = []
    if dry_run or not allow_network:
        raw_records = _dry_run_records(query_spec)
        total_count = len(raw_records)
        page_count = 1 if raw_records else 0
    else:
        try:
            headers = {"User-Agent": str(config.get("user_agent") or DEFAULT_USER_AGENT)}
            cursor = str(config.get("next_cursor") or config.get("start_cursor") or "*")
            while len(raw_records) < max_results:
                page_params = dict(params)
                page_params["per-page"] = str(min(200, max_results - len(raw_records)))
                page_params["cursor"] = cursor
                payload, calls = _fetch_with_retries(page_params, headers, DEFAULT_TIMEOUT, session=session)
                network_calls += calls
                meta = payload.get("meta") or {}
                if total_count is None:
                    total_count = int(meta.get("count") or 0)
                works = payload.get("results") or []
                page_count += 1
                raw_records.extend(openalex_work_to_metadata(work, query_spec) for work in works)
                next_cursor = meta.get("next_cursor")
                if not works or not next_cursor or len(raw_records) >= int(total_count or 0):
                    break
                cursor = str(next_cursor)
            raw_records = raw_records[:max_results]
            has_more = bool(total_count is not None and len(raw_records) < total_count and next_cursor)
        except Exception as exc:
            errors.append(str(exc))
            write_failure("openalex metadata request failed", {"query_id": query_spec.input_id, "error": str(exc), "params": params})
            raw_records = []

    filtered = apply_query_filters(raw_records, query_spec, require_keyword_match=require_keyword_match(config))
    deduped = deduplicate_records(filtered)
    LAST_SEARCH_INFO = {
        "source": "openalex",
        "query_id": query_spec.input_id,
        "query_summary": params,
        "raw_count": len(raw_records),
        "total_count": total_count,
        "retrieved_count": len(raw_records),
        "page_count": page_count,
        "has_more": has_more,
        "next_cursor": next_cursor if has_more else None,
        "filtered_count": len(filtered),
        "deduped_count": len(deduped),
        "year_from": query_spec.year_from,
        "year_to": query_spec.year_to,
        "keywords": query_spec.keywords,
        "include_terms": query_spec.include_terms,
        "exclude_terms": query_spec.exclude_terms,
        "network_calls": network_calls,
        "dry_run": dry_run or not allow_network,
        "errors": errors,
    }
    write_search_log(dict(event="openalex_search", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **LAST_SEARCH_INFO))
    return deduped
