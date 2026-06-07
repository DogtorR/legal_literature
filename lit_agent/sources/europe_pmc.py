"""Europe PMC metadata adapter."""

from __future__ import annotations

import time
from typing import Any

from ..filters import apply_query_filters
from ..manifest import write_search_log
from ..metadata import MetadataRecord, deduplicate_records, normalize_doi
from ..query import QuerySpec
from .common import dry_run_mock_records, require_keyword_match, safe_get_json


EUROPE_PMC_SEARCH_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
LAST_SEARCH_INFO: dict[str, Any] = {}


def build_europe_pmc_params(query_spec: QuerySpec, max_results: int | None = None) -> dict[str, str]:
    clauses: list[str] = []
    if query_spec.doi:
        clauses.append(f'DOI:"{normalize_doi(query_spec.doi)}"')
    elif query_spec.pmid:
        clauses.append(f"EXT_ID:{query_spec.pmid}")
    elif query_spec.title:
        clauses.append(f'TITLE:"{query_spec.title}"')
    elif query_spec.keywords:
        clauses.append(" ".join(query_spec.keywords))
    else:
        clauses.append("*")
    if query_spec.journal:
        clauses.append(f'JOURNAL:"{query_spec.journal}"')
    if query_spec.publication_date_from or query_spec.publication_date_to:
        start = query_spec.publication_date_from or "1800-01-01"
        end = query_spec.publication_date_to or "2200-12-31"
        clauses.append(f"FIRST_PDATE:[{start} TO {end}]")
    elif query_spec.year_from is not None or query_spec.year_to is not None:
        start = query_spec.year_from or 1800
        end = query_spec.year_to or 2200
        clauses.append(f"FIRST_PDATE:[{start}-01-01 TO {end}-12-31]")
    return {"query": " AND ".join(clauses), "format": "json", "pageSize": str(max(1, min(int(max_results or query_spec.max_results or 25), 1000)))}


def _fulltext_urls(item: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    entries = ((item.get("fullTextUrlList") or {}).get("fullTextUrl") or [])
    for entry in entries:
        url = entry.get("url")
        if url:
            urls.append(str(url))
    return urls


def _publication_date(item: dict[str, Any]) -> str:
    for key in ("firstPublicationDate", "electronicPublicationDate", "printPublicationDate"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    journal_info = item.get("journalInfo") or {}
    return str(journal_info.get("dateOfPublication") or "").strip()


def europe_pmc_item_to_metadata(item: dict[str, Any], query_spec: QuerySpec) -> dict[str, Any]:
    urls = _fulltext_urls(item)
    doi = normalize_doi(str(item.get("doi") or ""))
    pmcid = str(item.get("pmcid") or "")
    landing = f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}"
    record = MetadataRecord(
        query_id=query_spec.input_id,
        title=str(item.get("title") or ""),
        doi=doi,
        pmid=str(item.get("pmid") or item.get("id") or ""),
        publication_year=int(item["pubYear"]) if str(item.get("pubYear") or "").isdigit() else None,
        publication_date=_publication_date(item),
        publication_type=str(item.get("pubType") or ""),
        journal=str(item.get("journalTitle") or ""),
        authors=[name.strip() for name in str(item.get("authorString") or "").split(",") if name.strip()],
        abstract=str(item.get("abstractText") or ""),
        source="europe_pmc",
        landing_url=landing,
        oa_status="open" if str(item.get("isOpenAccess") or "").upper() == "Y" else "",
        best_oa_location=urls[0] if urls else (f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else ""),
        pdf_url_candidate=next((url for url in urls if url.lower().endswith(".pdf") or "pdf" in url.lower()), ""),
        license=str(item.get("license") or ""),
        sources=["europe_pmc"],
        oa_evidence=["europe pmc metadata only"],
    ).to_dict()
    record["pmcid"] = pmcid
    record["raw_source_summary"] = {"source": item.get("source"), "fulltext_urls": len(urls)}
    return record


def candidates_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        url = record.get("pdf_url_candidate") or record.get("best_oa_location")
        if url or record.get("pmcid"):
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
                    "source": "europe_pmc",
                    "candidate_url": url or f"https://europepmc.org/article/PMC/{record.get('pmcid')}",
                    "landing_url": record.get("landing_url", ""),
                    "reason": "europe_pmc_fulltext_requires_download_round_confirmation",
                    "suggested_manual_check": "Confirm current Europe PMC/PMC open full text policy before any future download.",
                }
            )
    return candidates


def search(query_spec: QuerySpec, config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> list[dict[str, Any]]:
    global LAST_SEARCH_INFO
    max_results = int(query_spec.max_results or config.get("max_results_per_source") or 25)
    params = build_europe_pmc_params(query_spec, max_results)
    network_calls = 0
    total_count: int | None = None
    page_count = 0
    has_more = False
    next_cursor: str | None = None
    errors: list[str] = []
    if dry_run or not allow_network:
        raw_records = dry_run_mock_records("europe_pmc", query_spec.input_id)
        total_count = len(raw_records)
        page_count = 1 if raw_records else 0
    else:
        try:
            raw_records = []
            cursor = str(config.get("next_cursor") or config.get("start_cursor") or "")
            while len(raw_records) < max_results:
                page_params = dict(params)
                page_params["pageSize"] = str(min(1000, max_results - len(raw_records)))
                if cursor:
                    page_params["cursorMark"] = cursor
                payload, calls = safe_get_json(EUROPE_PMC_SEARCH_ENDPOINT, params=page_params, source="europe_pmc", session=session)
                network_calls += calls
                total_count = int(payload.get("hitCount") or total_count or 0)
                items = (payload.get("resultList") or {}).get("result") or []
                page_count += 1
                raw_records.extend(europe_pmc_item_to_metadata(item, query_spec) for item in items)
                next_cursor = payload.get("nextCursorMark")
                if not items or not next_cursor or next_cursor == cursor or len(raw_records) >= int(total_count or 0):
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
        "source": "europe_pmc",
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
        "network_calls": network_calls,
        "dry_run": dry_run or not allow_network,
        "errors": errors,
        "year_from": query_spec.year_from,
        "year_to": query_spec.year_to,
        "keywords": query_spec.keywords,
        "include_terms": query_spec.include_terms,
        "exclude_terms": query_spec.exclude_terms,
    }
    write_search_log(dict(event="europe_pmc_search", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **LAST_SEARCH_INFO))
    return deduped
