"""PubMed metadata adapter using public NCBI E-utilities structure."""

from __future__ import annotations

import time
from typing import Any

from ..filters import apply_query_filters
from ..manifest import write_search_log
from ..metadata import MetadataRecord, deduplicate_records, normalize_doi
from ..query import QuerySpec
from .common import dry_run_mock_records, require_keyword_match, safe_get_json


PUBMED_ESEARCH_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
LAST_SEARCH_INFO: dict[str, Any] = {}


def build_pubmed_esearch_params(query_spec: QuerySpec, max_results: int | None = None, retstart: int = 0) -> dict[str, str]:
    terms: list[str] = []
    if query_spec.pmid:
        terms.append(f"{query_spec.pmid}[uid]")
    elif query_spec.doi:
        terms.append(f"{normalize_doi(query_spec.doi)}[doi]")
    elif query_spec.title:
        terms.append(f'"{query_spec.title}"[Title]')
    elif query_spec.keywords:
        terms.append(" ".join(query_spec.keywords))
    else:
        terms.append("all[sb]")
    if query_spec.journal:
        terms.append(f'"{query_spec.journal}"[Journal]')
    if query_spec.publication_date_from or query_spec.publication_date_to:
        start = (query_spec.publication_date_from or "1800-01-01").replace("-", "/")
        end = (query_spec.publication_date_to or "2200-12-31").replace("-", "/")
        terms.append(f'("{start}"[PDAT] : "{end}"[PDAT])')
    elif query_spec.year_from is not None or query_spec.year_to is not None:
        start = query_spec.year_from or 1800
        end = query_spec.year_to or 2200
        terms.append(f'("{start}"[PDAT] : "{end}"[PDAT])')
    return {
        "db": "pubmed",
        "retmode": "json",
        "retmax": str(max(1, min(int(max_results or query_spec.max_results or 25), 500))),
        "retstart": str(max(0, int(retstart))),
        "term": " AND ".join(terms),
        "tool": "LEGAL_LITERATURE_AGENT",
        "email": "contact@example.org",
    }


def _article_ids(item: dict[str, Any]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for article_id in item.get("articleids") or []:
        idtype = str(article_id.get("idtype") or "").lower()
        value = str(article_id.get("value") or "")
        if idtype and value:
            ids[idtype] = value
    return ids


def _pub_year(pubdate: str) -> int | None:
    for token in str(pubdate or "").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def pubmed_summary_to_metadata(item: dict[str, Any], query_spec: QuerySpec) -> dict[str, Any]:
    article_ids = _article_ids(item)
    pmid = str(item.get("uid") or article_ids.get("pubmed") or "")
    pmcid = article_ids.get("pmc", "")
    record = MetadataRecord(
        query_id=query_spec.input_id,
        title=str(item.get("title") or ""),
        doi=normalize_doi(article_ids.get("doi", "")),
        pmid=pmid,
        publication_year=_pub_year(str(item.get("pubdate") or "")),
        publication_date=str(item.get("pubdate") or ""),
        publication_type=";".join(str(pubtype) for pubtype in item.get("pubtype") or []),
        journal=str(item.get("fulljournalname") or item.get("source") or ""),
        authors=[str(author.get("name")) for author in item.get("authors") or [] if author.get("name")],
        abstract="",
        source="pubmed",
        landing_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        best_oa_location=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
        sources=["pubmed"],
        oa_evidence=["pubmed metadata only"],
    ).to_dict()
    record["pmcid"] = pmcid
    record["raw_source_summary"] = {"articleids": article_ids}
    return record


def candidates_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("pmcid") or record.get("best_oa_location"):
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
                    "source": "pubmed",
                    "candidate_url": record.get("best_oa_location", ""),
                    "landing_url": record.get("landing_url", ""),
                    "reason": "pubmed_pmcid_requires_pmc_confirmation",
                    "suggested_manual_check": "Confirm PMC/Europe PMC open full text status before any future download.",
                }
            )
    return candidates


def search(query_spec: QuerySpec, config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> list[dict[str, Any]]:
    global LAST_SEARCH_INFO
    max_results = int(query_spec.max_results or config.get("max_results_per_source") or 25)
    page_size = min(max_results, 500)
    params = build_pubmed_esearch_params(query_spec, page_size, retstart=0)
    network_calls = 0
    total_count: int | None = None
    page_count = 0
    has_more = False
    errors: list[str] = []
    if dry_run or not allow_network:
        raw_records = dry_run_mock_records("pubmed", query_spec.input_id)
        total_count = len(raw_records)
        page_count = 1 if raw_records else 0
    else:
        try:
            ids: list[str] = []
            retstart = max(0, int(config.get("retstart") or config.get("next_offset") or config.get("start_offset") or 0))
            while len(ids) < max_results:
                page_params = build_pubmed_esearch_params(query_spec, min(page_size, max_results - len(ids)), retstart=retstart)
                search_payload, calls = safe_get_json(PUBMED_ESEARCH_ENDPOINT, params=page_params, source="pubmed", session=session, rate_limit_seconds=0.34)
                network_calls += calls
                result = search_payload.get("esearchresult") or {}
                if total_count is None:
                    total_count = int(result.get("count") or 0)
                page_ids = [str(item) for item in result.get("idlist") or [] if item]
                page_count += 1
                if not page_ids:
                    break
                ids.extend(page_ids)
                retstart += len(page_ids)
                if retstart >= int(total_count or 0):
                    break
            ids = ids[:max_results]
            has_more = bool(total_count is not None and len(ids) < total_count)
            if ids:
                raw_records = []
                for offset in range(0, len(ids), 200):
                    chunk = ids[offset : offset + 200]
                    summary_params = {"db": "pubmed", "retmode": "json", "id": ",".join(chunk), "tool": "LEGAL_LITERATURE_AGENT", "email": "contact@example.org"}
                    summary_payload, calls = safe_get_json(PUBMED_ESUMMARY_ENDPOINT, params=summary_params, source="pubmed", session=session, rate_limit_seconds=0.34)
                    network_calls += calls
                    result = summary_payload.get("result") or {}
                    raw_records.extend(pubmed_summary_to_metadata(result[pmid], query_spec) for pmid in chunk if pmid in result)
            else:
                raw_records = []
        except Exception as exc:
            errors.append(str(exc))
            raw_records = []
    filtered = apply_query_filters(raw_records, query_spec, require_keyword_match=require_keyword_match(config))
    deduped = deduplicate_records(filtered)
    LAST_SEARCH_INFO = {
        "source": "pubmed",
        "query_id": query_spec.input_id,
        "query_summary": params,
        "raw_count": len(raw_records),
        "total_count": total_count,
        "retrieved_count": len(raw_records),
        "page_count": page_count,
        "has_more": has_more,
        "next_cursor": str((int(config.get("retstart") or config.get("next_offset") or config.get("start_offset") or 0) + len(raw_records))) if has_more else None,
        "next_offset": (int(config.get("retstart") or config.get("next_offset") or config.get("start_offset") or 0) + len(raw_records)) if has_more else None,
        "retstart": (int(config.get("retstart") or config.get("next_offset") or config.get("start_offset") or 0) + len(raw_records)) if has_more else None,
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
    write_search_log(dict(event="pubmed_search", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **LAST_SEARCH_INFO))
    return deduped
