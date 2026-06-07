"""Unpaywall OA confirmation adapter.

ROUND 005 uses Unpaywall only to confirm OA evidence and create candidates.
No PDF is downloaded from Unpaywall results.
"""

from __future__ import annotations

import time
import re
from typing import Any
from urllib.parse import quote

from ..legality import assess_unpaywall_oa
from ..manifest import read_jsonl, write_failure, write_search_log
from ..metadata import normalize_doi
from .common import safe_get_json


UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2"
LAST_SEARCH_INFO: dict[str, Any] = {}
SAFE_PLACEHOLDER_EMAIL = "contact@example.org"


def build_unpaywall_url(doi: str) -> str:
    return f"{UNPAYWALL_ENDPOINT}/{quote(normalize_doi(doi), safe='')}"


def build_unpaywall_params(config: dict[str, Any]) -> dict[str, str]:
    return {"email": str(config.get("contact_email") or SAFE_PLACEHOLDER_EMAIL)}


def _pmcid_from_oa_locations(locations: list[dict[str, Any]]) -> str:
    for location in locations:
        values = [
            str(location.get("url") or ""),
            str(location.get("url_for_pdf") or ""),
            str(location.get("url_for_landing_page") or ""),
            str(location.get("pmh_id") or ""),
        ]
        for value in values:
            match = re.search(r"\bPMC\d+\b", value, re.I)
            if match:
                return match.group(0).upper()
    return ""


def parse_unpaywall_response(payload: dict[str, Any], query_id: str = "") -> dict[str, Any]:
    best = payload.get("best_oa_location") or {}
    locations = payload.get("oa_locations") or []
    license_value = str(best.get("license") or payload.get("license") or "")
    url = str(best.get("url") or "")
    url_for_pdf = str(best.get("url_for_pdf") or "")
    result = {
        "query_id": query_id,
        "doi": normalize_doi(str(payload.get("doi") or "")),
        "pmcid": _pmcid_from_oa_locations(locations),
        "source": "unpaywall",
        "sources": ["unpaywall"],
        "unpaywall_checked": True,
        "is_oa": payload.get("is_oa"),
        "unpaywall_is_oa": payload.get("is_oa"),
        "oa_status": payload.get("oa_status"),
        "unpaywall_oa_status": payload.get("oa_status"),
        "best_oa_location": url_for_pdf or url,
        "unpaywall_best_oa_location": best,
        "oa_locations": locations,
        "license": license_value,
        "unpaywall_license": license_value,
        "host_type": best.get("host_type"),
        "version": best.get("version"),
        "url": url,
        "url_for_pdf": url_for_pdf,
        "pdf_url_candidate": url_for_pdf,
        "landing_url": url,
        "evidence": best.get("evidence") or payload.get("oa_status") or "",
    }
    result.update(assess_unpaywall_oa(result))
    return result


def missing_doi_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": record.get("query_id", ""),
        "title": record.get("title", ""),
        "doi": "",
        "journal": record.get("journal", ""),
        "publication_year": record.get("publication_year"),
        "publication_date": record.get("publication_date", ""),
        "source": "unpaywall",
        "candidate_url": "",
        "pdf_url_candidate": "",
        "landing_url": record.get("landing_url", ""),
        "license": "",
        "oa_status": "",
        "host_type": "",
        "version": "",
        "reason": "missing_doi_for_unpaywall_check",
        "suggested_manual_check": "Find a DOI or confirm OA status through another allowed source before any future download.",
        "is_legal_oa_candidate": False,
        "is_download_allowed_now": False,
    }


def candidate_from_result(result: dict[str, Any], source_record: dict[str, Any] | None = None) -> dict[str, Any]:
    source_record = source_record or {}
    reason = result.get("reason") or "unpaywall_no_confirmed_oa_location"
    candidate_url = result.get("url_for_pdf") or result.get("url") or result.get("best_oa_location") or ""
    return {
        "query_id": result.get("query_id") or source_record.get("query_id", ""),
        "title": source_record.get("title", ""),
        "doi": result.get("doi", ""),
        "pmid": result.get("pmid") or source_record.get("pmid", ""),
        "pmcid": result.get("pmcid") or source_record.get("pmcid", ""),
        "journal": source_record.get("journal", ""),
        "publication_year": source_record.get("publication_year"),
        "publication_date": source_record.get("publication_date", ""),
        "source": "unpaywall",
        "candidate_url": candidate_url,
        "pdf_url_candidate": result.get("url_for_pdf", ""),
        "landing_url": result.get("url", "") or source_record.get("landing_url", ""),
        "license": result.get("license", ""),
        "oa_status": result.get("oa_status", ""),
        "host_type": result.get("host_type", ""),
        "version": result.get("version", ""),
        "reason": reason,
        "suggested_manual_check": "Use this as OA evidence only; a later download round must re-check legality before saving any PDF.",
        "is_legal_oa_candidate": result.get("is_legal_oa_candidate", False),
        "is_download_allowed_now": False,
    }


def check_doi(doi: str, config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None, query_id: str = "") -> dict[str, Any]:
    doi = normalize_doi(doi)
    if not doi:
        return {"doi": "", "reason": "missing_doi_for_unpaywall_check", "is_download_allowed_now": False}
    if dry_run or not allow_network:
        result = {
            "query_id": query_id,
            "doi": doi,
            "source": "unpaywall",
            "sources": ["unpaywall"],
            "unpaywall_checked": False,
            "is_oa": None,
            "unpaywall_is_oa": None,
            "oa_status": "",
            "unpaywall_oa_status": "",
            "license": "",
            "unpaywall_license": "",
            "url": "",
            "url_for_pdf": "",
            "reason": "unpaywall_dry_run_not_checked",
            "is_legal_oa_candidate": False,
            "is_download_allowed_now": False,
        }
        return result
    try:
        payload, network_calls = safe_get_json(build_unpaywall_url(doi), params=build_unpaywall_params(config), source="unpaywall", session=session)
        result = parse_unpaywall_response(payload, query_id=query_id)
        result["network_calls"] = network_calls
        return result
    except Exception as exc:
        write_failure("unpaywall oa check failed", {"doi": doi, "error": str(exc)})
        return {"doi": doi, "source": "unpaywall", "reason": "unpaywall_check_failed", "error": str(exc), "is_download_allowed_now": False, "network_calls": 0}


def collect_oa_check_inputs(max_results: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl("metadata_results.jsonl") + read_jsonl("candidates.jsonl")
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for record in records:
        doi = normalize_doi(str(record.get("doi") or ""))
        key = doi or f"missing:{record.get('title','')}:{record.get('candidate_url','')}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if max_results and len(selected) >= max_results:
            break
    return selected


def check_records(records: list[dict[str, Any]], config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None, max_results: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    network_calls = 0
    missing_doi_count = 0
    confirmed_oa_count = 0
    no_oa_count = 0
    for record in records[: max_results or len(records)]:
        doi = normalize_doi(str(record.get("doi") or ""))
        if not doi:
            missing_doi_count += 1
            candidates.append(missing_doi_candidate(record))
            continue
        result = check_doi(doi, config, dry_run=dry_run, allow_network=allow_network, session=session, query_id=str(record.get("query_id") or "oa_check"))
        network_calls += int(result.get("network_calls", 0))
        result.update(
            {
                "title": record.get("title", ""),
                "pmid": record.get("pmid") or result.get("pmid", ""),
                "pmcid": record.get("pmcid") or result.get("pmcid", ""),
                "journal": record.get("journal", ""),
                "publication_year": record.get("publication_year"),
                "publication_date": record.get("publication_date", ""),
                "landing_url": result.get("landing_url") or record.get("landing_url", ""),
                "unpaywall_checked": not (dry_run or not allow_network),
            }
        )
        checked.append(result)
        if result.get("is_legal_oa_candidate"):
            confirmed_oa_count += 1
        elif result.get("reason") != "unpaywall_dry_run_not_checked":
            no_oa_count += 1
        candidates.append(candidate_from_result(result, source_record=record))
    info = {
        "checked_doi_count": len(checked),
        "confirmed_oa_count": confirmed_oa_count,
        "no_oa_count": no_oa_count,
        "missing_doi_count": missing_doi_count,
        "candidates_written": len(candidates),
        "network_calls": network_calls,
        "dry_run": dry_run or not allow_network,
    }
    global LAST_SEARCH_INFO
    LAST_SEARCH_INFO = info
    write_search_log(dict(event="unpaywall_oa_check", source="unpaywall", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info))
    return checked, candidates, info


def search(query_spec: Any, config: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> list[dict[str, Any]]:
    if getattr(query_spec, "doi", ""):
        return [check_doi(query_spec.doi, config, dry_run=dry_run, allow_network=allow_network, session=session, query_id=query_spec.input_id)]
    return []
