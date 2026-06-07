"""OpenAlex OA count calibration helpers.

This module only calls the OpenAlex metadata API. It does not download or save
PDF, HTML, XML, or other full-text content.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .sources.openalex import DEFAULT_USER_AGENT, OPENALEX_WORKS_ENDPOINT


WEB_REFERENCE = {
    "works_approx": 28190,
    "oa_ratio_approx": 0.638,
    "oa_works_approx": 18000,
}


def _looks_boolean_query(query: str) -> bool:
    upper = f" {query.upper()} "
    return any(token in upper for token in (" AND ", " OR ", " NOT ")) or any(char in query for char in "()")


def _params(query: str, year_from: int | None, year_to: int | None, *, is_oa: bool | None = None, per_page: int = 1, cursor: str | None = None) -> dict[str, str]:
    filters: list[str] = []
    params = {
        "per-page": str(max(1, min(int(per_page), 200))),
        "mailto": "contact@example.org",
    }
    if _looks_boolean_query(query):
        filters.append(f"title_and_abstract.search:{query}")
    else:
        params["search"] = query
    if year_from is not None:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if is_oa is not None:
        filters.append(f"open_access.is_oa:{str(is_oa).lower()}")
    if filters:
        params["filter"] = ",".join(filters)
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _api_url(params: dict[str, str]) -> str:
    return f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(params)}"


def _get_json(params: dict[str, str], *, timeout: int = 30, session: Any = None) -> dict[str, Any]:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if session is not None:
        response = session.get(OPENALEX_WORKS_ENDPOINT, params=params, headers=headers, timeout=timeout)
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise RuntimeError(f"OpenAlex HTTP {status_code}")
        return response.json()
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            request = Request(_api_url(params), headers=headers)
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(min(2.0 * attempt, 10.0))
    assert last_error is not None
    raise last_error


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP Error {exc.code}: {body or exc.reason}"
    return str(exc)


def _count(query: str, year_from: int | None, year_to: int | None, *, is_oa: bool | None = None, session: Any = None) -> tuple[int, dict[str, str]]:
    params = _params(query, year_from, year_to, is_oa=is_oa, per_page=1)
    payload = _get_json(params, session=session)
    meta = payload.get("meta") or {}
    return int(meta.get("count") or 0), params


def _location_values(work: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for key in ("primary_location", "best_oa_location"):
        value = work.get(key)
        if isinstance(value, dict):
            locations.append(value)
    for value in work.get("locations") or []:
        if isinstance(value, dict):
            locations.append(value)
    return locations


def _classify_oa_work(work: dict[str, Any]) -> dict[str, Any]:
    open_access = work.get("open_access") or {}
    locations = _location_values(work)
    pdf_urls = [str(location.get("pdf_url") or "") for location in locations if location.get("pdf_url")]
    landing_urls = [str(location.get("landing_page_url") or "") for location in locations if location.get("landing_page_url")]
    licenses = [str(location.get("license") or "") for location in locations if location.get("license")]
    oa_url = str(open_access.get("oa_url") or "")
    return {
        "has_oa_url": bool(oa_url),
        "has_pdf_url": bool(pdf_urls),
        "has_landing_page": bool(landing_urls),
        "has_license": bool(licenses),
        "has_html_xml_or_fulltext_url": bool(landing_urls and not pdf_urls),
        "only_landing_page": bool(landing_urls and not pdf_urls),
        "oa_url": oa_url,
        "pdf_urls": pdf_urls[:5],
        "landing_urls": landing_urls[:5],
        "licenses": sorted(set(licenses)),
    }


def _sample_work(work: dict[str, Any]) -> dict[str, Any]:
    ids = work.get("ids") or {}
    classified = _classify_oa_work(work)
    return {
        "openalex_id": work.get("id") or ids.get("openalex") or "",
        "doi": work.get("doi") or ids.get("doi") or "",
        "title": work.get("title") or work.get("display_name") or "",
        "publication_year": work.get("publication_year"),
        "open_access": work.get("open_access") or {},
        **classified,
    }


def _scan_oa_locations(query: str, year_from: int | None, year_to: int | None, oa_count: int, *, session: Any = None, max_pages: int | None = None) -> dict[str, Any]:
    counts = {
        "has_oa_url_count": 0,
        "has_pdf_url_count": 0,
        "has_landing_page_count": 0,
        "has_license_count": 0,
        "has_html_xml_or_fulltext_url_count": 0,
        "only_landing_page_count": 0,
    }
    samples: list[dict[str, Any]] = []
    cursor = "*"
    page_count = 0
    scanned = 0
    errors: list[str] = []
    while scanned < oa_count:
        if max_pages is not None and page_count >= max_pages:
            break
        params = _params(query, year_from, year_to, is_oa=True, per_page=200, cursor=cursor)
        try:
            payload = _get_json(params, session=session)
        except Exception as exc:
            errors.append(str(exc))
            break
        page_count += 1
        results = list(payload.get("results") or [])
        if not results:
            break
        for work in results:
            classified = _classify_oa_work(work)
            for key in counts:
                source_key = key.removesuffix("_count")
                if classified.get(source_key):
                    counts[key] += 1
            if len(samples) < 10:
                samples.append(_sample_work(work))
        scanned += len(results)
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)
        time.sleep(0.12)
    return {
        **counts,
        "oa_records_scanned": scanned,
        "oa_scan_page_count": page_count,
        "sample_oa_records": samples,
        "scan_errors": errors,
        "scan_complete": scanned >= oa_count and not errors,
    }


def _write_report(output_dir: Path, all_years: dict[str, Any] | None, recent: dict[str, Any] | None, current: dict[str, Any]) -> Path:
    all_years = all_years or {}
    recent = recent or {}
    lines = [
        "# OpenAlex OA Calibration Report",
        "",
        "## OpenAlex Web Reference",
        f"- Works: approximately {WEB_REFERENCE['works_approx']:,}",
        f"- OA ratio: approximately {WEB_REFERENCE['oa_ratio_approx'] * 100:.1f}%",
        f"- OA works: approximately {WEB_REFERENCE['oa_works_approx']:,}",
        "",
        "## API Calibration",
        f"- API all-years total_count: {all_years.get('total_count', 'not_run')}",
        f"- API all-years oa_count: {all_years.get('oa_count', 'not_run')}",
        f"- API all-years oa_ratio: {all_years.get('oa_ratio', 'not_run')}",
        f"- API all-years has_oa_url_count: {all_years.get('has_oa_url_count', 'not_run')}",
        f"- API all-years has_pdf_url_count: {all_years.get('has_pdf_url_count', 'not_run')}",
        f"- API all-years has_license_count: {all_years.get('has_license_count', 'not_run')}",
        f"- API all-years status: {all_years.get('status', 'not_run')}",
        f"- API all-years failure_reason: {all_years.get('failure_reason', '')}",
        "",
        f"- API 2025-2026 total_count: {recent.get('total_count', 'not_run')}",
        f"- API 2025-2026 oa_count: {recent.get('oa_count', 'not_run')}",
        f"- API 2025-2026 oa_ratio: {recent.get('oa_ratio', 'not_run')}",
        f"- API 2025-2026 has_oa_url_count: {recent.get('has_oa_url_count', 'not_run')}",
        f"- API 2025-2026 has_pdf_url_count: {recent.get('has_pdf_url_count', 'not_run')}",
        f"- API 2025-2026 has_license_count: {recent.get('has_license_count', 'not_run')}",
        f"- API 2025-2026 status: {recent.get('status', 'not_run')}",
        f"- API 2025-2026 failure_reason: {recent.get('failure_reason', '')}",
        "",
        "## Current Agent Corpus Comparison",
        f"- corpus_manifest_records: {current.get('corpus_manifest_records', 848)}",
        f"- approved_for_download: {current.get('approved_for_download', 158)}",
        f"- downloaded_pdf_records: {current.get('downloaded_pdf_records', 3)}",
        "",
        "## Difference Explanation",
        "- The OpenAlex web count may be all-years, while the current Agent corpus is focused on 2025-2026.",
        "- An OpenAlex OA work is not the same thing as a directly downloadable legal PDF.",
        "- Some OA works expose HTML, XML, or landing pages rather than PDF URLs; this stage intentionally does not collect full text.",
        "- The Agent batch collection still had partial and failed batches during the latest corpus run.",
        "- Pre-download QA is intentionally stricter than OpenAlex OA metadata because every download needs explicit legal OA evidence.",
        "",
        "This calibration did not download PDFs or save full-text HTML/XML content.",
    ]
    path = output_dir / "openalex_oa_calibration_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _current_agent_counts() -> dict[str, int]:
    run_dir = Path("agent_runs/crispr_broad_full_collection_2025_2026_medium")

    def count_jsonl(name: str) -> int:
        path = run_dir / name
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())

    return {
        "corpus_manifest_records": count_jsonl("corpus_manifest.jsonl") or 848,
        "approved_for_download": count_jsonl("approved_for_download.jsonl") or 158,
        "downloaded_pdf_records": count_jsonl("downloaded_pdfs_manifest.jsonl") or 3,
    }


def calibrate_openalex_oa_counts(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    output_dir: str = "agent_runs/openalex_oa_calibration",
    *,
    session: Any = None,
    max_scan_pages: int | None = None,
) -> dict[str, Any]:
    """Calibrate OpenAlex OA counts without downloading or saving full text."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_params = _params(query, year_from, year_to, per_page=1)
    oa_params = _params(query, year_from, year_to, is_oa=True, per_page=1)
    label = "all_years" if year_from is None and year_to is None else f"{year_from or 'any'}_{year_to or 'any'}"
    try:
        total_count, total_params = _count(query, year_from, year_to, session=session)
        oa_count, oa_params = _count(query, year_from, year_to, is_oa=True, session=session)
    except Exception as exc:
        error = _error_text(exc)
        payload = {
            "status": "error",
            "query_used": query,
            "filters_used": {"year_from": year_from, "year_to": year_to, "open_access.is_oa": None},
            "source_api_url_or_params": {
                "total_count": total_params,
                "oa_count": oa_params,
                "total_count_url": _api_url(total_params),
                "oa_count_url": _api_url(oa_params),
            },
            "total_count": 0,
            "oa_count": 0,
            "oa_ratio": 0.0,
            "has_oa_url_count": 0,
            "has_pdf_url_count": 0,
            "has_landing_page_count": 0,
            "has_license_count": 0,
            "has_html_xml_or_fulltext_url_count": 0,
            "only_landing_page_count": 0,
            "sample_oa_records": [],
            "scan_errors": [error],
            "failure_reason": error,
            "actual_downloads": 0,
            "artifacts": {},
        }
        json_path = out / f"openalex_oa_calibration_{label}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        all_path = out / "openalex_oa_calibration_all_years.json"
        recent_path = out / "openalex_oa_calibration_2025_2026.json"
        all_payload = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else None
        recent_payload = json.loads(recent_path.read_text(encoding="utf-8")) if recent_path.exists() else None
        if label == "all_years":
            all_payload = payload
        if label == "2025_2026":
            recent_payload = payload
        report_path = _write_report(out, all_payload, recent_payload, _current_agent_counts())
        payload["artifacts"] = {
            "calibration_json": str(json_path.resolve()),
            "calibration_report": str(report_path.resolve()),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    scan = _scan_oa_locations(query, year_from, year_to, oa_count, session=session, max_pages=max_scan_pages)
    ratio = (oa_count / total_count) if total_count else 0.0
    payload = {
        "status": "ok" if not scan.get("scan_errors") else "partial",
        "query_used": query,
        "filters_used": {"year_from": year_from, "year_to": year_to, "open_access.is_oa": None},
        "source_api_url_or_params": {
            "total_count": total_params,
            "oa_count": oa_params,
            "total_count_url": _api_url(total_params),
            "oa_count_url": _api_url(oa_params),
        },
        "total_count": total_count,
        "oa_count": oa_count,
        "oa_ratio": ratio,
        **scan,
        "artifacts": {},
        "actual_downloads": 0,
    }
    json_path = out / f"openalex_oa_calibration_{label}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    all_path = out / "openalex_oa_calibration_all_years.json"
    recent_path = out / "openalex_oa_calibration_2025_2026.json"
    all_payload = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else None
    recent_payload = json.loads(recent_path.read_text(encoding="utf-8")) if recent_path.exists() else None
    if label == "all_years":
        all_payload = payload
    if label == "2025_2026":
        recent_payload = payload
    report_path = _write_report(out, all_payload, recent_payload, _current_agent_counts())
    payload["artifacts"] = {
        "calibration_json": str(json_path.resolve()),
        "calibration_report": str(report_path.resolve()),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
