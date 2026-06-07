"""DOI landing-page discovery for PDF/full-text candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from collections import Counter
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .legality import is_forbidden_text, is_forbidden_url
from .failure_analysis import classify_failure
from .metadata import normalize_doi
from .pdf_discovery import is_direct_pdf_url
from .sources.common import safe_headers


LANDING_RISK_TERMS = ("login", "paywall", "subscription", "captcha", "cookie required", "session token", "institutional login")
HIGH_PRIORITY_OA_SOURCES = {"unpaywall", "pmc", "europe_pmc", "publisher_explicit_oa"}
OPEN_LICENSE_TERMS = {"cc-by", "cc-by-nc", "cc0", "public-domain", "cc-by-sa", "cc-by-nc-nd", "cc-by-nc-sa"}
MOCK_TEXT_TERMS = ("mock", "placeholder", "test doi")
MOCK_DOI_PREFIXES = ("10.1000/",)
MOCK_DOMAINS = {"example.org", "example.com", "example.net"}
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


@dataclass(slots=True)
class LandingDiscoveryResult:
    doi: str = ""
    landing_url: str = ""
    source: str = "doi_landing_page"
    status: str = "planned"
    reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    blocked_count: int = 0
    requires_confirmation_count: int = 0
    with_existing_oa_evidence_count: int = 0
    network_calls: int = 0
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _LandingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag.lower(), {key.lower(): value or "" for key, value in attrs}))


def normalize_doi_url(doi: str) -> str:
    normalized = normalize_doi(doi)
    if not normalized:
        return ""
    return f"https://doi.org/{quote(normalized, safe='/')}"


def _record_sources(record: dict[str, Any]) -> set[str]:
    values = record.get("sources") or record.get("evidence_sources") or record.get("oa_evidence") or record.get("source") or []
    if not isinstance(values, list):
        values = [values]
    source = record.get("source")
    if source:
        values.append(source)
    return {str(value).lower() for value in values if value}


def _is_open_license(value: Any) -> bool:
    lowered = str(value or "").lower()
    return any(term in lowered for term in OPEN_LICENSE_TERMS)


def _has_pmc_evidence(record: dict[str, Any]) -> bool:
    if record.get("pmcid"):
        return True
    open_access = record.get("openAccess")
    if open_access is True or str(open_access).lower() == "true":
        return True
    return str(record.get("source") or "").lower() in {"pmc", "europe_pmc"} and str(record.get("oa_status") or "").lower() in {"open", "gold", "green", "hybrid"}


def is_mock_landing_record(record: dict[str, Any]) -> bool:
    doi = normalize_doi(str(record.get("doi") or ""))
    if not doi:
        return True
    if doi.startswith(MOCK_DOI_PREFIXES):
        return True
    text = " ".join(str(record.get(field) or "") for field in ("title", "reason", "evidence_summary", "suggested_manual_check", "query_id", "input_id")).lower()
    if any(term in text for term in MOCK_TEXT_TERMS):
        return True
    for field in ("landing_url", "url", "candidate_url", "pdf_url_candidate", "url_for_pdf", "best_oa_location"):
        url = str(record.get(field) or "")
        if urlparse(url).netloc.lower() in MOCK_DOMAINS:
            return True
    return False


def landing_record_priority(record: dict[str, Any]) -> tuple[int, str]:
    source = str(record.get("source") or "").lower()
    sources = _record_sources(record)
    reason = str(record.get("reason") or "").lower()
    oa_status = str(record.get("oa_status") or record.get("unpaywall_oa_status") or "").lower()
    if record.get("decision") == "allowed_for_future_download" and (source in HIGH_PRIORITY_OA_SOURCES or sources & HIGH_PRIORITY_OA_SOURCES):
        return 0, "allowed_oa_audit"
    if source == "unpaywall" and "confirmed_oa" in reason:
        return 1, "unpaywall_confirmed_candidate"
    if record.get("unpaywall_is_oa") is True or oa_status in {"gold", "hybrid", "green"}:
        return 2, "metadata_oa_status"
    if _has_pmc_evidence(record):
        return 3, "pmc_europe_pmc_open_access"
    if record.get("landing_url") and _is_open_license(record.get("license")) and (source == "publisher_explicit_oa" or "publisher" in sources):
        return 4, "publisher_open_license"
    return 5, "other_needs_confirmation"


def _failure_doi(record: dict[str, Any]) -> str:
    context = record.get("context") if isinstance(record.get("context"), dict) else {}
    return normalize_doi(str(record.get("doi") or context.get("doi") or ""))


def _known_failure_map(failure_records: list[dict[str, Any]] | None) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for failure in failure_records or []:
        doi = _failure_doi(failure)
        if not doi:
            continue
        known.setdefault(doi, set()).add(classify_failure(failure))
    return known


def _known_failure_exclusion(categories: set[str]) -> str:
    if categories & {"preflight_403", "preflight_404", "unpaywall_404"}:
        return "already_failed_403_404"
    if "landing_login_paywall_block" in categories:
        return "already_blocked_login_paywall"
    if "landing_direct_pdf_guard" in categories:
        return "direct_pdf_guard_blocked"
    if "missing_pdf_url" in categories:
        return "repeated_missing_pdf_url"
    return ""


def select_landing_page_records(records: list[dict[str, Any]], max_results: int | None = None, failure_records: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_mock_count = 0
    excluded_known_failed_count = 0
    known_failure_breakdown: Counter[str] = Counter()
    known_failures = _known_failure_map(failure_records)
    ranked: list[tuple[int, int, dict[str, Any], str]] = []
    for index, record in enumerate(records):
        if is_mock_landing_record(record):
            excluded_mock_count += 1
            known_failure_breakdown["historical_mock_or_placeholder"] += 1
            continue
        doi = normalize_doi(str(record.get("doi") or ""))
        known_reason = _known_failure_exclusion(known_failures.get(doi, set()))
        if known_reason:
            excluded_known_failed_count += 1
            known_failure_breakdown[known_reason] += 1
            continue
        priority, label = landing_record_priority(record)
        ranked.append((priority, -index, record, label))

    ranked.sort(key=lambda item: (item[0], item[1]))
    deduped_ranked: list[tuple[int, int, dict[str, Any], str]] = []
    seen_dois: set[str] = set()
    duplicate_count = 0
    for item in ranked:
        doi = normalize_doi(str(item[2].get("doi") or ""))
        if doi in seen_dois:
            duplicate_count += 1
            continue
        seen_dois.add(doi)
        deduped_ranked.append(item)
    selected_ranked = deduped_ranked[: max_results or len(deduped_ranked)]
    selected = [record for _, _, record, _ in selected_ranked]
    breakdown = Counter(label for _, _, _, label in selected_ranked)
    stats = {
        "selected_count": len(selected),
        "excluded_mock_count": excluded_mock_count,
        "excluded_known_failed_count": excluded_known_failed_count,
        "duplicate_doi_count": duplicate_count,
        "known_failure_breakdown": dict(sorted(known_failure_breakdown.items())),
        "priority_breakdown": dict(sorted(breakdown.items())),
    }
    return selected, stats


def _record_landing_url(record: dict[str, Any]) -> str:
    landing = str(record.get("landing_url") or record.get("url") or "")
    if landing:
        return landing
    return normalize_doi_url(str(record.get("doi") or ""))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def extract_pdf_candidates_from_html(html: str, base_url: str) -> list[str]:
    parser = _LandingHTMLParser()
    parser.feed(html or "")
    candidates: list[str] = []
    for tag, attrs in parser.tags:
        if tag == "meta":
            name = (attrs.get("name") or attrs.get("property") or "").lower()
            content = attrs.get("content") or ""
            if name == "citation_pdf_url" and content:
                candidates.append(urljoin(base_url, content))
            if name == "og:url" and content and is_direct_pdf_url(content):
                candidates.append(urljoin(base_url, content))
        if tag == "link":
            rel = attrs.get("rel", "").lower()
            content_type = attrs.get("type", "").lower()
            href = attrs.get("href") or ""
            if href and ("alternate" in rel and "pdf" in content_type):
                candidates.append(urljoin(base_url, href))
            if href and rel == "canonical" and is_direct_pdf_url(href):
                candidates.append(urljoin(base_url, href))
        if tag == "a":
            href = attrs.get("href") or ""
            if href and is_direct_pdf_url(href):
                candidates.append(urljoin(base_url, href))
    return _unique(candidates)


def _has_existing_oa_evidence(record: dict[str, Any]) -> bool:
    if record.get("decision") == "allowed_for_future_download":
        return True
    if record.get("is_legal_oa_candidate") is True:
        return True
    evidence = " ".join(str(value) for value in record.get("evidence_sources") or record.get("oa_evidence") or [])
    source = str(record.get("source") or "")
    return any(term in evidence.lower() for term in ("unpaywall", "pmc", "europe_pmc", "publisher")) or source in {"unpaywall", "europe_pmc", "pmc", "publisher_explicit_oa"}


def classify_landing_candidate(candidate_url: str, context_record: dict[str, Any]) -> dict[str, Any]:
    doi = str(context_record.get("doi") or "")
    title = str(context_record.get("title") or "")
    landing_url = _record_landing_url(context_record)
    base = {
        "query_id": context_record.get("query_id", ""),
        "title": title,
        "doi": doi,
        "source": "doi_landing_page",
        "candidate_url": candidate_url,
        "pdf_url_candidate": candidate_url,
        "landing_url": landing_url,
        "license": context_record.get("license", ""),
        "oa_status": context_record.get("oa_status", ""),
        "is_download_allowed_now": False,
    }
    risk_text = " ".join(str(context_record.get(field) or "") for field in ("reason", "evidence_summary", "suggested_manual_check"))
    if is_forbidden_url(candidate_url) or is_forbidden_text(risk_text):
        return {**base, "decision": "doi_landing_blocked_forbidden", "reason": "doi_landing_blocked_forbidden", "planned_status": "blocked", "risk_flags": ["forbidden"]}
    if not candidate_url.startswith("https://"):
        return {**base, "decision": "doi_landing_candidate_needs_confirmation", "reason": "doi_landing_pdf_requires_https_confirmation", "planned_status": "candidate"}
    if _has_existing_oa_evidence(context_record):
        return {**base, "decision": "doi_landing_candidate_with_existing_oa_evidence", "reason": "doi_landing_pdf_with_existing_oa_evidence_not_downloaded_round_011", "planned_status": "candidate_direct_pdf_discovered_not_downloaded_round_011"}
    return {**base, "decision": "doi_landing_candidate_needs_confirmation", "reason": "doi_landing_pdf_requires_oa_confirmation", "planned_status": "candidate"}


def _header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    if hasattr(headers, "get"):
        return str(headers.get(name) or headers.get(name.lower()) or "")
    return ""


def _redirect_location(response: Any, current_url: str) -> str:
    location = _header(response, "Location")
    return urljoin(str(getattr(response, "url", current_url) or current_url), location) if location else ""


def _block_pdf_landing_url(url: str) -> None:
    if is_direct_pdf_url(url):
        raise RuntimeError("doi_landing_direct_pdf_url_not_fetched")


def _safe_get_html(url: str, *, session: Any = None, timeout: int = 15, retries: int = 3) -> tuple[str, int, str]:
    headers = safe_headers({"Accept": "text/html,application/xhtml+xml"})
    attempts = 0
    last_error: Exception | None = None
    opener = build_opener(_NoRedirectHandler())
    for attempt in range(1, retries + 1):
        current_url = url
        try:
            time.sleep(min(0.2 * attempt, 1.0))
            for _ in range(5):
                _block_pdf_landing_url(current_url)
                attempts += 1
                if session is not None:
                    response = session.get(current_url, headers=headers, timeout=timeout, allow_redirects=False)
                    status = int(getattr(response, "status_code", 200))
                    final_url = str(getattr(response, "url", current_url) or current_url)
                else:
                    request = Request(current_url, headers=headers)
                    try:
                        response = opener.open(request, timeout=timeout)
                        status = int(getattr(response, "status", 200))
                    except HTTPError as exc:
                        response = exc
                        status = int(exc.code)
                    final_url = str(getattr(response, "url", current_url) or current_url)
                if status in REDIRECT_STATUSES:
                    next_url = _redirect_location(response, current_url)
                    if not next_url:
                        raise RuntimeError(f"landing_redirect_without_location_{status}")
                    if is_direct_pdf_url(next_url):
                        raise RuntimeError("doi_landing_redirect_to_pdf_not_fetched")
                    current_url = next_url
                    continue
                content_type = _header(response, "Content-Type").lower()
                content_disposition = _header(response, "Content-Disposition").lower()
                if is_direct_pdf_url(final_url) or "application/pdf" in content_type or content_type.endswith("/pdf") or ".pdf" in content_disposition:
                    raise RuntimeError("doi_landing_pdf_response_not_fetched")
                if status in {401, 403}:
                    raise RuntimeError(f"access_requires_auth_or_paywall:{status}")
                if status >= 400:
                    raise RuntimeError(f"landing_http_status_{status}")
                text = getattr(response, "text", None)
                if text is None:
                    content = getattr(response, "content", None)
                    if content is None:
                        content = response.read()
                    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                return str(text), attempts, final_url
            raise RuntimeError("landing_redirect_limit_exceeded")
        except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def resolve_landing_page(record: dict[str, Any], *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> LandingDiscoveryResult:
    doi = str(record.get("doi") or "")
    landing_url = _record_landing_url(record)
    result = LandingDiscoveryResult(doi=doi, landing_url=landing_url, dry_run=dry_run or not allow_network)
    if is_forbidden_url(landing_url):
        result.status = "blocked"
        result.reason = "doi_landing_blocked_forbidden"
        result.blocked_count = 1
        return result
    if is_direct_pdf_url(landing_url):
        result.status = "blocked"
        result.reason = "doi_landing_direct_pdf_url_not_fetched"
        result.blocked_count = 1
        return result
    if dry_run or not allow_network:
        result.status = "planned"
        result.reason = "dry_run_landing_page_not_fetched"
        return result
    try:
        html, network_calls, final_url = _safe_get_html(landing_url, session=session)
        result.network_calls = network_calls
        result.landing_url = final_url
        lowered = html.lower()
        if any(term in lowered for term in LANDING_RISK_TERMS):
            result.status = "blocked"
            result.reason = "doi_landing_blocked_requires_login_or_paywall"
            result.blocked_count = 1
            return result
        candidates = [classify_landing_candidate(url, {**record, "landing_url": final_url}) for url in extract_pdf_candidates_from_html(html, final_url)]
        result.candidates = candidates
        result.candidate_count = len(candidates)
        result.blocked_count = sum(item.get("planned_status") == "blocked" for item in candidates)
        result.requires_confirmation_count = sum(item.get("decision") == "doi_landing_candidate_needs_confirmation" for item in candidates)
        result.with_existing_oa_evidence_count = sum(item.get("decision") == "doi_landing_candidate_with_existing_oa_evidence" for item in candidates)
        result.status = "completed"
        result.reason = "landing_page_discovery_completed"
        return result
    except Exception as exc:
        result.reason = str(exc)
        if result.reason.startswith("doi_landing_") and "pdf" in result.reason:
            result.status = "blocked"
        else:
            result.status = "failed"
        result.blocked_count = 1
        return result
