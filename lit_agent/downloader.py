"""Download planning and guarded downloader interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .hashing import compute_sha256
from .legality import assess_legal_oa_candidate, is_forbidden_text, is_forbidden_url
from .manifest import build_manifest_record, read_jsonl, write_candidate, write_download_plan, write_failure, write_manifest, write_search_log
from .pdf_discovery import discover_direct_pdf_url, is_direct_pdf_url
from .sources.common import safe_headers


ALLOWED_DOWNLOAD_SOURCES = {
    "pmc",
    "europe_pmc",
    "arxiv",
    "biorxiv",
    "medrxiv",
    "unpaywall",
    "unpaywall_confirmed_oa",
    "publisher_explicit_oa",
    "institutional_repository_public_pdf",
    "author_homepage_public_pdf",
    "doaj",
}
BLOCKED_DOWNLOAD_SOURCES = {"openalex", "crossref", "pubmed", "openalex_only", "crossref_only", "pubmed_only", "unknown_pdf", "paywalled_publisher", "file_sharing_unknown", "forum_resource", "shadow_library"}
URL_RISK_TERMS = ("cookie", "session", "token", "login", "vpn", "paywall", "captcha", "bypass")
MAX_PDF_BYTES = 50 * 1024 * 1024


def _requests_or_none() -> Any:
    try:
        import requests  # type: ignore
    except Exception:
        return None
    return requests


@dataclass(slots=True)
class DownloadRequest:
    input_id: str = ""
    query_id: str = ""
    title: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    openalex_id: str = ""
    publication_year: int | None = None
    publication_date: str = ""
    publication_type: str = ""
    journal: str = ""
    authors: list[str] = field(default_factory=list)
    keywords_matched: list[str] = field(default_factory=list)
    source: str = ""
    candidate_url: str = ""
    pdf_url: str = ""
    landing_url: str = ""
    license: str = ""
    oa_status: str = ""
    host_type: str = ""
    version: str = ""
    legality_decision: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    output_dir: str = "downloads"
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DownloadResult:
    status: str
    reason: str
    source: str = ""
    pdf_url: str = ""
    landing_url: str = ""
    local_path: str = ""
    sha256: str = ""
    bytes_written: int = 0
    downloaded_at: str = ""
    manifest_record: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    network_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _url_for_request(record: dict[str, Any]) -> str:
    pdf = str(record.get("pdf_url_candidate") or record.get("pdf_url") or "")
    candidate = str(record.get("candidate_url") or "")
    if pdf:
        return pdf
    if candidate.lower().endswith(".pdf") or "/pdf" in candidate.lower():
        return candidate
    return ""


def request_from_legality_decision(record: dict[str, Any]) -> DownloadRequest:
    discovery = discover_direct_pdf_url(record, dry_run=True, allow_network=False)
    pdf_url = discovery.pdf_url or _url_for_request(record)
    return DownloadRequest(
        input_id=str(record.get("input_id") or ""),
        query_id=str(record.get("query_id") or ""),
        title=str(record.get("title") or ""),
        doi=str(record.get("doi") or ""),
        pmid=str(record.get("pmid") or ""),
        pmcid=str(record.get("pmcid") or ""),
        openalex_id=str(record.get("openalex_id") or ""),
        publication_year=record.get("publication_year"),
        publication_date=str(record.get("publication_date") or ""),
        publication_type=str(record.get("publication_type") or ""),
        journal=str(record.get("journal") or ""),
        authors=list(record.get("authors") or []),
        keywords_matched=list(record.get("keywords_matched") or record.get("matched_keywords") or []),
        source=str(record.get("source") or ""),
        candidate_url=str(record.get("candidate_url") or ""),
        pdf_url=pdf_url,
        landing_url=str(record.get("landing_url") or ""),
        license=str(record.get("license") or ""),
        oa_status=str(record.get("oa_status") or ""),
        host_type=str(record.get("host_type") or ""),
        version=str(record.get("version") or ""),
        legality_decision=str(record.get("decision") or ""),
        evidence_sources=list(record.get("evidence_sources") or discovery.evidence_sources or []),
        risk_flags=list(record.get("risk_flags") or discovery.risk_flags or []),
    )


def plan_downloads_from_legality_audit(path: str | Path = "legality_audit.jsonl", max_downloads: int | None = None, record_filter: Callable[[dict[str, Any]], bool] | None = None) -> list[DownloadRequest]:
    indexed_requests: list[tuple[int, DownloadRequest]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(read_jsonl(path)):
        if record_filter is not None and not record_filter(record):
            continue
        if record.get("decision") != "allowed_for_future_download":
            continue
        request = request_from_legality_decision(record)
        key = (request.doi.lower(), request.pdf_url or request.landing_url)
        if key in seen:
            continue
        seen.add(key)
        indexed_requests.append((index, request))
    def plan_priority(item: tuple[int, DownloadRequest]) -> tuple[int, int, int]:
        index, request = item
        url = request.pdf_url.lower()
        stable_repository = "europepmc.org" in url or "ncbi.nlm.nih.gov/pmc" in url
        return (0 if stable_repository else 1, 0 if request.pdf_url else 1, -index)

    indexed_requests.sort(key=plan_priority)
    requests = [request for _, request in indexed_requests]
    if max_downloads:
        return requests[:max_downloads]
    return requests


def _is_doi_landing(url: str) -> bool:
    return "doi.org/" in urlparse(url).netloc + urlparse(url).path


def is_allowed_download_source(source: str, evidence_sources: list[str]) -> bool:
    values = {source, *evidence_sources}
    if values & BLOCKED_DOWNLOAD_SOURCES:
        return False
    return bool(values & ALLOWED_DOWNLOAD_SOURCES)


def _url_has_access_risk(url: str) -> bool:
    lowered = (url or "").lower()
    lowered = lowered.replace("cookies_not_supported", "").replace("cookie_not_supported", "")
    return any(term in lowered for term in URL_RISK_TERMS)


def _headers_lookup(headers: Any, name: str) -> str:
    if not headers:
        return ""
    if hasattr(headers, "get"):
        return str(headers.get(name) or headers.get(name.lower()) or "")
    return ""


def _response_status(response: Any) -> int:
    return int(getattr(response, "status_code", getattr(response, "status", 200)) or 200)


def _response_url(response: Any, fallback: str) -> str:
    return str(getattr(response, "url", "") or fallback)


def _pdf_confirmed_by_headers_or_url(url: str, headers: Any) -> bool:
    content_type = _headers_lookup(headers, "Content-Type").lower()
    disposition = _headers_lookup(headers, "Content-Disposition").lower()
    return "application/pdf" in content_type or ".pdf" in disposition or is_direct_pdf_url(url)


def _pdf_confirmed_by_download_response(headers: Any) -> bool:
    content_type = _headers_lookup(headers, "Content-Type").lower()
    disposition = _headers_lookup(headers, "Content-Disposition").lower()
    return "application/pdf" in content_type or ".pdf" in disposition


def _looks_like_html(first_bytes: bytes, headers: Any) -> bool:
    content_type = _headers_lookup(headers, "Content-Type").lower()
    prefix = first_bytes.lstrip().lower()
    return "text/html" in content_type or prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _should_probe_pdf_with_get(url: str) -> bool:
    parsed = urlparse(url)
    lowered = f"{parsed.path}?{parsed.query}".lower()
    return "europepmc.org" in parsed.netloc.lower() and ("getpdf" in lowered or "pdf=render" in lowered)


def _network_pdf_get_probe(url: str, headers: dict[str, str], timeout: int) -> tuple[bool, str, str, Any, int]:
    requests = _requests_or_none()
    if requests is not None:
        response = requests.get(url, headers={**headers, "Range": "bytes=0-7"}, timeout=(5, timeout), stream=True)
        status = _response_status(response)
        final_url = _response_url(response, url)
        response_headers = getattr(response, "headers", {})
        if status in {401, 403}:
            return False, "access_requires_auth_or_paywall", final_url, response_headers, 1
        if status >= 400:
            return False, f"http_status_{status}", final_url, response_headers, 1
        first = next(response.iter_content(chunk_size=8), b"")
        response.close()
        if not final_url.startswith("https://"):
            return False, "final_url_must_be_https", final_url, response_headers, 1
        if _url_has_access_risk(final_url) or is_forbidden_url(final_url):
            return False, "redirect_url_forbidden_or_risky", final_url, response_headers, 1
        if not (first.startswith(b"%PDF") or _pdf_confirmed_by_headers_or_url(final_url, response_headers)):
            return False, "preflight_not_pdf", final_url, response_headers, 1
        return True, "preflight_pdf_confirmed", final_url, response_headers, 1

    request = Request(url, headers={**headers, "Range": "bytes=0-7"})
    with urlopen(request, timeout=timeout) as response:
        final_url = str(getattr(response, "url", "") or url)
        status = int(getattr(response, "status", 200) or 200)
        response_headers = getattr(response, "headers", {})
        if status in {401, 403}:
            return False, "access_requires_auth_or_paywall", final_url, response_headers, 1
        if status >= 400:
            return False, f"http_status_{status}", final_url, response_headers, 1
        first = response.read(8)
        if not final_url.startswith("https://"):
            return False, "final_url_must_be_https", final_url, response_headers, 1
        if _url_has_access_risk(final_url) or is_forbidden_url(final_url):
            return False, "redirect_url_forbidden_or_risky", final_url, response_headers, 1
        if not (first.startswith(b"%PDF") or _pdf_confirmed_by_headers_or_url(final_url, response_headers)):
            return False, "preflight_not_pdf", final_url, response_headers, 1
        return True, "preflight_pdf_confirmed", final_url, response_headers, 1


def _network_pdf_preflight(url: str, *, session: Any = None, timeout: int = 15) -> tuple[bool, str, str, Any, int]:
    headers = safe_headers({"Accept": "application/pdf"})
    try:
        if session is not None:
            if hasattr(session, "head"):
                response = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
            else:
                response = session.get(url, headers={**headers, "Range": "bytes=0-4"}, timeout=timeout, stream=True)
            status = _response_status(response)
            final_url = _response_url(response, url)
            if status in {401, 403}:
                return False, "access_requires_auth_or_paywall", final_url, getattr(response, "headers", {}), 1
            if status >= 400:
                return False, f"http_status_{status}", final_url, getattr(response, "headers", {}), 1
            if not final_url.startswith("https://"):
                return False, "final_url_must_be_https", final_url, getattr(response, "headers", {}), 1
            if _url_has_access_risk(final_url) or is_forbidden_url(final_url):
                return False, "redirect_url_forbidden_or_risky", final_url, getattr(response, "headers", {}), 1
            if not _pdf_confirmed_by_headers_or_url(final_url, getattr(response, "headers", {})):
                return False, "preflight_not_pdf", final_url, getattr(response, "headers", {}), 1
            return True, "preflight_pdf_confirmed", final_url, getattr(response, "headers", {}), 1

        if _should_probe_pdf_with_get(url):
            return _network_pdf_get_probe(url, headers, timeout)

        request = Request(url, headers=headers, method="HEAD")
        with urlopen(request, timeout=timeout) as response:
            final_url = str(getattr(response, "url", "") or url)
            status = int(getattr(response, "status", 200) or 200)
            response_headers = getattr(response, "headers", {})
            if status in {401, 403}:
                return False, "access_requires_auth_or_paywall", final_url, response_headers, 1
            if status >= 400:
                return False, f"http_status_{status}", final_url, response_headers, 1
            if not final_url.startswith("https://"):
                return False, "final_url_must_be_https", final_url, response_headers, 1
            if _url_has_access_risk(final_url) or is_forbidden_url(final_url):
                return False, "redirect_url_forbidden_or_risky", final_url, response_headers, 1
            if not _pdf_confirmed_by_headers_or_url(final_url, response_headers):
                return False, "preflight_not_pdf", final_url, response_headers, 1
            return True, "preflight_pdf_confirmed", final_url, response_headers, 1
    except Exception as exc:
        return False, f"preflight_failed:{exc}", url, {}, 1


def final_preflight_check(request: DownloadRequest, *, allow_network: bool = False, session: Any = None) -> DownloadResult:
    record = request.to_dict()
    record.update({"decision": request.legality_decision, "pdf_url_candidate": request.pdf_url})
    decision = assess_legal_oa_candidate(record)
    if request.legality_decision != "allowed_for_future_download":
        return DownloadResult(status="blocked", reason="legality_audit_decision_not_allowed", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    if decision.is_forbidden or is_forbidden_url(request.pdf_url) or is_forbidden_url(request.landing_url):
        return DownloadResult(status="blocked", reason="forbidden_url_detected", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, risk_flags=["forbidden"])
    if is_forbidden_text(" ".join(request.risk_flags)):
        return DownloadResult(status="blocked", reason="login_or_bypass_risk_detected", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, risk_flags=["bypass"])
    if _url_has_access_risk(request.pdf_url) or _url_has_access_risk(request.landing_url):
        return DownloadResult(status="blocked", reason="url_contains_login_or_bypass_risk", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, risk_flags=["bypass"])
    if not request.pdf_url:
        return DownloadResult(status="blocked", reason="missing_pdf_url", source=request.source, landing_url=request.landing_url, risk_flags=["missing_url"])
    parsed = urlparse(request.pdf_url)
    if parsed.scheme != "https":
        return DownloadResult(status="blocked", reason="pdf_url_must_be_https", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    if _is_doi_landing(request.pdf_url):
        return DownloadResult(status="blocked", reason="doi_landing_page_is_not_pdf_url", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    if not is_allowed_download_source(request.source, request.evidence_sources):
        return DownloadResult(status="blocked", reason="source_not_allowed_for_download", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    if not request.license and not request.evidence_sources:
        return DownloadResult(status="blocked", reason="missing_license_or_oa_evidence", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    if allow_network:
        ok, reason, final_url, _headers, network_calls = _network_pdf_preflight(request.pdf_url, session=session)
        if not ok:
            return DownloadResult(status="blocked", reason=reason, source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, network_calls=network_calls)
        request.pdf_url = final_url
        return DownloadResult(status="preflight_passed", reason=reason, source=request.source, pdf_url=final_url, landing_url=request.landing_url, network_calls=network_calls)
    if not is_direct_pdf_url(request.pdf_url):
        return DownloadResult(status="blocked", reason="direct_pdf_url_not_confirmed_without_network", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
    return DownloadResult(status="preflight_passed", reason="eligible_for_future_download_but_not_downloaded", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)


def _safe_filename(request: DownloadRequest) -> str:
    base = request.doi or request.title or request.pmid or "download"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")[:80] or "download"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def _unique_output_path(output_dir: str | Path, filename: str) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".pdf"
    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate unique download filename")


def _iter_response_bytes(response: Any):
    if hasattr(response, "iter_content"):
        yield from response.iter_content(chunk_size=65536)
        return
    if hasattr(response, "read"):
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            yield chunk


def _download_bytes_to_temp(url: str, tmp_path: Path, *, session: Any = None, timeout: int = 30) -> tuple[int, bytes, Any]:
    headers = safe_headers({"Accept": "application/pdf"})
    if session is not None:
        response = session.get(url, headers=headers, timeout=timeout, stream=True)
        status = _response_status(response)
        if status in {401, 403}:
            raise RuntimeError("access_requires_auth_or_paywall")
        if status >= 400:
            raise RuntimeError(f"http_status_{status}")
        response_headers = getattr(response, "headers", {})
        total = 0
        first = b""
        with tmp_path.open("wb") as handle:
            for chunk in _iter_response_bytes(response):
                if not chunk:
                    continue
                if not first:
                    first = bytes(chunk[:8])
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise RuntimeError("pdf_exceeds_max_size")
                handle.write(chunk)
        return total, first, response_headers

    requests = _requests_or_none()
    if requests is not None:
        response = requests.get(url, headers=headers, timeout=(5, timeout), stream=True)
        status = _response_status(response)
        if status in {401, 403}:
            raise RuntimeError("access_requires_auth_or_paywall")
        if status >= 400:
            raise RuntimeError(f"http_status_{status}")
        response_headers = getattr(response, "headers", {})
        total = 0
        first = b""
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if not first:
                    first = bytes(chunk[:8])
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise RuntimeError("pdf_exceeds_max_size")
                handle.write(chunk)
        response.close()
        return total, first, response_headers

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", 200) or 200)
        if status in {401, 403}:
            raise RuntimeError("access_requires_auth_or_paywall")
        if status >= 400:
            raise RuntimeError(f"http_status_{status}")
        response_headers = getattr(response, "headers", {})
        total = 0
        first = b""
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                if not first:
                    first = chunk[:8]
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise RuntimeError("pdf_exceeds_max_size")
                handle.write(chunk)
    return total, first, response_headers


def download_pdf(request: DownloadRequest, *, dry_run: bool = True, allow_download: bool = False, session: Any = None) -> DownloadResult:
    preflight = final_preflight_check(request, allow_network=bool(allow_download and not dry_run), session=session)
    if preflight.status != "preflight_passed":
        preflight.manifest_record = build_manifest_record(request.to_dict(), preflight.to_dict())
        return preflight
    if dry_run or not allow_download:
        result = DownloadResult(status="skipped_dry_run", reason="dry_run_or_allow_download_false", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url)
        result.manifest_record = build_manifest_record(request.to_dict(), result.to_dict())
        return result
    downloaded_at = datetime.now(timezone.utc).isoformat()
    output_path = _unique_output_path(request.output_dir, _safe_filename(request))
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        bytes_written, first_bytes, headers = _download_bytes_to_temp(request.pdf_url, tmp_path, session=session)
        if bytes_written <= 0:
            raise RuntimeError("downloaded_file_empty")
        if not first_bytes.startswith(b"%PDF") and _looks_like_html(first_bytes, headers):
            raise RuntimeError("download_response_html_not_pdf")
        if not first_bytes.startswith(b"%PDF") and not _pdf_confirmed_by_download_response(headers):
            raise RuntimeError("downloaded_content_not_pdf")
        tmp_path.replace(output_path)
        sha256 = compute_sha256(output_path)
        result = DownloadResult(status="downloaded", reason="downloaded", source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, local_path=str(output_path), sha256=sha256, bytes_written=bytes_written, downloaded_at=downloaded_at, network_calls=preflight.network_calls + 1)
        result.manifest_record = build_manifest_record(request.to_dict(), result.to_dict())
        return result
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        write_failure("pdf download failed", {"doi": request.doi, "pdf_url": request.pdf_url, "error": str(exc)})
        result = DownloadResult(status="failed", reason=str(exc), source=request.source, pdf_url=request.pdf_url, landing_url=request.landing_url, network_calls=preflight.network_calls + 1)
        return result


def _candidate_from_blocked_request(request: DownloadRequest, result: DownloadResult) -> dict[str, Any]:
    return {
        "query_id": request.query_id,
        "title": request.title,
        "doi": request.doi,
        "pmid": request.pmid,
        "pmcid": request.pmcid,
        "journal": request.journal,
        "publication_year": request.publication_year,
        "publication_date": request.publication_date,
        "source": request.source,
        "candidate_url": request.pdf_url or request.candidate_url,
        "pdf_url_candidate": request.pdf_url,
        "landing_url": request.landing_url,
        "license": request.license,
        "oa_status": request.oa_status,
        "reason": "missing_direct_legal_pdf_url" if result.reason == "missing_pdf_url" else result.reason,
        "suggested_manual_check": "Confirm a direct legal PDF URL and OA evidence before any future download.",
        "is_download_allowed_now": False,
    }


def execute_download_plan(requests: list[DownloadRequest], *, dry_run: bool = True, allow_download: bool = False, session: Any = None) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for request in requests:
        result = download_pdf(request, dry_run=dry_run, allow_download=allow_download, session=session)
        plan_record = {
            "query_id": request.query_id,
            "title": request.title,
            "doi": request.doi,
            "source": request.source,
            "journal": request.journal,
            "publication_year": request.publication_year,
            "publication_date": request.publication_date,
            "pdf_url": request.pdf_url,
            "landing_url": request.landing_url,
            "license": request.license,
            "oa_status": request.oa_status,
            "evidence_sources": request.evidence_sources,
            "legality_decision": request.legality_decision,
            "final_preflight_status": result.status,
            "planned_status": "planned_dry_run" if result.status == "skipped_dry_run" else result.status,
            "reason": result.reason,
            "local_path": result.local_path,
            "sha256": result.sha256,
            "bytes_written": result.bytes_written,
            "network_calls": result.network_calls,
        }
        write_download_plan(plan_record)
        if result.status in {"blocked", "failed"}:
            write_candidate(_candidate_from_blocked_request(request, result))
            write_failure("download preflight or retrieval failed", {"doi": request.doi, "source": request.source, "pdf_url": request.pdf_url, "reason": result.reason, "status": result.status})
        if result.manifest_record:
            manifest_record = dict(result.manifest_record)
            if manifest_record.get("status") == "skipped_dry_run":
                manifest_record["status"] = "planned_dry_run"
            write_manifest(manifest_record)
        results.append(result)
    write_search_log("download_plan", {"planned": sum(r.status == "skipped_dry_run" for r in results), "blocked": sum(r.status == "blocked" for r in results), "failed": sum(r.status == "failed" for r in results), "downloads": sum(r.status == "downloaded" for r in results), "network_calls": sum(r.network_calls for r in results)})
    return results
