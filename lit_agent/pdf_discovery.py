"""Direct legal PDF URL discovery helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from .legality import is_forbidden_text, is_forbidden_url


METADATA_ONLY_SOURCES = {"openalex", "crossref", "pubmed"}
LEGAL_DISCOVERY_SOURCES = {
    "unpaywall",
    "europe_pmc",
    "pmc",
    "arxiv",
    "biorxiv",
    "medrxiv",
    "publisher_explicit_oa",
    "institutional_repository_public_pdf",
    "author_homepage_public_pdf",
    "doaj",
}
PDF_PATH_MARKERS = (".pdf", "/pdf", "download/pdf", "viewfile", "getfile.php", "pdf=render", "getpdf")
NATURE_ARTICLE_HOSTS = {"www.nature.com", "nature.com"}


@dataclass(slots=True)
class PdfDiscoveryResult:
    status: str
    reason: str
    pdf_url: str = ""
    landing_url: str = ""
    source: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return dict(getattr(record, "__dict__", {}))


def _source_values(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("source", "source_type"):
        value = record.get(field)
        if value:
            values.add(str(value).lower())
    for field in ("sources", "evidence_sources", "oa_evidence"):
        value = record.get(field) or []
        if isinstance(value, list):
            values.update(str(item).lower() for item in value if item)
        elif value:
            values.add(str(value).lower())
    return values


def _location_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key in ("url_for_pdf", "pdf_url", "url", "landing_page_url"):
            url = value.get(key)
            if url:
                urls.append(str(url))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_location_urls(item))
    elif value:
        urls.append(str(value))
    return urls


def is_direct_pdf_url(url: str) -> bool:
    lowered = (url or "").lower()
    parsed = urlparse(url)
    path_and_query = f"{parsed.path}?{parsed.query}".lower()
    return any(marker in lowered or marker in path_and_query for marker in PDF_PATH_MARKERS)


def _arxiv_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    if "arxiv.org" not in parsed.netloc:
        return ""
    path = parsed.path.strip("/")
    if path.startswith("pdf/"):
        return f"https://arxiv.org/{path}"
    if path.startswith("abs/"):
        return f"https://arxiv.org/pdf/{path[4:]}"
    return ""


def _pmc_pdf_url(record: dict[str, Any]) -> str:
    pmcid = str(record.get("pmcid") or "").strip()
    if not pmcid:
        return ""
    pmcid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    return f"https://europepmc.org/api/getPdf?pmcid={pmcid}"


def _nature_article_id_from_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    if doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/") :]
    if not doi.startswith("10.1038/"):
        return ""
    article_id = doi.rsplit("/", 1)[-1]
    return article_id if article_id.startswith(("s", "ncomms")) else ""


def _nature_article_id_from_url(url: str) -> str:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower().removeprefix("www.")
    if host not in NATURE_ARTICLE_HOSTS:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] != "articles":
        return ""
    article_id = parts[1]
    return article_id[:-4] if article_id.endswith(".pdf") else article_id


def _nature_reference_pdf_url(record: dict[str, Any], urls: list[str]) -> str:
    article_id = _nature_article_id_from_doi(str(record.get("doi") or ""))
    if not article_id:
        for url in urls:
            article_id = _nature_article_id_from_url(url)
            if article_id:
                break
    if not article_id:
        return ""
    return f"https://www.nature.com/articles/{article_id}_reference.pdf"


def _candidate_urls(record: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for field in ("url_for_pdf", "pdf_url", "pdf_url_candidate", "candidate_url", "best_oa_location", "unpaywall_best_oa_location", "landing_url", "url"):
        urls.extend(_location_urls(record.get(field)))
    for location in record.get("oa_locations") or []:
        urls.extend(_location_urls(location))
    unique: list[str] = []
    for url in urls:
        if url and url not in unique:
            unique.append(url)
    pmc_pdf = _pmc_pdf_url(record)
    if pmc_pdf and pmc_pdf not in unique:
        unique.insert(0, pmc_pdf)
    nature_pdf = _nature_reference_pdf_url(record, unique)
    if nature_pdf and nature_pdf not in unique:
        insert_at = 1 if pmc_pdf and unique and unique[0] == pmc_pdf else 0
        unique.insert(insert_at, nature_pdf)
    return unique


def _discovery_allowed(record: dict[str, Any], sources: set[str]) -> bool:
    if sources & LEGAL_DISCOVERY_SOURCES:
        return True
    if record.get("is_legal_oa_candidate") is True and "unpaywall" in sources:
        return True
    return False


def discover_direct_pdf_url(record: Any, *, dry_run: bool = True, allow_network: bool = False, session: Any = None) -> PdfDiscoveryResult:
    """Find a direct legal PDF URL candidate without downloading content."""

    data = _as_dict(record)
    sources = _source_values(data)
    source = str(data.get("source") or next(iter(sources), ""))
    landing_url = str(data.get("landing_url") or data.get("url") or "")
    evidence_sources = sorted(sources)
    text_for_risk = " ".join(str(data.get(field) or "") for field in ("reason", "evidence", "evidence_summary", "suggested_manual_check"))
    urls = _candidate_urls(data)
    if any(is_forbidden_url(url) for url in urls) or is_forbidden_text(text_for_risk):
        return PdfDiscoveryResult(status="blocked", reason="blocked_forbidden_source", source=source, landing_url=landing_url, evidence_sources=evidence_sources, risk_flags=["forbidden"])
    if sources <= METADATA_ONLY_SOURCES and sources:
        return PdfDiscoveryResult(status="candidate", reason="metadata_only_source_no_direct_pdf_discovery", source=source, landing_url=landing_url, evidence_sources=evidence_sources, risk_flags=["metadata_only_source"])
    if not _discovery_allowed(data, sources):
        return PdfDiscoveryResult(status="candidate", reason="insufficient_oa_evidence_for_download", source=source, landing_url=landing_url, evidence_sources=evidence_sources, risk_flags=["insufficient_evidence"])

    for url in urls:
        arxiv_pdf = _arxiv_pdf_url(url)
        candidate = arxiv_pdf or url
        if candidate.startswith("https://") and is_direct_pdf_url(candidate):
            return PdfDiscoveryResult(status="direct_pdf_url_found", reason="direct_pdf_url_from_legal_oa_evidence", pdf_url=candidate, landing_url=landing_url or url, source=source, evidence_sources=evidence_sources)

    if data.get("pmcid") and (sources & LEGAL_DISCOVERY_SOURCES or data.get("is_legal_oa_candidate") is True or data.get("unpaywall_is_oa") is True):
        return PdfDiscoveryResult(status="direct_pdf_url_found", reason="pmc_official_pdf_url_from_pmcid", pdf_url=_pmc_pdf_url(data), landing_url=landing_url, source=source or "pmc", evidence_sources=evidence_sources)

    return PdfDiscoveryResult(status="candidate", reason="missing_direct_legal_pdf_url", source=source, landing_url=landing_url, evidence_sources=evidence_sources, risk_flags=["missing_direct_pdf_url"])
