"""Conservative legality assessment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse


FORBIDDEN_SOURCE_KEYWORDS = [
    "sci-hub",
    "scihub",
    "libgen",
    "z-library",
    "zlibrary",
    "z-lib",
    "shadow library",
    "pirate mirror",
    "annas-archive",
    "pdfdrive",
    "booksc",
    "b-ok",
    "不明网盘",
    "论坛盗版资源",
]

FORBIDDEN_TEXT_KEYWORDS = [
    "cookie",
    "session",
    "institution login",
    "campus vpn",
    "school vpn",
    "paywall bypass",
    "captcha solver",
    "proxy credentials",
    "token reuse",
]

OPEN_LICENSE_TERMS = {"cc-by", "cc-by-nc", "cc0", "public-domain", "cc-by-sa", "cc-by-nc-nd", "cc-by-nc-sa"}


ALLOWED_SOURCE_TYPES = [
    "PubMed Central",
    "Europe PMC",
    "arXiv",
    "bioRxiv",
    "medRxiv",
    "Unpaywall confirmed OA",
    "DOAJ",
    "publisher explicit OA",
    "author homepage public PDF",
    "institutional repository public PDF",
]


def _normalized_text(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "")


def is_forbidden_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    haystack = " ".join([url, parsed.netloc, parsed.path]).lower()
    compact = _normalized_text(haystack)
    return any(keyword.replace(" ", "") in compact for keyword in FORBIDDEN_SOURCE_KEYWORDS)


def is_forbidden_text(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    compact = _normalized_text(lowered)
    return any(keyword.replace(" ", "") in compact for keyword in FORBIDDEN_TEXT_KEYWORDS)


@dataclass(slots=True)
class LegalityDecision:
    input_id: str = ""
    query_id: str = ""
    title: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    journal: str = ""
    publication_year: int | None = None
    publication_date: str = ""
    source: str = ""
    candidate_url: str = ""
    pdf_url_candidate: str = ""
    landing_url: str = ""
    license: str = ""
    oa_status: str = ""
    host_type: str = ""
    version: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    is_forbidden: bool = False
    is_legal_oa: bool = False
    is_download_allowed_now: bool = False
    decision: str = "candidate_needs_confirmation"
    reason: str = ""
    required_next_step: str = ""
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(record: dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value not in (None, "", []):
            return str(value)
    return ""


def _sources(record: dict[str, Any]) -> list[str]:
    values = record.get("sources") or record.get("evidence_sources") or record.get("source") or []
    if isinstance(values, list):
        return [str(value) for value in values if value]
    return [str(values)] if values else []


def _open_license(license_value: str) -> bool:
    lowered = (license_value or "").lower()
    return any(term in lowered for term in OPEN_LICENSE_TERMS) or "creativecommons.org/licenses/by/" in lowered


def assess_legal_oa_candidate(record: dict[str, Any]) -> LegalityDecision:
    candidate_url = _field(record, "candidate_url", "url_for_pdf", "best_oa_location", "unpaywall_best_oa_location")
    pdf_url_candidate = _field(record, "pdf_url_candidate", "url_for_pdf")
    landing_url = _field(record, "landing_url", "url")
    license_value = _field(record, "license", "unpaywall_license")
    oa_status = _field(record, "oa_status", "unpaywall_oa_status")
    host_type = _field(record, "host_type")
    version = _field(record, "version")
    source = _field(record, "source")
    sources = _sources(record)
    evidence_text = " ".join(str(record.get(key, "")) for key in ("reason", "evidence", "oa_evidence", "suggested_manual_check", "evidence_summary"))
    urls = [candidate_url, pdf_url_candidate, landing_url]
    decision = LegalityDecision(
        input_id=_field(record, "input_id"),
        query_id=_field(record, "query_id"),
        title=_field(record, "title"),
        doi=_field(record, "doi"),
        pmid=_field(record, "pmid"),
        pmcid=_field(record, "pmcid"),
        journal=_field(record, "journal"),
        publication_year=record.get("publication_year"),
        publication_date=_field(record, "publication_date"),
        source=source,
        candidate_url=candidate_url,
        pdf_url_candidate=pdf_url_candidate,
        landing_url=landing_url,
        license=license_value,
        oa_status=oa_status,
        host_type=host_type,
        version=version,
        evidence_sources=sources,
        evidence_summary=evidence_text,
        is_download_allowed_now=False,
    )
    if any(is_forbidden_url(url) for url in urls) or is_forbidden_text(evidence_text):
        bypass = is_forbidden_text(evidence_text)
        decision.is_forbidden = True
        decision.decision = "blocked_requires_login_or_bypass" if bypass else "blocked_forbidden_source"
        decision.reason = "forbidden_url_or_bypass_text_detected"
        decision.required_next_step = "Do not process this candidate URL."
        decision.risk_flags = ["forbidden"]
        return decision
    if not any(urls):
        decision.decision = "blocked_missing_url"
        decision.reason = "missing_candidate_or_landing_url"
        decision.required_next_step = "Recover a legal landing URL or OA evidence source."
        decision.risk_flags = ["missing_url"]
        return decision

    unpaywall_confirmed = (
        (record.get("unpaywall_is_oa") is True or record.get("is_legal_oa_candidate") is True)
        and host_type in {"publisher", "repository"}
        and (candidate_url or pdf_url_candidate or landing_url)
    )
    europe_pmc_confirmed = source == "europe_pmc" and (record.get("pmcid") or oa_status.lower() in {"open", "gold", "green"})
    source_type = _field(record, "source_type")
    explicit_public_source = source_type in {"author_homepage_public_pdf", "institutional_repository_public_pdf", "publisher_explicit_oa"}
    license_supported = _open_license(license_value)

    if unpaywall_confirmed or europe_pmc_confirmed or (license_supported and (source in {"unpaywall", "europe_pmc"} or explicit_public_source)) or explicit_public_source:
        decision.is_legal_oa = True
        decision.decision = "allowed_for_future_download"
        decision.reason = "eligible_after_round_006_gate_but_download_disabled"
        decision.required_next_step = "Future download round must re-check URL and compute sha256 before saving."
        return decision

    if source in {"openalex", "crossref", "pubmed"}:
        decision.decision = "candidate_needs_confirmation"
        decision.reason = f"{source}_metadata_or_oa_clue_requires_independent_oa_confirmation"
        decision.required_next_step = "Confirm through Unpaywall, PMC/Europe PMC, publisher explicit OA, or repository evidence."
        decision.risk_flags = ["metadata_only_source"]
        return decision

    if license_value and not license_supported:
        decision.decision = "blocked_unsupported_license"
        decision.reason = "unsupported_or_unknown_license"
        decision.required_next_step = "Verify license manually or through an allowed OA source."
        decision.risk_flags = ["unsupported_license"]
        return decision

    decision.decision = "candidate_needs_confirmation"
    decision.reason = "insufficient_legal_oa_evidence"
    decision.required_next_step = "Gather stronger OA evidence before download eligibility."
    decision.risk_flags = ["insufficient_evidence"]
    return decision


def assess_oa_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative OA decision for a future PDF candidate."""

    pdf_url = str(record.get("pdf_url") or record.get("pdf_url_candidate") or "")
    landing_url = str(record.get("landing_url") or "")
    source_type = str(record.get("source_type") or record.get("source") or "")
    evidence = str(record.get("oa_evidence") or record.get("evidence") or "")
    oa_status = str(record.get("oa_status") or "").lower()

    if is_forbidden_url(pdf_url) or is_forbidden_url(landing_url):
        return {
            "is_legal_oa": False,
            "decision": "blocked",
            "reason": "forbidden source keyword detected",
            "evidence": evidence,
        }

    allowed_source = any(source_type.lower() == item.lower() for item in ALLOWED_SOURCE_TYPES)
    explicit_flag = record.get("is_legal_oa") is True
    confirmed_oa = oa_status in {"gold", "green", "hybrid", "bronze", "open"}

    if (allowed_source or explicit_flag) and (pdf_url or landing_url):
        return {
            "is_legal_oa": True,
            "decision": "downloadable",
            "reason": "legal OA evidence is explicitly present",
            "evidence": evidence or source_type,
        }

    if confirmed_oa and "confirmed" in evidence.lower() and (pdf_url or landing_url):
        return {
            "is_legal_oa": True,
            "decision": "downloadable",
            "reason": "confirmed open-access evidence is present",
            "evidence": evidence,
        }

    return {
        "is_legal_oa": False,
        "decision": "candidate",
        "reason": "legal OA status is uncertain; do not download",
        "evidence": evidence,
    }


def assess_unpaywall_oa(record: dict[str, Any]) -> dict[str, Any]:
    """Assess Unpaywall OA evidence without allowing ROUND 005 downloads."""

    urls = [
        str(record.get("url") or ""),
        str(record.get("url_for_pdf") or ""),
        str(record.get("candidate_url") or ""),
        str(record.get("landing_url") or ""),
    ]
    if any(is_forbidden_url(url) for url in urls):
        return {
            "is_legal_oa_candidate": False,
            "is_download_allowed_now": False,
            "oa_evidence": "forbidden URL detected",
            "license": record.get("license", ""),
            "reason": "forbidden_url_detected_in_unpaywall_result",
        }
    is_oa = record.get("is_oa") is True or record.get("unpaywall_is_oa") is True
    license_value = str(record.get("license") or record.get("unpaywall_license") or "")
    host_type = str(record.get("host_type") or "")
    evidence = {
        "oa_status": record.get("oa_status") or record.get("unpaywall_oa_status"),
        "host_type": host_type,
        "version": record.get("version"),
        "url": record.get("url"),
        "url_for_pdf": record.get("url_for_pdf"),
    }
    if is_oa:
        return {
            "is_legal_oa_candidate": True,
            "is_download_allowed_now": False,
            "oa_evidence": evidence,
            "license": license_value,
            "reason": "unpaywall_confirmed_oa_not_downloaded_in_round_005",
        }
    return {
        "is_legal_oa_candidate": False,
        "is_download_allowed_now": False,
        "oa_evidence": evidence,
        "license": license_value,
        "reason": "unpaywall_no_confirmed_oa_location",
    }
