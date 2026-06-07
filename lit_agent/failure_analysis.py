"""Failure classification and JSONL dedupe audit helpers."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any


FAILURE_METADATA = {
    "validation_test_expected": ("info", "Keep as test evidence; no production action needed."),
    "unpaywall_404": ("warning", "Do not retry aggressively; recheck DOI only in a future metadata refresh."),
    "unpaywall_no_doi_or_mock": ("info", "Recover DOI or exclude mock/test records from OA checks."),
    "preflight_403": ("warning", "Do not bypass access controls; leave as blocked candidate."),
    "preflight_404": ("warning", "Do not retry known missing URLs unless a fresh OA source provides a new URL."),
    "missing_pdf_url": ("info", "Keep as candidate until a legal direct PDF URL is independently confirmed."),
    "landing_direct_pdf_guard": ("info", "Treat as guarded clue; do not parse as landing HTML."),
    "landing_login_paywall_block": ("warning", "Do not bypass login, cookie, subscription, or paywall signals."),
    "task_command_defect": ("info", "Use corrected local audit command and preserve the defect record."),
    "jsonl_parse": ("error", "Fix or quarantine malformed JSONL before downstream processing."),
    "source_network": ("warning", "Record source failure and continue other legal metadata sources."),
    "forbidden_or_bypass": ("blocking", "Stop processing the URL and audit for forbidden access."),
    "other": ("warning", "Review manually and add a more specific classifier if repeated."),
}


def _record_text(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True).lower()


def classify_failure(record: dict[str, Any]) -> str:
    text = _record_text(record)
    reason = str(record.get("reason") or "").lower()
    context = record.get("context") if isinstance(record.get("context"), dict) else {}
    error = str(context.get("error") or record.get("error") or "").lower()
    status = str(context.get("status") or record.get("status") or "").lower()
    pdf_url = str(context.get("pdf_url") or record.get("pdf_url") or "").lower()

    if "sci-hub" in text or "libgen" in text or "z-library" in text or "paywall bypass" in text or "captcha solver" in text:
        return "forbidden_or_bypass"
    if "bad jsonl line skipped" in reason or "jsondecodeerror" in text:
        return "jsonl_parse"
    if "pytest validation" in text or "validation_test_expected" in text:
        return "validation_test_expected"
    if "glob('')" in text or "unacceptable pattern" in error or "task downloads check command failed" in reason:
        return "task_command_defect"
    if "doi landing page blocked direct pdf" in reason or "direct_pdf_url_not_fetched" in text or "redirect_to_pdf_not_fetched" in text or "pdf_response_not_fetched" in text:
        return "landing_direct_pdf_guard"
    if "doi landing page blocked for login or paywall" in reason or "requires_login_or_paywall" in text or "cookieabsent" in text or "cookie_not_supported" in text:
        return "landing_login_paywall_block"
    if "unpaywall" in text and ("404" in text or "not found" in text):
        return "unpaywall_404"
    if "unpaywall" in text and ("missing_doi" in text or "10.1000" in text or "mock" in text or "placeholder" in text):
        return "unpaywall_no_doi_or_mock"
    if "missing_pdf_url" in text or (not pdf_url and "download preflight" in reason):
        return "missing_pdf_url"
    if status == "403" or "access_requires_auth_or_paywall:403" in text or "http 403" in text:
        return "preflight_403"
    if status == "404" or "landing_http_status_404" in text or "http 404" in text:
        return "preflight_404"
    if "network" in text or "timeout" in text or "urlerror" in text or "check failed" in text:
        return "source_network"
    return "other"


def summarize_failures(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        category = classify_failure(record)
        severity, action = FAILURE_METADATA[category]
        bucket = buckets.setdefault(
            category,
            {
                "count": 0,
                "severity": severity,
                "recommended_action": action,
                "example": record,
            },
        )
        bucket["count"] += 1
    for category, (severity, action) in FAILURE_METADATA.items():
        buckets.setdefault(
            category,
            {
                "count": 0,
                "severity": severity,
                "recommended_action": action,
                "example": None,
            },
        )
    return {
        "total_rows": len(records),
        "categories": dict(sorted(buckets.items())),
    }


def candidate_dedupe_key(record: dict[str, Any]) -> str:
    return "|".join(
        str(record.get(field) or "").strip().lower()
        for field in ("doi", "candidate_url", "pdf_url_candidate", "reason", "source")
    )


def search_log_dedupe_key(record: dict[str, Any]) -> str:
    context = record.get("context") if isinstance(record.get("context"), dict) else record
    query_summary = context.get("query_summary") if isinstance(context.get("query_summary"), dict) else {}
    query_summary_key = json.dumps(query_summary, ensure_ascii=False, sort_keys=True)
    return "|".join(
        [
            str(record.get("event") or context.get("event") or "").strip().lower(),
            str(record.get("source") or context.get("source") or "").strip().lower(),
            str(context.get("query_id") or "").strip().lower(),
            query_summary_key,
        ]
    )


def summarize_dedupe(records: list[dict[str, Any]], *, key_name: str) -> dict[str, Any]:
    key_func = candidate_dedupe_key if key_name == "candidate" else search_log_dedupe_key
    keys = [key_func(record) for record in records]
    counts = Counter(keys)
    duplicates = [(key, count) for key, count in counts.items() if count > 1]
    top = sorted(duplicates, key=lambda item: (-item[1], item[0]))[:10]
    return {
        "total_rows": len(records),
        "unique_keys": len(counts),
        "duplicate_rows": sum(count - 1 for _, count in duplicates),
        "top_duplicate_keys": [{"key": key, "count": count} for key, count in top],
    }
