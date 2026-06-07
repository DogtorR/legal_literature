"""Topic guards that prevent LLM keyword expansion from drifting off-topic."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


CRISPR_TERMS = (
    "crispr",
    "cas",
    "cas12",
    "cas12a",
    "cas13",
    "cas13a",
    "cas14",
    "sherlock",
    "detectr",
    "holmes",
)

DETECTION_TERMS = (
    "detect",
    "detection",
    "diagnostic",
    "diagnostics",
    "diagnosis",
    "biosens",
    "biosensor",
    "sensor",
    "sensing",
    "assay",
    "test",
    "testing",
    "liquid biopsy",
    "single-nucleotide variant",
    "single nucleotide variant",
    "variant profiling",
    "snv",
    "nucleic acid",
    "molecular diagnostic",
    "point-of-care",
    "point of care",
    "poct",
)

TEXT_FIELDS = ("title", "abstract", "journal", "authors", "keywords", "matched_keywords", "publication_type")


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if is_dataclass(record):
        return asdict(record)
    return dict(getattr(record, "__dict__", {}))


def _text_blob(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts).lower()


def has_any_term(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        lowered = term.lower()
        if not lowered:
            continue
        if lowered.isalnum() and len(lowered) <= 3:
            import re

            if re.search(rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])", text):
                return True
            continue
        if lowered in text:
            return True
    return False


def is_crispr_detection_record(record: Any, *, crispr_terms: Iterable[str] = CRISPR_TERMS, detection_terms: Iterable[str] = DETECTION_TERMS) -> bool:
    text = _text_blob(_as_dict(record))
    return has_any_term(text, crispr_terms) and has_any_term(text, detection_terms)


def apply_topic_guard(records: Iterable[Any], guard: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not guard or not guard.get("enabled"):
        return [_as_dict(record) for record in records]
    guard_type = str(guard.get("type") or "").lower()
    if guard_type not in {"crispr_detection", "custom_dual_term"}:
        return [_as_dict(record) for record in records]

    crispr_terms = guard.get("primary_terms") or CRISPR_TERMS
    detection_terms = guard.get("secondary_terms") or DETECTION_TERMS
    kept: list[dict[str, Any]] = []
    for record in records:
        data = _as_dict(record)
        if has_any_term(_text_blob(data), crispr_terms) and has_any_term(_text_blob(data), detection_terms):
            data["topic_guard_passed"] = True
            kept.append(data)
    return kept
