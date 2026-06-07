"""Shared metadata records and conservative deduplication helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class MetadataRecord:
    query_id: str = ""
    title: str = ""
    doi: str = ""
    pmid: str = ""
    openalex_id: str = ""
    publication_year: int | None = None
    publication_date: str = ""
    publication_type: str = ""
    journal: str = ""
    authors: list[str] | None = None
    abstract: str = ""
    source: str = ""
    landing_url: str = ""
    oa_status: str = ""
    best_oa_location: str = ""
    pdf_url_candidate: str = ""
    license: str = ""
    matched_keywords: list[str] | None = None
    score: float = 0.0
    sources: list[str] | None = None
    oa_evidence: list[str] | None = None

    def __post_init__(self) -> None:
        self.authors = self.authors or []
        self.matched_keywords = self.matched_keywords or []
        self.sources = self.sources or ([self.source] if self.source else [])
        self.oa_evidence = self.oa_evidence or []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_doi(doi: str) -> str:
    text = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


def normalize_title(title: str) -> str:
    text = re.sub(r"\s+", " ", title or "").strip().lower()
    return re.sub(r"[^\w\s]", "", text)


def dedupe_key(record: MetadataRecord | dict[str, Any]) -> str:
    data = record.to_dict() if isinstance(record, MetadataRecord) else record
    doi = normalize_doi(str(data.get("doi") or ""))
    if doi:
        return f"doi:{doi}"
    for field in ("pmid", "openalex_id"):
        value = str(data.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    year = data.get("publication_year") or ""
    return f"title:{normalize_title(str(data.get('title') or ''))}:{year}"


def _as_dict(record: MetadataRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, MetadataRecord) else dict(record)


def _merge_list_values(first: Any, second: Any) -> list[Any]:
    values: list[Any] = []
    for value in (first, second):
        if value in (None, ""):
            continue
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    merged: list[Any] = []
    for value in values:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_records(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key in {"sources", "oa_evidence", "matched_keywords", "authors"}:
            merged[key] = _merge_list_values(merged.get(key), value)
        elif key == "source":
            merged["source"] = merged.get("source") or value
            merged["sources"] = _merge_list_values(merged.get("sources"), value)
        elif key in {"oa_status", "best_oa_location", "license"}:
            if value and value != merged.get(key):
                merged[key] = merged.get(key) or value
        elif key.startswith("unpaywall_"):
            if value not in (None, "", []):
                merged[key] = value
        elif value not in (None, "", []):
            merged[key] = merged.get(key) or value
    return merged


def deduplicate_records(records: Iterable[MetadataRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        data = _as_dict(record)
        data["doi"] = normalize_doi(str(data.get("doi") or ""))
        if data.get("source") and not data.get("sources"):
            data["sources"] = [data["source"]]
        key = dedupe_key(data)
        by_key[key] = _merge_records(by_key[key], data) if key in by_key else data
    return list(by_key.values())


def deduplicate(records: Iterable[MetadataRecord | dict[str, Any]]) -> list[dict[str, Any]]:
    return deduplicate_records(records)
