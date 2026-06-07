"""Local metadata filtering helpers for dry-run and tests."""

from __future__ import annotations

from datetime import date
from dataclasses import asdict, is_dataclass
import re
from typing import Any, Iterable

from .query import QuerySpec


TEXT_FIELDS = ("title", "abstract", "journal", "authors", "keywords")
MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def _text_blob(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _record_year(record: dict[str, Any]) -> int | None:
    value = record.get("publication_year") or record.get("year")
    if value in (None, ""):
        return None
    return int(value)


def _parse_date(value: Any) -> tuple[date | None, str]:
    text = str(value or "").strip()
    if not text:
        return None, ""
    iso = re.search(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", text)
    if iso:
        year, month = int(iso.group(1)), int(iso.group(2))
        day = int(iso.group(3) or 1)
        precision = "day" if iso.group(3) else "month"
        return date(year, month, day), precision
    year_month = re.search(r"\b(19\d{2}|20\d{2})\s+([A-Za-z]{3,9})\b", text)
    if year_month and year_month.group(2).lower() in MONTH_NAMES:
        return date(int(year_month.group(1)), MONTH_NAMES[year_month.group(2).lower()], 1), "month"
    month_year = re.search(r"\b([A-Za-z]{3,9})\s+(19\d{2}|20\d{2})\b", text)
    if month_year and month_year.group(1).lower() in MONTH_NAMES:
        return date(int(month_year.group(2)), MONTH_NAMES[month_year.group(1).lower()], 1), "month"
    year = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year:
        return date(int(year.group(1)), 1, 1), "year"
    return None, ""


def _record_date(record: dict[str, Any]) -> tuple[date | None, str]:
    for field in ("publication_date", "published_date", "first_publication_date", "pubdate"):
        parsed, precision = _parse_date(record.get(field))
        if parsed:
            return parsed, precision
    year = _record_year(record)
    return (date(year, 1, 1), "year") if year is not None else (None, "")


def _normalize_journal(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def _has_exact_query(query_spec: QuerySpec) -> bool:
    return bool(query_spec.doi or query_spec.pmid or query_spec.title)


def matched_keywords(record: Any, query_spec: QuerySpec) -> list[str]:
    blob = _text_blob(_record_dict(record))
    terms = list(query_spec.keywords or []) + list(query_spec.include_terms or [])
    return sorted({term for term in terms if term and term.lower() in blob})


def year_in_range(record: Any, query_spec: QuerySpec) -> tuple[bool, str | None]:
    data = _record_dict(record)
    year = _record_year(data)
    if year is None:
        return True, "missing_year"
    if query_spec.year_from is not None and year < query_spec.year_from:
        return False, "year_before_range"
    if query_spec.year_to is not None and year > query_spec.year_to:
        return False, "year_after_range"
    return True, None


def publication_date_in_range(record: Any, query_spec: QuerySpec) -> tuple[bool, str | None]:
    start, _ = _parse_date(query_spec.publication_date_from)
    end, _ = _parse_date(query_spec.publication_date_to)
    if not start and not end:
        return True, None
    data = _record_dict(record)
    record_date, precision = _record_date(data)
    if not record_date:
        return False, "missing_publication_date"
    if precision == "year" and ((start and start.month != 1) or (end and end.month != 12)):
        return False, "publication_date_precision_too_low"
    if start and record_date < start:
        return False, "publication_date_before_range"
    if end and record_date > end:
        return False, "publication_date_after_range"
    return True, None


def text_matches_keywords(record: Any, query_spec: QuerySpec) -> bool:
    if not query_spec.keywords or _has_exact_query(query_spec):
        return True
    blob = _text_blob(_record_dict(record))
    return any(term.lower() in blob for term in query_spec.keywords)


def include_terms_match(record: Any, query_spec: QuerySpec) -> bool:
    if not query_spec.include_terms:
        return True
    blob = _text_blob(_record_dict(record))
    return any(term.lower() in blob for term in query_spec.include_terms)


def exclude_terms_block(record: Any, query_spec: QuerySpec) -> bool:
    if not query_spec.exclude_terms:
        return False
    blob = _text_blob(_record_dict(record))
    return any(term.lower() in blob for term in query_spec.exclude_terms)


def journal_matches_scope(record: Any, journals: Iterable[str]) -> bool:
    wanted = {_normalize_journal(journal) for journal in journals if str(journal).strip()}
    if not wanted:
        return True
    data = _record_dict(record)
    values = [
        data.get("journal"),
        data.get("host_venue"),
        data.get("container_title"),
        (data.get("primary_location") or {}).get("source") if isinstance(data.get("primary_location"), dict) else "",
    ]
    return any(_normalize_journal(value) in wanted for value in values if value)


def journal_matches_family(record: Any, family: str, journals: Iterable[str] | None = None) -> bool:
    family_key = _normalize_journal(family)
    if not family_key:
        return True
    data = _record_dict(record)
    values = [
        data.get("journal"),
        data.get("host_venue"),
        data.get("container_title"),
        (data.get("primary_location") or {}).get("source") if isinstance(data.get("primary_location"), dict) else "",
    ]
    normalized_values = [_normalize_journal(value) for value in values if value]
    if journals and journal_matches_scope(data, journals):
        return True
    if family_key == "nature":
        nature_portfolio_prefixes = ("nature ", "npj ", "communications ")
        nature_portfolio_exact = {"nature", "scientific reports", "scientific data"}
        return any(
            value in nature_portfolio_exact or value.startswith(nature_portfolio_prefixes)
            for value in normalized_values
        )
    return any(value == family_key or value.startswith(f"{family_key} ") for value in normalized_values)


def record_matches_scope(record: Any, scope: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not scope:
        return True, None
    data = _record_dict(record)
    journals = scope.get("journals") or []
    if scope.get("require_journal_match") and not journal_matches_scope(data, journals):
        return False, "journal_not_in_scope"
    if scope.get("require_journal_family_match") and not journal_matches_family(data, str(scope.get("journal_family") or ""), journals):
        return False, "journal_family_not_in_scope"
    date_spec = QuerySpec(
        publication_date_from=str(scope.get("publication_date_from") or ""),
        publication_date_to=str(scope.get("publication_date_to") or ""),
    )
    date_ok, date_reason = publication_date_in_range(data, date_spec)
    if scope.get("require_date_match") and not date_ok:
        return False, date_reason or "publication_date_not_in_scope"
    return True, None


def apply_record_scope_filters(records: Iterable[Any], scope: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats: dict[str, int] = {"scope_input_count": 0, "scope_kept_count": 0, "scope_removed_count": 0}
    for record in records:
        stats["scope_input_count"] += 1
        data = _record_dict(record)
        ok, reason = record_matches_scope(data, scope)
        if ok:
            kept.append(data)
            continue
        stats["scope_removed_count"] += 1
        if reason:
            stats[f"scope_removed_{reason}"] = stats.get(f"scope_removed_{reason}", 0) + 1
    stats["scope_kept_count"] = len(kept)
    return kept, stats


def apply_query_filters(records: Iterable[Any], query_spec: QuerySpec, *, require_keyword_match: bool = True) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        data = _record_dict(record)
        reasons: list[str] = list(data.get("filter_reasons") or [])
        year_ok, year_reason = year_in_range(data, query_spec)
        if year_reason:
            reasons.append(year_reason)
        if not year_ok:
            continue
        date_ok, date_reason = publication_date_in_range(data, query_spec)
        if date_reason:
            reasons.append(date_reason)
        if not date_ok:
            continue
        if query_spec.journal and not journal_matches_scope(data, [query_spec.journal]):
            continue
        if require_keyword_match and not text_matches_keywords(data, query_spec):
            continue
        if not include_terms_match(data, query_spec):
            continue
        if exclude_terms_block(data, query_spec):
            continue
        matches = matched_keywords(data, query_spec)
        if matches:
            data["matched_keywords"] = matches
        if reasons:
            data["filter_reasons"] = sorted(set(reasons))
        filtered.append(data)
    return filtered
