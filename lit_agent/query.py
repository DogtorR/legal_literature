"""Query normalization for CSV, YAML config, and CLI inputs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .manifest import write_failure


QUERY_FIELDS = [
    "input_id",
    "title",
    "doi",
    "pmid",
    "keywords",
    "year_from",
    "year_to",
    "publication_date_from",
    "publication_date_to",
    "publication_year",
    "publication_type",
    "journal",
    "author",
    "max_results",
    "include_terms",
    "exclude_terms",
    "source_preference",
]


@dataclass(slots=True)
class QuerySpec:
    input_id: str = ""
    title: str = ""
    doi: str = ""
    pmid: str = ""
    keywords: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    publication_date_from: str = ""
    publication_date_to: str = ""
    publication_year: int | None = None
    publication_type: str = ""
    journal: str = ""
    author: str = ""
    max_results: int | None = None
    include_terms: list[str] | None = None
    exclude_terms: list[str] | None = None
    source_preference: list[str] | None = None

    def __post_init__(self) -> None:
        self.keywords = self.keywords or []
        self.include_terms = self.include_terms or []
        self.exclude_terms = self.exclude_terms or []
        self.source_preference = self.source_preference or []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_terms(value: Any) -> list[str]:
    return normalize_terms(value)


def normalize_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if ";" in text or "|" in text:
        text = text.replace("|", ";")
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        write_failure("invalid integer query value", {"value": value})
        raise ValueError(f"Expected integer query value, got {value!r}") from exc


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def query_from_dict(data: dict[str, Any], default_input_id: str = "query") -> QuerySpec:
    spec = QuerySpec(
        input_id=_string(data.get("input_id") or data.get("query_id") or default_input_id),
        title=_string(data.get("title")),
        doi=_string(data.get("doi")).lower(),
        pmid=_string(data.get("pmid")),
        keywords=_parse_terms(data.get("keywords")),
        year_from=_parse_int(data.get("year_from")),
        year_to=_parse_int(data.get("year_to")),
        publication_date_from=_string(data.get("publication_date_from")),
        publication_date_to=_string(data.get("publication_date_to")),
        publication_year=_parse_int(data.get("publication_year")),
        publication_type=_string(data.get("publication_type")),
        journal=_string(data.get("journal")),
        author=_string(data.get("author")),
        max_results=_parse_int(data.get("max_results") or data.get("max_results_per_source")),
        include_terms=_parse_terms(data.get("include_terms")),
        exclude_terms=_parse_terms(data.get("exclude_terms")),
        source_preference=_parse_terms(data.get("source_preference") or data.get("sources")),
    )
    validate_query_spec(spec)
    return spec


def validate_query_spec(spec: QuerySpec) -> None:
    if spec.year_from is not None and spec.year_to is not None and spec.year_from > spec.year_to:
        write_failure("invalid query year range", {"input_id": spec.input_id, "year_from": spec.year_from, "year_to": spec.year_to})
        raise ValueError("year_from cannot be greater than year_to")
    if spec.max_results is not None and spec.max_results <= 0:
        write_failure("invalid max_results", {"input_id": spec.input_id, "max_results": spec.max_results})
        raise ValueError("max_results must be greater than 0")


def _merge_defaults(row: dict[str, Any], defaults: dict[str, Any] | None) -> dict[str, Any]:
    if not defaults:
        return row
    merged = dict(defaults)
    for key, value in row.items():
        if value not in (None, ""):
            merged[key] = value
    return merged


def _config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "keywords": config.get("keywords"),
        "year_from": config.get("year_from"),
        "year_to": config.get("year_to"),
        "publication_date_from": config.get("publication_date_from"),
        "publication_date_to": config.get("publication_date_to"),
        "publication_type": ";".join(config.get("publication_types") or []),
        "journal": (config.get("journals") or [""])[0],
        "max_results": config.get("max_results_per_source"),
        "include_terms": config.get("include_terms"),
        "exclude_terms": config.get("exclude_terms"),
        "source_preference": config.get("sources"),
    }


def load_queries_csv(path: str | Path, defaults: dict[str, Any] | None = None) -> list[QuerySpec]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [query_from_dict(_merge_defaults(row, defaults), default_input_id=f"row_{idx}") for idx, row in enumerate(reader, 1)]


def query_specs_from_config(config_or_path: dict[str, Any] | str | Path) -> list[QuerySpec]:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    defaults = _config_defaults(config)
    defaults["input_id"] = config.get("query_name") or config.get("project_name") or "config_query"
    query = query_from_dict(defaults, default_input_id="config_query")
    return [query]


def merge_cli_overrides(specs: Iterable[QuerySpec], args: Any) -> list[QuerySpec]:
    if isinstance(args, dict):
        overrides = args
    else:
        overrides = {
            "year_from": getattr(args, "year_from", None),
            "year_to": getattr(args, "year_to", None),
            "keywords": getattr(args, "keywords", None),
            "max_results": getattr(args, "max_results", None),
            "include_terms": getattr(args, "include_terms", None),
            "exclude_terms": getattr(args, "exclude_terms", None),
        }
    normalized: list[QuerySpec] = []
    for spec in specs:
        data = spec.to_dict()
        for key, value in overrides.items():
            if value is not None and value != "":
                data[key] = value
        normalized.append(query_from_dict(data, default_input_id=spec.input_id or "cli_query"))
    return normalized


def queries_from_csv(path: str | Path) -> list[QuerySpec]:
    return load_queries_csv(path)


def queries_from_config(config_or_path: dict[str, Any] | str | Path) -> list[QuerySpec]:
    return query_specs_from_config(config_or_path)


def apply_overrides(specs: Iterable[QuerySpec], overrides: dict[str, Any]) -> list[QuerySpec]:
    return merge_cli_overrides(specs, overrides)
