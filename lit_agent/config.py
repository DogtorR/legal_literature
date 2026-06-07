"""Configuration loading and validation for the legal literature agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import write_failure


DEFAULT_CONFIG: dict[str, Any] = {
    "project_name": "",
    "query_name": "",
    "keywords": [],
    "year_from": None,
    "year_to": None,
    "publication_date_from": "",
    "publication_date_to": "",
    "publication_types": [],
    "include_terms": [],
    "exclude_terms": [],
    "journals": [],
    "authors": [],
    "max_results_per_source": 100,
    "max_downloads": 0,
    "dry_run": True,
    "sources": ["openalex"],
    "queries_file": "",
    "download_policy": "legal_oa_only",
    "download_policy_details": {},
}

ALLOWED_SOURCES = {"openalex", "crossref", "pubmed", "europe_pmc", "unpaywall", "arxiv", "doaj"}
ALLOWED_DOWNLOAD_POLICIES = {"legal_oa_only", "metadata_only", "dry_run_only"}
FORBIDDEN_BYPASS_FIELDS = {
    "use_cookie",
    "use_cookies",
    "cookie",
    "cookies",
    "login",
    "vpn",
    "school_vpn",
    "institutional_account",
    "paywall_bypass",
    "captcha_bypass",
}
DOWNLOAD_POLICY_DETAIL_FIELDS = {
    "only_legal_open_access",
    "uncertain_to_candidates",
    "require_license_or_oa_evidence",
    "downloads_enabled",
}


LIST_FIELDS = {
    "keywords",
    "publication_types",
    "include_terms",
    "exclude_terms",
    "journals",
    "authors",
    "sources",
}


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_inline_list(value: str) -> list[Any]:
    value = value.strip()
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(part) for part in inner.split(",")]
    return [_coerce_scalar(value)]


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the example config."""

    data: dict[str, Any] = {}
    current_key: str | None = None
    current_mapping_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent and current_mapping_key and ":" in stripped and not stripped.startswith("- "):
            key, value = stripped.split(":", 1)
            mapping = data.setdefault(current_mapping_key, {})
            if mapping == []:
                mapping = {}
                data[current_mapping_key] = mapping
            if not isinstance(mapping, dict):
                raise ValueError(f"YAML key {current_mapping_key!r} cannot be both list/scalar and mapping")
            mapping[key.strip()] = _coerce_scalar(value.strip())
            continue
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError("YAML list item found before a key")
            data.setdefault(current_key, [])
            data[current_key].append(_coerce_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        current_mapping_key = None
        if value == "":
            data[key] = []
        elif value.startswith("[") and value.endswith("]"):
            data[key] = _parse_inline_list(value)
        elif value == "{}":
            data[key] = {}
            current_mapping_key = key
        else:
            data[key] = _coerce_scalar(value)
        if value == "":
            current_mapping_key = key
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("search_config.yaml must contain a mapping")
    return loaded


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace("|", ";").split(";") if part.strip()]


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        write_failure("invalid integer config value", {"value": value})
        raise ValueError(f"Expected integer value, got {value!r}") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return True
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    write_failure("invalid boolean config value", {"value": value})
    raise ValueError(f"Expected boolean value, got {value!r}")


def _reject_bypass_fields(raw: dict[str, Any]) -> None:
    forbidden = sorted(set(raw) & FORBIDDEN_BYPASS_FIELDS)
    policy_value = raw.get("download_policy") or ""
    policy = str(policy_value).lower()
    if isinstance(policy_value, dict):
        forbidden.extend(sorted(set(policy_value) & FORBIDDEN_BYPASS_FIELDS))
    policy_mentions_bypass = any(field in policy for field in FORBIDDEN_BYPASS_FIELDS)
    if forbidden or policy_mentions_bypass:
        context = {"forbidden_fields": forbidden, "download_policy": raw.get("download_policy")}
        write_failure("forbidden bypass field detected in config", context)
        raise ValueError("Configuration contains forbidden bypass fields or policy")


def _validate_year_range(year_from: int | None, year_to: int | None) -> None:
    if year_from is not None and year_to is not None and year_from > year_to:
        write_failure("invalid year range", {"year_from": year_from, "year_to": year_to})
        raise ValueError("year_from cannot be greater than year_to")


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw:
        _reject_bypass_fields(raw)
    config = dict(DEFAULT_CONFIG)
    if raw:
        config.update(raw)
    if config.get("download_policy") == [] and any(field in config for field in DOWNLOAD_POLICY_DETAIL_FIELDS):
        config["download_policy"] = {field: config[field] for field in DOWNLOAD_POLICY_DETAIL_FIELDS if field in config}
    for field in LIST_FIELDS:
        config[field] = _as_list(config.get(field))
    config["year_from"] = _as_int_or_none(config.get("year_from"))
    config["year_to"] = _as_int_or_none(config.get("year_to"))
    _validate_year_range(config["year_from"], config["year_to"])
    config["max_results_per_source"] = int(config.get("max_results_per_source") or 100)
    config["max_downloads"] = int(config.get("max_downloads") or 0)
    config["dry_run"] = _as_bool(config.get("dry_run", True))
    unknown_sources = sorted(set(config["sources"]) - ALLOWED_SOURCES)
    if unknown_sources:
        write_failure("unknown metadata source in config", {"sources": unknown_sources})
        raise ValueError(f"Unknown sources: {', '.join(unknown_sources)}")
    policy_value = config.get("download_policy") or "legal_oa_only"
    if isinstance(policy_value, dict):
        details = dict(policy_value)
        config["download_policy_details"] = details
        if details.get("only_legal_open_access", True):
            config["download_policy"] = "legal_oa_only"
        elif details.get("metadata_only"):
            config["download_policy"] = "metadata_only"
        else:
            config["download_policy"] = "dry_run_only"
    else:
        config["download_policy"] = str(policy_value)
        config["download_policy_details"] = dict(config.get("download_policy_details") or {})
    if config["download_policy"] not in ALLOWED_DOWNLOAD_POLICIES:
        write_failure("unknown download policy", {"download_policy": config["download_policy"]})
        raise ValueError(f"Unknown download_policy: {config['download_policy']}")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and normalize a search_config.yaml file."""

    config_path = Path(path)
    raw = _load_yaml(config_path)
    return normalize_config(raw)


def load_search_config(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for ROUND 001 code."""

    return load_config(path)
