"""Shared safe HTTP helpers for metadata-only source adapters."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..manifest import write_failure


DEFAULT_USER_AGENT = "LEGAL_LITERATURE_AGENT/0.1 (mailto:contact@example.org)"
FORBIDDEN_HEADER_NAMES = {"cookie", "authorization", "proxy-authorization"}


def safe_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
    merged = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        for key, value in headers.items():
            if key.lower() in FORBIDDEN_HEADER_NAMES:
                write_failure("forbidden request header", {"header": key})
                raise ValueError(f"Forbidden request header: {key}")
            merged[key] = value
    if not merged.get("User-Agent"):
        merged["User-Agent"] = DEFAULT_USER_AGENT
    return merged


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    retries: int = 3,
    rate_limit_seconds: float = 0.2,
    session: Any = None,
    source: str = "unknown",
) -> tuple[dict[str, Any], int]:
    """GET JSON with safe headers, retry accounting, and failure logging."""

    params = params or {}
    request_headers = safe_headers(headers)
    attempts = 0
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        attempts += 1
        try:
            if rate_limit_seconds:
                time.sleep(rate_limit_seconds)
            if session is not None:
                response = session.get(url, params=params, headers=request_headers, timeout=timeout)
                status_code = int(getattr(response, "status_code", 200))
                if status_code >= 400:
                    raise RuntimeError(f"{source} HTTP {status_code}")
                payload = response.json()
            else:
                separator = "&" if "?" in url else "?"
                full_url = f"{url}{separator}{urlencode(params, doseq=True)}" if params else url
                request = Request(full_url, headers=request_headers)
                with urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{source} response JSON must be an object")
            return payload, attempts
        except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    write_failure(f"{source} metadata request failed", {"url": url, "params": params, "error": str(last_error), "attempts": attempts})
    raise RuntimeError(str(last_error))


def dry_run_mock_records(source: str, query_id: str) -> list[dict[str, Any]]:
    from ..manifest import read_jsonl

    records = read_jsonl("examples/mock_metadata.jsonl")
    return [dict(record, source=source, sources=[source], query_id=query_id) for record in records]


def require_keyword_match(config: dict[str, Any]) -> bool:
    guard = config.get("topic_guard") or {}
    return not (isinstance(guard, dict) and guard.get("enabled"))
