"""Round/query scope filters for append-only audit logs."""

from __future__ import annotations

import re
from typing import Any


def scope_tokens(specs: list[Any], config: dict[str, Any]) -> set[str]:
    tokens = {str(config.get("query_name") or "").strip().lower()}
    for spec in specs:
        value = getattr(spec, "input_id", "")
        if value:
            tokens.add(str(value).strip().lower())
    return {token for token in tokens if token}


def _record_round(value: str) -> int | None:
    match = re.search(r"round[_-](\d{3})", value.lower())
    return int(match.group(1)) if match else None


def _record_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("query_id", "input_id", "query_name", "source_query", "round", "task_nonce"):
        value = record.get(field)
        if value not in (None, "", []):
            values.append(str(value))
    context = record.get("context")
    if isinstance(context, dict):
        values.extend(_record_values(context))
    return values


def record_in_scope(record: dict[str, Any], *, scope: str = "all", tokens: set[str] | None = None, since_round: int | None = None) -> bool:
    if scope == "all" and since_round is None:
        return True
    tokens = tokens or set()
    values = [value.lower() for value in _record_values(record)]
    if since_round is not None:
        rounds = [_record_round(value) for value in values]
        if not any(round_number is not None and round_number >= since_round for round_number in rounds):
            return False
    if scope == "current":
        return any(token and token in value for token in tokens for value in values)
    return True


def filter_records_for_scope(records: list[dict[str, Any]], *, scope: str = "all", tokens: set[str] | None = None, since_round: int | None = None) -> list[dict[str, Any]]:
    return [record for record in records if record_in_scope(record, scope=scope, tokens=tokens, since_round=since_round)]
