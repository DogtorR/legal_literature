"""Lightweight structured models shared by graph nodes and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchIntent:
    query: str
    keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    max_results: int = 20
    require_legal_oa: bool = True
    allow_download: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    status: str
    tool: str
    data: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    route: str
    reason: str = ""
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DownloadPolicyDecision:
    allowed: bool
    blocked: bool = False
    reason: str = ""
    require_legal_oa: bool = True
    allow_download: bool = False
    yes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
