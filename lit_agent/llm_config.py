"""External LLM provider configuration for the conversational agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_LLM_CONFIG = Path("llm_config.yaml")


@dataclass(slots=True)
class ProviderConfig:
    name: str
    type: str
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 4000
    timeout: int = 60
    retry_count: int = 2

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass(slots=True)
class AgentConfig:
    language: str = "zh"
    require_download_confirmation: bool = True
    default_year_from: int | None = None
    default_year_to: int | None = None
    default_topic_expansion: bool = True
    default_sources: list[str] | None = None
    max_results: int = 20
    max_downloads: int = 20
    system_prompt: str = ""
    task_parse_prompt: str = ""
    report_prompt: str = ""

    def __post_init__(self) -> None:
        self.default_sources = self.default_sources or ["openalex", "crossref", "europe_pmc", "pubmed", "unpaywall"]


@dataclass(slots=True)
class LLMRuntimeConfig:
    active_provider: str
    provider: ProviderConfig
    agent: AgentConfig
    raw: dict[str, Any]


def _load_structured_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return json.loads(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("LLM config must contain a mapping")
    return loaded


def _as_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [part.strip() for part in str(value).replace("|", ";").split(";") if part.strip()]


def load_llm_config(path: str | Path = DEFAULT_LLM_CONFIG, provider_override: str | None = None) -> LLMRuntimeConfig:
    config_path = Path(path)
    raw = _load_structured_text(config_path)
    active = str(provider_override or raw.get("active_provider") or "").strip()
    providers = raw.get("providers") or {}
    if not active:
        raise ValueError("LLM config is missing active_provider")
    if active not in providers:
        raise ValueError(f"Unknown LLM provider: {active}")
    provider_raw = dict(providers[active] or {})
    provider = ProviderConfig(
        name=active,
        type=str(provider_raw.get("type") or "openai_compatible"),
        base_url=str(provider_raw.get("base_url") or "").rstrip("/"),
        api_key_env=str(provider_raw.get("api_key_env") or ""),
        model=str(provider_raw.get("model") or ""),
        temperature=float(provider_raw.get("temperature", 0.2)),
        max_tokens=int(provider_raw.get("max_tokens", 4000)),
        timeout=int(provider_raw.get("timeout", 60)),
        retry_count=int(provider_raw.get("retry_count", 2)),
    )
    if provider.type != "openai_compatible":
        raise ValueError(f"Unsupported provider type: {provider.type}")
    if not provider.base_url or not provider.model:
        raise ValueError(f"Provider {active} must define base_url and model")

    agent_raw = dict(raw.get("agent") or {})
    agent = AgentConfig(
        language=str(agent_raw.get("language") or "zh"),
        require_download_confirmation=bool(agent_raw.get("require_download_confirmation", True)),
        default_year_from=_as_int_or_none(agent_raw.get("default_year_from")),
        default_year_to=_as_int_or_none(agent_raw.get("default_year_to")),
        default_topic_expansion=bool(agent_raw.get("default_topic_expansion", True)),
        default_sources=_as_list(agent_raw.get("default_sources"), ["openalex", "crossref", "europe_pmc", "pubmed", "unpaywall"]),
        max_results=int(agent_raw.get("max_results", 20)),
        max_downloads=int(agent_raw.get("max_downloads", 20)),
        system_prompt=str(agent_raw.get("system_prompt") or ""),
        task_parse_prompt=str(agent_raw.get("task_parse_prompt") or ""),
        report_prompt=str(agent_raw.get("report_prompt") or ""),
    )
    return LLMRuntimeConfig(active_provider=active, provider=provider, agent=agent, raw=raw)
