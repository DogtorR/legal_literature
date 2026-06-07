"""Explicit LLM node adapters for the LangGraph literature agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_models import SearchIntent
from .agent_state import LiteratureAgentState
from .agent_task import (
    CRISPR_SEARCH_KEYWORDS,
    DNA_NANOSTRUCTURE_SEARCH_KEYWORDS,
    DNA_NANOSTRUCTURE_TERMS,
    LiteratureTask,
    _extract_month_range,
    _explicit_journals,
    _is_crispr_request,
    _is_dna_nanostructure_request,
    _parsed_dna_nanostructure_request,
    parse_task,
)
from .env_loader import load_dotenv
from .llm_client import OpenAICompatibleClient, build_llm_client
from .llm_config import DEFAULT_LLM_CONFIG, AgentConfig, load_llm_config


def build_codex_client(
    *,
    provider: str = "",
    llm_config: str | Path = DEFAULT_LLM_CONFIG,
    env_file: str = ".env",
    no_env: bool = False,
) -> tuple[AgentConfig, OpenAICompatibleClient | None]:
    if not no_env:
        load_dotenv(env_file)
    runtime = load_llm_config(llm_config, provider_override=provider or None)
    if runtime.provider.api_key_env and not runtime.provider.api_key:
        return runtime.agent, None
    return runtime.agent, build_llm_client(runtime.provider)


def _agent_config_from_state(state: LiteratureAgentState) -> AgentConfig:
    options = state.get("options") or {}
    try:
        agent_config, _client = build_codex_client(
            provider=str(options.get("provider") or ""),
            llm_config=str(options.get("llm_config") or DEFAULT_LLM_CONFIG),
            env_file=str(options.get("env_file") or ".env"),
            no_env=bool(options.get("no_env", False)),
        )
        return agent_config
    except Exception:
        return AgentConfig()


def _search_intent_from_task(task: LiteratureTask) -> SearchIntent:
    return SearchIntent(
        query=task.topic or task.original_request,
        keywords=list(task.keywords),
        sources=list(task.sources),
        year_from=task.year_from,
        year_to=task.year_to,
        max_results=task.max_results,
        require_legal_oa=True,
        allow_download=False,
    )


def _merge_llm_task(user_request: str, agent_config: AgentConfig, parsed: dict[str, Any]) -> LiteratureTask:
    fallback = parse_task(user_request, agent_config)
    data = fallback.to_dict()
    for key, value in parsed.items():
        if key in data and value not in (None, "", []):
            data[key] = value

    if _is_crispr_request(user_request):
        data["topic"] = "CRISPR detection"
        data["topic_guard"] = fallback.topic_guard
        keywords = [str(item) for item in data.get("keywords") or [] if str(item).strip()]
        data["keywords"] = [
            keyword
            for keyword in keywords
            if "crispr" in keyword.lower() or "cas" in keyword.lower() or keyword.upper() in {"SHERLOCK", "DETECTR", "HOLMES"}
        ] or CRISPR_SEARCH_KEYWORDS
    elif _is_dna_nanostructure_request(user_request) or _parsed_dna_nanostructure_request(data):
        data["topic"] = "DNA nanostructures"
        data["keywords"] = DNA_NANOSTRUCTURE_SEARCH_KEYWORDS
        data["include_terms"] = DNA_NANOSTRUCTURE_TERMS
        data["sources"] = [source for source in fallback.sources if source != "crossref"]

    explicit_journals = _explicit_journals(user_request)
    if explicit_journals:
        data["journal_filter_mode"] = "explicit_journals"
        data["journal_names"] = explicit_journals
    date_from, date_to = _extract_month_range(user_request)
    if date_from and date_to:
        data["publication_date_from"] = date_from
        data["publication_date_to"] = date_to
        data["year_from"] = int(date_from[:4])
        data["year_to"] = int(date_to[:4])

    data["original_request"] = user_request
    data["require_legal_oa"] = True
    data["allow_download"] = False
    data["needs_user_confirmation"] = agent_config.require_download_confirmation
    return LiteratureTask(**data)


def _fallback_understand(state: LiteratureAgentState, reason: str = "") -> LiteratureAgentState:
    user_request = state.get("user_request", "")
    agent_config = _agent_config_from_state(state)
    task = parse_task(user_request, agent_config)
    result: LiteratureAgentState = dict(state)
    result["task"] = task.to_dict()
    result["search_intent"] = _search_intent_from_task(task).to_dict()
    if reason:
        failures = list(result.get("failures") or [])
        failures.append({"stage": "understand_request_node", "reason": reason, "fallback": "rule_parse"})
        result["failures"] = failures
    return result


def understand_request_with_llm(state: LiteratureAgentState) -> LiteratureAgentState:
    user_request = state.get("user_request", "")
    options = state.get("options") or {}
    try:
        agent_config, client = build_codex_client(
            provider=str(options.get("provider") or ""),
            llm_config=str(options.get("llm_config") or DEFAULT_LLM_CONFIG),
            env_file=str(options.get("env_file") or ".env"),
            no_env=bool(options.get("no_env", False)),
        )
    except Exception as exc:
        return _fallback_understand(state, f"llm_config_failed: {exc}")
    if client is None:
        return _fallback_understand(state, "llm_client_unavailable")

    schema_hint = {
        "topic": "short topic",
        "keywords": ["search keyword"],
        "include_terms": ["optional broad include term"],
        "exclude_terms": ["editorial", "comment", "letter"],
        "year_from": 2024,
        "year_to": 2025,
        "publication_date_from": "2026-05-01",
        "publication_date_to": "2026-05-31",
        "journal_filter_mode": "none|journal_family|jcr_q1|impact_factor|explicit_journals",
        "journal_names": ["Nature Communications"],
        "require_jcr_q1": False,
        "min_impact_factor": None,
        "topic_guard": {"enabled": True, "type": "crispr_detection"},
        "require_legal_oa": True,
        "allow_network_metadata": False,
        "max_results": agent_config.max_results,
        "max_downloads": agent_config.max_downloads,
        "sources": agent_config.default_sources,
        "notes": ["short note"],
    }
    system_prompt = agent_config.system_prompt or (
        "Parse Chinese or English literature requests into strict JSON. "
        "Never authorize paywall bypass, cookies, institutional login, VPN, CAPTCHA solving, Sci-Hub, LibGen, or Z-Library. "
        "For CRISPR detection tasks, do not drift to generic detection without CRISPR."
    )
    prompt = (
        f"Return only JSON matching this shape: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"User request: {user_request}"
    )
    try:
        content = client.complete(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            response_format="json_object",
        )
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        task = _merge_llm_task(user_request, agent_config, parsed)
    except Exception as exc:
        return _fallback_understand(state, f"llm_understanding_failed: {exc}")

    result: LiteratureAgentState = dict(state)
    result["task"] = task.to_dict()
    result["search_intent"] = _search_intent_from_task(task).to_dict()
    messages = list(result.get("messages") or [])
    messages.append({"role": "system", "node": "understand_request_node", "content": "request_understood_with_llm"})
    result["messages"] = messages
    return result


def _rule_diagnosis(state: LiteratureAgentState) -> dict[str, Any]:
    failures = list(state.get("failures") or [])
    metadata_count = len(state.get("metadata_results") or [])
    oa_count = len(state.get("oa_results") or [])
    planned = len(state.get("download_plan") or [])
    downloaded = len([item for item in state.get("download_results") or [] if item.get("status") in {"ok", "downloaded"}])
    reasons: list[str] = []
    actions: list[str] = []
    if downloaded:
        reasons.append("Legal OA downloads completed.")
    elif planned:
        reasons.append("Legal OA candidates were planned, but no real download completed.")
        actions.append("Review allow_download, yes, and individual download failures.")
    elif oa_count:
        reasons.append("OA evidence was checked, but no legal download plan was produced.")
        actions.append("Inspect OA evidence and planning policy.")
    elif metadata_count:
        reasons.append("Metadata was found, but legal OA evidence was not confirmed.")
        actions.append("Try additional OA sources or increase max_results.")
    else:
        reasons.append("No metadata survived the current search.")
        actions.append("Try broader keywords or enable public metadata lookup.")
    if failures:
        reasons.append(f"{len(failures)} failure records were captured.")
    return {"source": "rules", "reasons": reasons, "recommended_actions": actions, "retryable": bool(not downloaded and failures)}


def diagnose_with_llm(state: LiteratureAgentState) -> LiteratureAgentState:
    options = state.get("options") or {}
    if options.get("use_llm_diagnosis") is False:
        result: LiteratureAgentState = dict(state)
        result["diagnostics"] = _rule_diagnosis(state)
        return result
    try:
        _agent_config, client = build_codex_client(
            provider=str(options.get("provider") or ""),
            llm_config=str(options.get("llm_config") or DEFAULT_LLM_CONFIG),
            env_file=str(options.get("env_file") or ".env"),
            no_env=bool(options.get("no_env", False)),
        )
    except Exception as exc:
        result = dict(state)
        diagnostics = _rule_diagnosis(state)
        diagnostics["llm_error"] = f"llm_config_failed: {exc}"
        result["diagnostics"] = diagnostics
        return result
    if client is None:
        result = dict(state)
        diagnostics = _rule_diagnosis(state)
        diagnostics["llm_error"] = "llm_client_unavailable"
        result["diagnostics"] = diagnostics
        return result

    payload = {
        "metadata_results": state.get("metadata_results") or [],
        "oa_results": state.get("oa_results") or [],
        "download_plan": state.get("download_plan") or [],
        "failures": state.get("failures") or [],
        "artifacts": state.get("artifacts") or {},
    }
    try:
        content = client.complete(
            [
                {
                    "role": "system",
                    "content": "Return strict JSON with keys: reasons, recommended_actions, retryable. Do not suggest illegal download sources or bypass.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format="json_object",
        )
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM diagnosis response was not a JSON object")
        parsed["source"] = "llm"
    except Exception as exc:
        parsed = _rule_diagnosis(state)
        parsed["llm_error"] = f"llm_diagnosis_failed: {exc}"
    result = dict(state)
    result["diagnostics"] = parsed
    return result
