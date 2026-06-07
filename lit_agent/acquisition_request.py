"""Natural-language acquisition request parsing and query profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import json
import re
from pathlib import Path
from typing import Any


LEGAL_FULLTEXT_FORMATS = ["pdf", "pmc_xml", "europe_pmc_xml", "publisher_html", "repository_file", "preprint_file"]
DEFAULT_SOURCES = ["pubmed", "openalex", "crossref", "europe_pmc"]
CRISPR_BROAD_QUERY = '(CRISPR OR "CRISPR-Cas" OR Cas9 OR Cas12 OR Cas12a OR Cas13 OR Cas13a OR Cas14 OR Cpf1 OR C2c2 OR SHERLOCK OR DETECTR OR HOLMES)'
DEFAULT_EXCLUDE_TERMS = ["editorial", "comment", "letter"]
GENE_EDITING_EXCLUDE_TERMS = [
    "gene editing",
    "genome editing",
    "base editing",
    "prime editing",
    "knockout",
    "knock-in",
    "gene therapy",
    "editing efficiency",
    "HDR",
    "NHEJ",
]


@dataclass(slots=True)
class LiteratureAcquisitionRequest:
    original_request: str
    intent: str = "acquisition"
    topic: str = ""
    topic_terms: list[str] = field(default_factory=list)
    corpus_mode: str = "broad"
    query_profile: str = "llm_generated"
    primary_query: str = ""
    expansion_queries: list[str] = field(default_factory=list)
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    all_years: bool = True
    recent_years: int | None = None
    sources: list[str] = field(default_factory=lambda: list(DEFAULT_SOURCES))
    fulltext_formats: list[str] = field(default_factory=lambda: list(LEGAL_FULLTEXT_FORMATS))
    legal_policy: str = "legal_oa_only"
    metadata_only_if_no_fulltext: bool = True
    output_dir: str = ""
    scope_too_broad: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    suggested_profiles: list[dict[str, Any]] = field(default_factory=list)
    risk_level: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "literature"


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _merge_terms(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                merged.append(text)
    return merged


def _requests_gene_editing_exclusion(text: str) -> bool:
    patterns = [
        r"\u975e\u7eaf\u57fa\u56e0\u7f16\u8f91",
        r"\u9664\u7eaf\u57fa\u56e0\u7f16\u8f91",
        r"\u6392\u9664.*\u57fa\u56e0\u7f16\u8f91",
        r"\u4e0d\u8981.*\u57fa\u56e0\u7f16\u8f91",
        r"\u975e\u57fa\u56e0\u7f16\u8f91",
        r"not\s+(pure\s+)?gene\s+editing",
        r"exclude\s+(pure\s+)?gene\s+editing",
        r"excluding\s+(pure\s+)?gene\s+editing",
        r"non[-\s]?gene[-\s]?editing",
    ]
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _year_filter(text: str) -> tuple[int | None, int | None, bool, int | None]:
    if re.search(r"(\u6240\u6709\u5e74\u4efd|\u5168\u90e8\u5e74\u4efd|\u4e0d\u9650\u5e74\u4efd|\u5168\u5e74\u4efd|all\s+years|any\s+year|no\s+year\s+limit)", text, re.I):
        return None, None, True, None
    match = re.search(r"(20\d{2}|19\d{2})\s*[-\u2013\u2014\u81f3\u5230]\s*(20\d{2}|19\d{2})", text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return min(start, end), max(start, end), False, None
    if re.search(r"(\u8fd1\u4e24\u5e74|\u6700\u8fd1\u4e24\u5e74|last\s+two\s+years|past\s+two\s+years)", text, re.I):
        current = date.today().year
        return current - 1, current, False, 2
    since = re.search(r"(?:since|\u81ea|\u4ee5\u6765)\s*(20\d{2}|19\d{2})", text, re.I)
    if since:
        return int(since.group(1)), date.today().year, False, None
    years = [int(item) for item in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if years:
        return min(years), max(years), False, None
    return None, None, True, None


def _materials_suggestions() -> list[dict[str, Any]]:
    return [
        {"profile": "llm_generated", "description": "Use LLM-generated search terms for non-CRISPR topics."},
        {"profile": "crispr_broad", "description": "Use the stable CRISPR broad rule profile for CRISPR topics."},
    ]


def _base_request(request: str) -> LiteratureAcquisitionRequest:
    year_from, year_to, all_years, recent_years = _year_filter(request)
    parsed = LiteratureAcquisitionRequest(original_request=request, year_from=year_from, year_to=year_to, all_years=all_years, recent_years=recent_years)
    if _contains_any(request.lower(), ["\u5168\u6587\u683c\u5f0f\u4e0d\u9650", "\u80fd\u62ff\u5168\u6587\u5c31\u62ff", "fulltext", "full text", "\u683c\u5f0f\u4e0d\u9650"]):
        parsed.fulltext_formats = list(LEGAL_FULLTEXT_FORMATS)
    if _contains_any(request.lower(), ["\u62ff\u4e0d\u5230\u5168\u6587\u4fdd\u7559 metadata", "\u4fdd\u7559 metadata", "metadata-only", "metadata only"]):
        parsed.metadata_only_if_no_fulltext = True
    return parsed


def parse_acquisition_request_rules_fallback(request: str) -> LiteratureAcquisitionRequest:
    text = request.lower()
    parsed = _base_request(request)
    if _contains_any(text, ["\u6709\u591a\u5c11", "\u591a\u5c11\u7bc7", "\u6570\u91cf", "count", "statistics", "summary", "\u7edf\u8ba1", "openalex \u4e0a\u6709\u591a\u5c11", "pubmed \u4e0a\u6709\u591a\u5c11", "\u672c\u5730\u6709\u591a\u5c11"]):
        parsed.intent = "count_summary"
    if "crispr" in text or "cas12" in text or "cas13" in text or "sherlock" in text or "detectr" in text or "holmes" in text:
        parsed.topic = "CRISPR"
        parsed.topic_terms = ["CRISPR", "Cas12", "Cas12a", "Cas13", "Cas13a"]
        parsed.query_profile = "crispr_broad"
        parsed.corpus_mode = "broad"
    else:
        parsed.topic = request.strip()[:120] or "literature"
        parsed.topic_terms = [parsed.topic]
        parsed.query_profile = "llm_generated"
        parsed.needs_clarification = True
        parsed.clarification_question = "This non-CRISPR request needs LLM query generation. Configure an LLM provider or provide an explicit query."
        parsed.suggested_profiles = _materials_suggestions()
    if parsed.intent == "count_summary" and not parsed.topic:
        parsed.topic = "CRISPR" if "crispr" in text else (request.strip()[:120] or "literature")
        parsed.topic_terms = [parsed.topic]
    profile = build_query_profile_from_request(parsed)
    parsed.primary_query = profile["primary_query"]
    parsed.expansion_queries = profile["expansion_queries"]
    parsed.include_terms = profile.get("include_terms", [])
    extra_exclude_terms = GENE_EDITING_EXCLUDE_TERMS if _requests_gene_editing_exclusion(request) else []
    parsed.exclude_terms = _merge_terms(profile.get("exclude_terms", DEFAULT_EXCLUDE_TERMS), extra_exclude_terms)
    parsed.output_dir = parsed.output_dir or _default_output_dir(parsed)
    return parsed


def parse_acquisition_request_with_llm(request: str) -> LiteratureAcquisitionRequest:
    """Parse with an explicit LLM attempt, falling back to deterministic rules."""

    parsed = parse_acquisition_request_rules_fallback(request)
    rule_requires_clarification = parsed.needs_clarification
    rule_intent = parsed.intent
    request_is_crispr = parsed.query_profile == "crispr_broad"
    try:
        from .llm_nodes import build_codex_client

        _config, client = build_codex_client()
        if client is None:
            return parsed
        schema = {
            "intent": parsed.intent,
            "topic": parsed.topic,
            "topic_terms": parsed.topic_terms,
            "query_profile": parsed.query_profile,
            "primary_query": parsed.primary_query,
            "expansion_queries": parsed.expansion_queries[:8],
            "include_terms": parsed.include_terms,
            "exclude_terms": parsed.exclude_terms,
            "scope_too_broad": parsed.scope_too_broad,
            "needs_clarification": parsed.needs_clarification,
            "clarification_question": parsed.clarification_question,
        }
        prompt = (
            "Parse the literature acquisition request into strict JSON. Keep legal_policy=legal_oa_only. "
            "Never authorize paywall/login/cookie/CAPTCHA/VPN/Sci-Hub/LibGen/Z-Library access. "
            f"Rules fallback JSON: {json.dumps(schema, ensure_ascii=False)}\nRequest: {request}"
        )
        content = client.complete(
            [{"role": "system", "content": "Return only a JSON object."}, {"role": "user", "content": prompt}],
            response_format="json_object",
        )
        data = json.loads(content)
        if not isinstance(data, dict):
            return parsed
        for key in ("intent", "topic", "query_profile", "primary_query", "clarification_question"):
            if data.get(key):
                setattr(parsed, key, str(data[key]))
        for key in ("topic_terms", "expansion_queries", "include_terms", "exclude_terms", "suggested_profiles"):
            if isinstance(data.get(key), list):
                setattr(parsed, key, data[key])
        for key in ("scope_too_broad", "needs_clarification"):
            if key in data:
                setattr(parsed, key, bool(data[key]))
        if rule_requires_clarification and request_is_crispr:
            parsed.scope_too_broad = True
            parsed.needs_clarification = True
            parsed.clarification_question = parsed.clarification_question or "The requested topic is too broad. Please choose a narrower profile before collection starts."
            parsed.suggested_profiles = parsed.suggested_profiles or _materials_suggestions()
        if rule_intent == "count_summary":
            parsed.intent = "count_summary"
        if request_is_crispr:
            parsed.query_profile = "crispr_broad"
        else:
            parsed.query_profile = "llm_generated"
            parsed.needs_clarification = bool(parsed.needs_clarification and not (parsed.primary_query or parsed.topic))
            parsed.scope_too_broad = False
            parsed.clarification_question = None if not parsed.needs_clarification else parsed.clarification_question
        llm_exclude_terms = list(parsed.exclude_terms)
        profile = build_query_profile_from_request(parsed)
        parsed.primary_query = profile["primary_query"]
        parsed.expansion_queries = profile["expansion_queries"]
        parsed.include_terms = profile.get("include_terms", [])
        request_exclude_terms = GENE_EDITING_EXCLUDE_TERMS if _requests_gene_editing_exclusion(request) else []
        parsed.exclude_terms = _merge_terms(profile.get("exclude_terms", DEFAULT_EXCLUDE_TERMS), llm_exclude_terms, request_exclude_terms)
        parsed.legal_policy = "legal_oa_only"
        parsed.metadata_only_if_no_fulltext = True
        parsed.output_dir = parsed.output_dir or _default_output_dir(parsed)
        return parsed
    except Exception:
        return parsed


def build_query_profile_from_request(parsed_request: LiteratureAcquisitionRequest) -> dict[str, Any]:
    profile = parsed_request.query_profile
    profiles: dict[str, dict[str, Any]] = {
        "crispr_broad": {
            "primary_query": CRISPR_BROAD_QUERY,
            "expansion_queries": ["CRISPR", "CRISPR-Cas mechanism", "CRISPR structure", "Cas9 structure", "Cas12 structure", "Cas12a structure", "Cas13 structure", "Cas13a structure", "SHERLOCK", "DETECTR", "HOLMES"],
            "include_terms": ["CRISPR", "Cas", "SHERLOCK", "DETECTR", "HOLMES"],
        },
    }
    if profile == "crispr_broad":
        data = dict(profiles["crispr_broad"])
    else:
        primary = parsed_request.primary_query or parsed_request.topic or parsed_request.original_request
        expansion = [query for query in parsed_request.expansion_queries if str(query).strip()]
        if not expansion:
            expansion = [term for term in parsed_request.topic_terms if str(term).strip() and str(term).strip() != primary]
        data = {
            "primary_query": primary,
            "expansion_queries": expansion,
            "include_terms": parsed_request.include_terms or parsed_request.topic_terms,
        }
    queries = []
    seen = set()
    for query in [data["primary_query"], *data.get("expansion_queries", [])]:
        text = str(query).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            queries.append(text)
    data["primary_query"] = queries[0] if queries else parsed_request.original_request
    data["expansion_queries"] = queries[1:]
    data.setdefault("exclude_terms", DEFAULT_EXCLUDE_TERMS)
    return data


def _default_output_dir(parsed: LiteratureAcquisitionRequest) -> str:
    if parsed.needs_clarification and "material" in parsed.topic.lower():
        return "agent_runs/materials_scope_planning"
    years = "all_years" if parsed.all_years else f"{parsed.year_from}_{parsed.year_to}"
    if parsed.query_profile == "crispr_broad":
        return f"agent_runs/crispr_broad_{years}"
    return f"agent_runs/{_slug(parsed.topic)}_{years}"


def write_acquisition_plan(parsed: LiteratureAcquisitionRequest, output_dir: str) -> str:
    path = Path(output_dir) / "acquisition_request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())
