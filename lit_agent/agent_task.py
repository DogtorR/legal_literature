"""Task parsing and config generation for conversational literature requests."""

from __future__ import annotations

import json
import re
from calendar import monthrange
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .llm_config import AgentConfig
from .topic_guard import CRISPR_TERMS, DETECTION_TERMS


CRISPR_SEARCH_KEYWORDS = [
    "CRISPR detection",
    "CRISPR diagnostics",
    "CRISPR-based detection",
    "CRISPR-based diagnostics",
    "CRISPR biosensing",
    "CRISPR sensor",
    "CRISPR assay",
    "Cas12 detection",
    "Cas12a detection",
    "Cas13 detection",
    "Cas13a detection",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
    "CRISPR nucleic acid detection",
    "CRISPR molecular diagnostics",
    "CRISPR point-of-care detection",
]

DNA_NANOSTRUCTURE_SEARCH_KEYWORDS = ["DNA"]
DNA_NANOSTRUCTURE_TERMS = [
    "DNA nanostructure",
    "DNA nanostructures",
    "DNA nanotechnology",
    "DNA origami",
    "DNA nanoarchitecture",
    "DNA nanoarchitectures",
    "DNA nanodevice",
    "DNA nanodevices",
    "DNA nanoparticle",
    "DNA nanoparticles",
    "DNA framework",
    "DNA frameworks",
    "DNA assembly",
    "DNA assemblies",
    "DNA tile",
    "DNA tiles",
]
DNA_NANOSTRUCTURE_NATURE_QUERY_JOURNALS = [
    "Nature",
    "Nature Communications",
    "Nature Nanotechnology",
    "Nature Materials",
    "Nature Chemistry",
]

NATURE_JOURNALS = [
    "Nature",
    "Nature Biotechnology",
    "Nature Biomedical Engineering",
    "Nature Chemistry",
    "Nature Chemical Biology",
    "Nature Communications",
    "Nature Materials",
    "Nature Methods",
    "Nature Nanotechnology",
    "Nature Medicine",
    "Nature Protocols",
    "Nature Reviews Methods Primers",
    "Nature Structural & Molecular Biology",
    "Nature Synthesis",
    "Communications Biology",
    "Communications Chemistry",
    "Communications Materials",
    "Scientific Reports",
]
KNOWN_EXPLICIT_JOURNALS = ["Nature Communications", "Science Advances"]
MONTH_NAME_TO_NUMBER = {
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


@dataclass(slots=True)
class LiteratureTask:
    original_request: str
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=lambda: ["editorial", "comment", "letter"])
    year_from: int | None = None
    year_to: int | None = None
    publication_date_from: str = ""
    publication_date_to: str = ""
    journal_filter_mode: str = "none"
    journal_names: list[str] = field(default_factory=list)
    require_jcr_q1: bool = False
    min_impact_factor: float | None = None
    topic_guard: dict[str, Any] = field(default_factory=dict)
    require_legal_oa: bool = True
    allow_download: bool = False
    allow_network_metadata: bool = False
    max_results: int = 20
    max_downloads: int = 20
    sources: list[str] = field(default_factory=lambda: ["openalex", "crossref", "europe_pmc", "pubmed", "unpaywall"])
    needs_user_confirmation: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_years(text: str) -> tuple[int | None, int | None]:
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", text)]
    if not years:
        return None, None
    return min(years), max(years)


def _extract_month_range(text: str) -> tuple[str, str]:
    current_year = re.search(r"(?:\u4eca\u5e74|this\s+year)\s*(1[0-2]|[1-9])\s*(?:\u6708)?", text, re.I)
    if current_year:
        year, month = date.today().year, int(current_year.group(1))
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{monthrange(year, month)[1]:02d}"
    chinese = re.search(r"(?<!\d)(20\d{2}|19\d{2})\s*年\s*(1[0-2]|[1-9])\s*月", text)
    if chinese:
        year, month = int(chinese.group(1)), int(chinese.group(2))
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{monthrange(year, month)[1]:02d}"
    iso = re.search(r"(?<!\d)(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])(?!\d)", text)
    if iso:
        year, month = int(iso.group(1)), int(iso.group(2))
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{monthrange(year, month)[1]:02d}"
    english = re.search(r"\b([A-Za-z]{3,9})\s+(20\d{2}|19\d{2})\b|\b(20\d{2}|19\d{2})\s+([A-Za-z]{3,9})\b", text)
    if english:
        month_text = (english.group(1) or english.group(4) or "").lower()
        year_text = english.group(2) or english.group(3)
        month = MONTH_NAME_TO_NUMBER.get(month_text)
        if month and year_text:
            year = int(year_text)
            return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{monthrange(year, month)[1]:02d}"
    return "", ""


def _extract_min_impact_factor(text: str) -> float | None:
    patterns = [
        r"(?:IF|impact\s*factor)\s*(?:>=|>|above|over)\s*(\d+(?:\.\d+)?)",
        r"(?:影响因子)\s*(?:大于|超过|高于|>=|>)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:or\s+above|and\s+above)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return None


def _is_crispr_request(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CRISPR_TERMS)


def _is_dna_nanostructure_request(text: str) -> bool:
    lowered = text.lower()
    has_dna = "dna" in lowered
    has_nano = any(
        term in lowered
        for term in (
            "nanostructure",
            "nanostructures",
            "nanotechnology",
            "origami",
            "nanodevice",
            "nanoarchitecture",
            "纳米",
            "納米",
        )
    )
    return has_dna and has_nano


def _parsed_dna_nanostructure_request(data: dict[str, Any]) -> bool:
    text = json.dumps(
        {
            "topic": data.get("topic"),
            "keywords": data.get("keywords"),
            "include_terms": data.get("include_terms"),
            "notes": data.get("notes"),
        },
        ensure_ascii=False,
    )
    return _is_dna_nanostructure_request(text)


def _mentions_download(text: str) -> bool:
    return "下载" in text or "download" in text.lower()


def _mentions_q1(text: str) -> bool:
    lowered = text.lower()
    return "q1" in lowered or "一区" in text or "一區" in text


def _explicit_journals(text: str) -> list[str]:
    lowered = text.lower()
    return [journal for journal in KNOWN_EXPLICIT_JOURNALS if journal.lower() in lowered]


def _fallback_parse(user_text: str, agent_config: AgentConfig) -> LiteratureTask:
    lowered = user_text.lower()
    year_from, year_to = _extract_years(user_text)
    date_from, date_to = _extract_month_range(user_text)
    task = LiteratureTask(
        original_request=user_text,
        year_from=year_from or agent_config.default_year_from,
        year_to=year_to or agent_config.default_year_to,
        publication_date_from=date_from,
        publication_date_to=date_to,
        min_impact_factor=_extract_min_impact_factor(user_text),
        max_results=agent_config.max_results,
        max_downloads=agent_config.max_downloads,
        sources=list(agent_config.default_sources or []),
        needs_user_confirmation=agent_config.require_download_confirmation,
    )
    if _is_crispr_request(user_text):
        task.topic = "CRISPR detection"
        task.keywords = CRISPR_SEARCH_KEYWORDS
        task.include_terms = []
        task.topic_guard = {
            "enabled": True,
            "type": "crispr_detection",
            "primary_terms": list(CRISPR_TERMS),
            "secondary_terms": list(DETECTION_TERMS),
        }
        task.notes.append("Topic guard enabled: records must match both CRISPR-related and detection/diagnostic-related terms.")
    elif _is_dna_nanostructure_request(user_text):
        task.topic = "DNA nanostructures"
        task.keywords = DNA_NANOSTRUCTURE_SEARCH_KEYWORDS
        task.include_terms = DNA_NANOSTRUCTURE_TERMS
        task.sources = [source for source in task.sources if source != "crossref"]
        task.notes.append("DNA nanostructure search expanded with DNA origami, nanotechnology, nanodevice, and assembly terms.")
    else:
        task.topic = user_text.strip()
        task.keywords = [user_text.strip()]
    explicit_journals = _explicit_journals(user_text)
    if explicit_journals:
        task.journal_filter_mode = "explicit_journals"
        task.journal_names = explicit_journals
        task.notes.append(f"Strict journal filter enabled: {', '.join(explicit_journals)}.")
    elif "nature" in lowered:
        task.journal_filter_mode = "journal_family"
        task.journal_names = NATURE_JOURNALS
    if date_from and date_to:
        task.year_from = int(date_from[:4])
        task.year_to = int(date_to[:4])
        task.notes.append(f"Strict publication date filter enabled: {date_from} to {date_to}.")
    if task.min_impact_factor is not None:
        task.journal_filter_mode = "impact_factor"
        task.notes.append(f"Journal impact factor threshold will use the local LetPub cache first: IF >= {task.min_impact_factor:g}.")
    if "jcr" in lowered and _mentions_q1(user_text):
        task.require_jcr_q1 = True
        task.journal_filter_mode = "jcr_q1"
        task.notes.append("JCR Q1 requires a trusted rank source such as LetPub cache or a user-provided table.")
    if _mentions_download(user_text):
        task.allow_download = False
        task.notes.append("Real PDF download remains disabled until the user confirms after OA/legal planning.")
    return task


def parse_task(user_text: str, agent_config: AgentConfig, llm_client: Any | None = None) -> LiteratureTask:
    """Parse a request with deterministic rules only.

    The ``llm_client`` argument is retained for compatibility with older callers,
    but it is intentionally ignored. LLM-based request understanding now lives in
    ``lit_agent.llm_nodes`` and is only called by explicit graph nodes.
    """

    _ = llm_client
    return _fallback_parse(user_text, agent_config)


def write_task_config(task: LiteratureTask, output_dir: str | Path = "agent_runs") -> tuple[Path, Path]:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task.topic or "literature_task").strip("_")[:60] or "literature_task"
    config_path = run_dir / f"{slug}_search_config.yaml"
    queries_path = run_dir / f"{slug}_queries.csv"
    query_id = slug.lower()

    query_journals = task.journal_names if task.journal_filter_mode in {"explicit_journals", "journal_family"} else [""]
    if task.topic == "DNA nanostructures" and task.journal_filter_mode == "journal_family":
        query_journals = [journal for journal in DNA_NANOSTRUCTURE_NATURE_QUERY_JOURNALS if journal in task.journal_names]
    query_journals = [journal for journal in query_journals if journal] or [""]
    query_lines = [
        "input_id,keywords,year_from,year_to,publication_date_from,publication_date_to,journal,include_terms,exclude_terms,max_results"
    ]
    for idx, journal in enumerate(query_journals, 1):
        row_id = query_id if len(query_journals) == 1 else f"{query_id}_{idx}"
        query_lines.append(
            f"{row_id},\"{';'.join(task.keywords)}\",{task.year_from or ''},{task.year_to or ''},"
            f"{task.publication_date_from},{task.publication_date_to},\"{journal}\","
            f"\"{';'.join(task.include_terms)}\",\"{';'.join(task.exclude_terms)}\",{task.max_results}"
        )
    queries_path.write_text("\n".join(query_lines) + "\n", encoding="utf-8")
    config_journals = task.journal_names if task.journal_filter_mode == "explicit_journals" else []
    config = {
        "project_name": "legal_literature_agent",
        "query_name": query_id,
        "queries_file": queries_path.name,
        "keywords": task.keywords,
        "year_from": task.year_from,
        "year_to": task.year_to,
        "publication_date_from": task.publication_date_from,
        "publication_date_to": task.publication_date_to,
        "include_terms": task.include_terms,
        "exclude_terms": task.exclude_terms,
        "journals": config_journals,
        "max_results_per_source": task.max_results,
        "max_downloads": task.max_downloads,
        "dry_run": True,
        "sources": task.sources,
        "topic_guard": task.topic_guard,
        "record_scope": {
            "journals": task.journal_names if task.journal_filter_mode in {"explicit_journals", "journal_family"} else [],
            "journal_family": "nature" if task.journal_filter_mode == "journal_family" else "",
            "publication_date_from": task.publication_date_from,
            "publication_date_to": task.publication_date_to,
            "require_journal_match": task.journal_filter_mode == "explicit_journals",
            "require_journal_family_match": task.journal_filter_mode == "journal_family",
            "require_date_match": bool(task.publication_date_from and task.publication_date_to),
        },
        "download_policy": {
            "only_legal_open_access": True,
            "uncertain_to_candidates": True,
            "require_license_or_oa_evidence": True,
            "downloads_enabled": False,
        },
        "journal_rank_policy": {
            "source": "letpub",
            "cache_path": "journal_rank_cache.jsonl",
            "min_impact_factor": task.min_impact_factor,
            "require_jcr_q1": task.require_jcr_q1,
            "allow_network_rank": False,
        },
        "agent_task": task.to_dict(),
    }
    try:
        import yaml  # type: ignore
    except Exception:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path, queries_path
