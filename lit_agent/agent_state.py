"""Core LangGraph state types for the literature agent."""

from __future__ import annotations

from typing import Any, TypedDict


class LiteratureAgentState(TypedDict, total=False):
    user_request: str
    task: dict[str, Any]
    search_intent: dict[str, Any]
    route: str
    options: dict[str, Any]
    metadata_results: list[dict[str, Any]]
    oa_results: list[dict[str, Any]]
    download_plan: list[dict[str, Any]]
    download_results: list[dict[str, Any]]
    artifacts: dict[str, Any]
    reports: dict[str, Any]
    failures: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    next_action: str
    error: str
    blocked: bool
    finished: bool
    query_list: list[str]
    search_log: list[dict[str, Any]]
    raw_records: list[dict[str, Any]]
    unique_records: list[dict[str, Any]]
    excluded_records: list[dict[str, Any]]
    scope_included_records: list[dict[str, Any]]
    scope_excluded_records: list[dict[str, Any]]
    needs_manual_scope_review: list[dict[str, Any]]
    scope_stats: dict[str, Any]
    topic_records: list[dict[str, Any]]
    oa_audit_records: list[dict[str, Any]]
    corpus_manifest: list[dict[str, Any]]
    coverage: dict[str, Any]
    recall_diagnostics: list[dict[str, Any]]
    manual_reference_counts: dict[str, Any]
    search_attempt_count: int
    oa_attempt_count: int
    max_search_attempts: int
    max_oa_attempts: int
    year_from: int | None
    year_to: int | None
    max_results_per_query: int
    exhaustive_mode: bool
    sources: list[str]
    batch_manifest: list[dict[str, Any]]
    collection_round: int
    low_growth_rounds: int
    previous_unique_records: int
    collection_progress: list[dict[str, Any]]
    collection_completion: dict[str, Any]
    merge: dict[str, Any]
    final_corpus: dict[str, Any]
    acquisition_request: dict[str, Any]
    parsed_request: dict[str, Any]
    collection_result: dict[str, Any]
    pre_download_qa: dict[str, Any]
    fulltext_collection: dict[str, Any]
    summary: dict[str, Any]
    count_summary: dict[str, Any]
