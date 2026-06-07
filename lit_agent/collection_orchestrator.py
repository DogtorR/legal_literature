"""Deterministic full-collection scheduler for broad CRISPR corpus batches."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

from .batch_runner import (
    build_corpus_search_batches,
    build_final_corpus_manifests,
    merge_batch_outputs,
    run_search_batch,
)
from .manifest import append_jsonl, read_jsonl
from .metadata import deduplicate_records

SOURCE_PRIORITY = ["pubmed", "openalex", "europe_pmc", "crossref"]
QUERY_PRIORITY = [
    "PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY",
    "CRISPR detection",
    "CRISPR diagnostics",
    "CRISPR structure",
    "CRISPR-Cas mechanism",
    "Cas9 structure",
    "Cas12 structure",
    "Cas12a structure",
    "Cas13 structure",
    "Cas13a structure",
    "Cas12 collateral cleavage",
    "Cas13 collateral cleavage",
    "CRISPR PAM recognition",
    "CRISPR PFS recognition",
    "CRISPR crRNA",
    "CRISPR guide RNA",
    "CRISPR sgRNA",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
]
UNDERCOVERED_PRIORITY_QUERIES = {
    "CRISPR structure",
    "CRISPR-Cas mechanism",
    "Cas9 structure",
    "Cas12 structure",
    "Cas12a structure",
    "Cas13 structure",
    "Cas13a structure",
    "Cas12 collateral cleavage",
    "Cas13 collateral cleavage",
    "CRISPR PAM recognition",
    "CRISPR PFS recognition",
    "CRISPR crRNA",
    "CRISPR guide RNA",
    "CRISPR sgRNA",
}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(path, record)


def _latest_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("batch_id")): row for row in read_jsonl(path)}


def plan_collection_batches(
    query_list: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int,
) -> list[dict[str, Any]]:
    return build_corpus_search_batches(query_list, sources, year_from, year_to, max_results_per_batch)


def _ensure_manifest(output_dir: str, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "batch_manifest.jsonl"
    if not manifest_path.exists() or not read_jsonl(manifest_path):
        for batch in batches:
            batch["output_file"] = str((out / "raw_records_by_batch" / f"{batch['batch_id']}.jsonl").resolve())
        _write_jsonl(manifest_path, batches)
    return list(_latest_manifest(manifest_path).values())


def _query_rank(query: str) -> int:
    if query.strip().startswith("(") and "CRISPR-Cas" in query:
        return 0
    if query in QUERY_PRIORITY:
        return QUERY_PRIORITY.index(query)
    return len(QUERY_PRIORITY) + 1


def _source_rank(source: str) -> int:
    return SOURCE_PRIORITY.index(source) if source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)


def _has_time_remaining(start_time: float | None, time_budget_seconds: int | None, graceful_stop_buffer_seconds: int) -> bool:
    if start_time is None or time_budget_seconds is None:
        return True
    return (time_budget_seconds - (time.monotonic() - start_time)) >= graceful_stop_buffer_seconds


def _selected_reason(batch: dict[str, Any]) -> str:
    parts = [str(batch.get("status") or "pending")]
    if _query_rank(str(batch.get("query") or "")) == 0:
        parts.append("primary_broad_query")
    if str(batch.get("source") or "") in SOURCE_PRIORITY:
        parts.append(f"source_priority_{batch.get('source')}")
    return "|".join(parts)


def select_batches_for_round(
    batch_manifest: list[dict[str, Any]],
    max_batches_per_round: int,
    retry_failed: bool = False,
    strategy: str = "coverage_first",
    per_query_round_cap: int = 4,
) -> list[dict[str, Any]]:
    runnable_statuses = {"pending", "partial"} | ({"failed"} if retry_failed else set())
    candidates = [dict(row) for row in batch_manifest if row.get("status") in runnable_statuses]
    candidates = [row for row in candidates if row.get("status") != "failed" or retry_failed]
    limit = max(0, int(max_batches_per_round))
    if strategy != "coverage_first":
        selected = candidates[:limit]
        for row in selected:
            row["selected_reason"] = _selected_reason(row)
        return selected
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for status in ("pending", "partial", "failed"):
        pool = [row for row in candidates if row.get("status") == status]
        if not pool:
            continue
        for query_rank in sorted({_query_rank(str(row.get("query") or "")) for row in pool}):
            query_pool = [row for row in pool if _query_rank(str(row.get("query") or "")) == query_rank]
            query_selected = 0
            for source in SOURCE_PRIORITY:
                for row in sorted((item for item in query_pool if item.get("source") == source), key=lambda item: str(item.get("batch_id") or "")):
                    if len(selected) >= limit or query_selected >= per_query_round_cap:
                        break
                    batch_id = str(row.get("batch_id"))
                    if batch_id in seen_ids:
                        continue
                    row["selected_reason"] = _selected_reason(row)
                    selected.append(row)
                    seen_ids.add(batch_id)
                    query_selected += 1
                if len(selected) >= limit or query_selected >= per_query_round_cap:
                    break
            for row in sorted((item for item in query_pool if item.get("source") not in SOURCE_PRIORITY), key=lambda item: str(item.get("batch_id") or "")):
                if len(selected) >= limit or query_selected >= per_query_round_cap:
                    break
                batch_id = str(row.get("batch_id"))
                if batch_id in seen_ids:
                    continue
                row["selected_reason"] = _selected_reason(row)
                selected.append(row)
                seen_ids.add(batch_id)
                query_selected += 1
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return selected


def run_batch_scheduler(
    batch_manifest: list[dict[str, Any]],
    *,
    output_dir: str,
    max_batches_per_round: int,
    max_additional_results_per_batch: int,
    retry_failed: bool = False,
    strategy: str = "coverage_first",
    start_time: float | None = None,
    time_budget_seconds: int | None = None,
    graceful_stop_buffer_seconds: int = 180,
    scope_profile: str = "crispr_broad",
    include_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
) -> dict[str, Any]:
    selected = select_batches_for_round(batch_manifest, max_batches_per_round, retry_failed=retry_failed, strategy=strategy)
    results: list[dict[str, Any]] = []
    manifest_path = Path(output_dir) / "batch_manifest.jsonl"
    stop_reason = ""
    for batch in selected:
        if not _has_time_remaining(start_time, time_budget_seconds, graceful_stop_buffer_seconds):
            stop_reason = "time_budget_reached"
            break
        payload = dict(batch)
        resume = payload.get("status") == "partial"
        if resume:
            payload["fetch_limit"] = int(max_additional_results_per_batch)
        result = run_search_batch(payload, output_dir, resume=resume)
        result["selected_reason"] = batch.get("selected_reason", "")
        append_jsonl(manifest_path, result)
        results.append(result)
    merged = merge_batch_outputs(output_dir, scope_profile=scope_profile, include_terms=include_terms, exclude_terms=exclude_terms)
    latest = list(_latest_manifest(Path(output_dir) / "batch_manifest.jsonl").values())
    status_counts = Counter(str(row.get("status")) for row in latest)
    return {
        "status": "ok",
        "scheduled_batches": len(results),
        "scheduler_strategy": strategy,
        "selected_batches": [{"batch_id": row.get("batch_id"), "query": row.get("query"), "source": row.get("source"), "selected_reason": row.get("selected_reason")} for row in selected],
        "batch_results": results,
        "total_batches": len(latest),
        "completed_batches": status_counts.get("completed", 0),
        "partial_batches": status_counts.get("partial", 0),
        "failed_batches": status_counts.get("failed", 0),
        "pending_batches": status_counts.get("pending", 0),
        "merge": merged,
        "batch_manifest": latest,
        "actual_downloads": 0,
        "stop_reason": stop_reason,
        "time_budget_reached": stop_reason == "time_budget_reached",
    }


def check_collection_completion(
    batch_manifest: list[dict[str, Any]],
    *,
    round_index: int = 0,
    low_growth_rounds: int = 0,
    max_rounds: int = 20,
    stop_if_no_growth_rounds: int = 2,
) -> dict[str, Any]:
    total_batches = len(batch_manifest)
    completed_batches = sum(1 for row in batch_manifest if row.get("status") == "completed")
    partial_batches = sum(1 for row in batch_manifest if row.get("status") == "partial")
    failed_batches = sum(1 for row in batch_manifest if row.get("status") == "failed")
    pending_batches = sum(1 for row in batch_manifest if row.get("status") == "pending")
    continuable = [
        row
        for row in batch_manifest
        if row.get("status") == "partial"
        and row.get("has_more")
        and row.get("continuation_supported", True)
        and (row.get("total_count") in (None, "") or int(row.get("retrieved_count") or 0) < int(row.get("total_count") or 0))
    ]
    if round_index >= max_rounds:
        return {"done": True, "should_continue": False, "stop_reason": "max_rounds_reached", "needs_llm_diagnosis": bool(partial_batches or failed_batches)}
    if low_growth_rounds >= stop_if_no_growth_rounds:
        return {"done": True, "should_continue": False, "stop_reason": "low_growth_stop", "needs_llm_diagnosis": bool(partial_batches or failed_batches)}
    if pending_batches:
        return {"done": False, "should_continue": True, "stop_reason": "pending_batches_remaining", "needs_llm_diagnosis": False}
    if continuable:
        return {"done": False, "should_continue": True, "stop_reason": "partial_batches_have_more", "needs_llm_diagnosis": False}
    if partial_batches:
        return {"done": True, "should_continue": False, "stop_reason": "partial_batches_not_continuable", "needs_llm_diagnosis": True}
    if failed_batches:
        return {"done": True, "should_continue": False, "stop_reason": "failed_batches_remaining", "needs_llm_diagnosis": True}
    if completed_batches == total_batches:
        return {"done": True, "should_continue": False, "stop_reason": "all_batches_completed", "needs_llm_diagnosis": False}
    return {"done": True, "should_continue": False, "stop_reason": "no_runnable_batches", "needs_llm_diagnosis": False}


def _raw_unique_count(output_dir: str) -> int:
    raw_dir = Path(output_dir) / "raw_records_by_batch"
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []:
        records.extend(read_jsonl(path))
    return len(deduplicate_records(records))


def _write_full_collection_report(output_dir: str, summary: dict[str, Any], progress: list[dict[str, Any]]) -> str:
    out = Path(output_dir)
    latest = list(_latest_manifest(out / "batch_manifest.jsonl").values())
    per_source = Counter()
    per_query = Counter()
    selected_reasons = Counter()
    pending_by_source = Counter()
    partial_by_source = Counter()
    completed_by_source = Counter()
    pending_by_query = Counter()
    partial_by_query = Counter()
    completed_by_query = Counter()
    unfinished = []
    for row in latest:
        count = int(row.get("retrieved_count") or 0)
        per_source[str(row.get("source"))] += count
        per_query[str(row.get("query"))] += count
        if row.get("selected_reason"):
            selected_reasons[str(row.get("selected_reason"))] += 1
        if row.get("status") == "pending":
            pending_by_source[str(row.get("source"))] += 1
            pending_by_query[str(row.get("query"))] += 1
        elif row.get("status") == "partial":
            partial_by_source[str(row.get("source"))] += 1
            partial_by_query[str(row.get("query"))] += 1
        elif row.get("status") == "completed":
            completed_by_source[str(row.get("source"))] += 1
            completed_by_query[str(row.get("query"))] += 1
        if row.get("status") != "completed":
            unfinished.append(row)
    total_retrieved = sum(per_query.values())
    overrepresented = [query for query, count in per_query.items() if total_retrieved and count / total_retrieved >= 0.25]
    undercovered = [
        query
        for query in UNDERCOVERED_PRIORITY_QUERIES
        if per_query.get(query, 0) < int(summary.get("undercovered_threshold", 25))
        or pending_by_query.get(query, 0)
        or partial_by_query.get(query, 0)
    ]
    query_balance_warning = bool(overrepresented or undercovered)
    lines = [
        "# Full Collection Report",
        "",
        f"- scheduler_strategy: {summary.get('scheduler_strategy', 'coverage_first')}",
        f"- rounds_run: {summary.get('rounds_run', 0)}",
        f"- stop_reason: {summary.get('stop_reason', '')}",
        f"- total_batches: {summary.get('total_batches', 0)}",
        f"- completed_batches: {summary.get('completed_batches', 0)}",
        f"- partial_batches: {summary.get('partial_batches', 0)}",
        f"- failed_batches: {summary.get('failed_batches', 0)}",
        f"- raw_records: {summary.get('total_raw_records', 0)}",
        f"- unique_records: {summary.get('unique_records', 0)}",
        f"- corpus_manifest_records: {summary.get('corpus_manifest_records', 0)}",
        f"- metadata_only_records: {summary.get('metadata_only_records', 0)}",
        f"- downloaded_pdf_records: {summary.get('downloaded_pdf_records', 0)}",
        f"- planned_legal_oa_count: {summary.get('planned_legal_oa_count', 0)}",
        f"- actual_downloads: {summary.get('actual_downloads', 0)}",
        "",
        "## Long Task Handling",
        f"- time_budget_seconds: {summary.get('time_budget_seconds')}",
        f"- graceful_stop_buffer_seconds: {summary.get('graceful_stop_buffer_seconds')}",
        f"- resume_supported: {summary.get('resume_supported', True)}",
        f"- checkpoint_every_batch: {summary.get('checkpoint_every_batch', True)}",
        f"- finalize_on_stop: {summary.get('finalize_on_stop', True)}",
        "",
        "## Current Collection State",
        f"- completed_batches: {summary.get('completed_batches', 0)}",
        f"- partial_batches: {summary.get('partial_batches', 0)}",
        f"- pending_batches: {summary.get('pending_batches', 0)}",
        f"- failed_batches: {summary.get('failed_batches', 0)}",
        f"- can_resume: {bool(summary.get('pending_batches', 0) or summary.get('partial_batches', 0))}",
        f"- next_recommended_action: {'resume_crispr_broad_full_collection' if (summary.get('pending_batches', 0) or summary.get('partial_batches', 0)) else 'final_review'}",
        "- This collection can be resumed safely. Metadata is the primary database layer; legal OA PDFs are an optional full-text layer.",
        "",
        "## Query Balance",
        f"- query_balance_warning: {query_balance_warning}",
        f"- overrepresented_queries: {overrepresented}",
        f"- undercovered_queries: {undercovered}",
        "",
        "## Per Query Corpus Contribution",
        *(f"- {query}: {count}" for query, count in per_query.most_common()),
        "",
        "## Round Growth",
        *(f"- round {row.get('round')}: new_unique_records={row.get('new_unique_records')} unique_records={row.get('unique_records')} stop_reason={row.get('stop_reason', '')}" for row in progress),
        "",
        "## Selected Batches Per Round",
        *(f"- round {row.get('round')}: {row.get('selected_batches', [])}" for row in progress),
        "",
        "## Selected Reason Distribution",
        *(f"- {reason}: {count}" for reason, count in selected_reasons.most_common()),
        "",
        "## Per Source Retrieved Count",
        *(f"- {source}: {count}" for source, count in per_source.most_common()),
        "",
        "## Batch Status By Source",
        "### Pending",
        *(f"- {source}: {count}" for source, count in pending_by_source.most_common()),
        "### Partial",
        *(f"- {source}: {count}" for source, count in partial_by_source.most_common()),
        "### Completed",
        *(f"- {source}: {count}" for source, count in completed_by_source.most_common()),
        "",
        "## Per Query Retrieved Count",
        *(f"- {query}: {count}" for query, count in per_query.most_common()),
        "",
        "## Pending Batches By Query",
        *(f"- {query}: {count}" for query, count in pending_by_query.most_common()),
        "",
        "## Partial Batches By Query",
        *(f"- {query}: {count}" for query, count in partial_by_query.most_common()),
        "",
        "## Completed Batches By Query",
        *(f"- {query}: {count}" for query, count in completed_by_query.most_common()),
        "",
        "## Unfinished Batches",
    ]
    lines.extend(
        f"- {row.get('batch_id')}: status={row.get('status')} has_more={row.get('has_more')} continuation_supported={row.get('continuation_supported')} stop_reason={row.get('stop_reason')} error={row.get('error')}"
        for row in unfinished
    )
    if not unfinished:
        lines.append("- none")
    path = out / "full_collection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def run_full_collection_loop(
    *,
    query_list: list[str],
    sources: list[str],
    year_from: int | None,
    year_to: int | None,
    max_results_per_batch: int = 500,
    max_additional_results_per_batch: int = 500,
    max_rounds: int = 20,
    max_batches_per_round: int = 20,
    stop_if_no_growth_rounds: int = 2,
    min_new_unique_records_per_round: int = 10,
    retry_failed: bool = False,
    scheduler_strategy: str = "coverage_first",
    time_budget_seconds: int | None = 3300,
    graceful_stop_buffer_seconds: int = 180,
    resume: bool = True,
    checkpoint_every_batch: bool = True,
    finalize_on_stop: bool = True,
    scope_profile: str = "crispr_broad",
    include_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    output_dir: str = "agent_runs/crispr_broad_full_collection",
) -> dict[str, Any]:
    start_time = time.monotonic()
    batches = plan_collection_batches(query_list, sources, year_from, year_to, max_results_per_batch)
    manifest_path = Path(output_dir) / "batch_manifest.jsonl"
    resume_detected = bool(resume and manifest_path.exists() and read_jsonl(manifest_path))
    manifest = _ensure_manifest(output_dir, batches)
    progress_path = Path(output_dir) / "full_collection_progress.jsonl"
    previous_progress = read_jsonl(progress_path) if resume and progress_path.exists() else []
    progress: list[dict[str, Any]] = list(previous_progress)
    previous_rounds = len(previous_progress)
    previous_raw_records = sum(1 for path in (Path(output_dir) / "raw_records_by_batch").glob("*.jsonl") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()) if (Path(output_dir) / "raw_records_by_batch").exists() else 0
    previous_unique = _raw_unique_count(output_dir)
    previous_unique_records = previous_unique
    low_growth_rounds = 0
    completion = check_collection_completion(manifest, round_index=0, low_growth_rounds=0, max_rounds=max_rounds, stop_if_no_growth_rounds=stop_if_no_growth_rounds)
    rounds_run_this_call = 0
    forced_stop_reason = ""
    while completion["should_continue"] and rounds_run_this_call < max_rounds:
        if not _has_time_remaining(start_time, time_budget_seconds, graceful_stop_buffer_seconds):
            forced_stop_reason = "time_budget_reached"
            row = {
                "round": previous_rounds + rounds_run_this_call,
                "scheduled_batches": 0,
                "selected_batches": [],
                "new_unique_records": 0,
                "unique_records": previous_unique,
                "low_growth_rounds": low_growth_rounds,
                "stop_reason": forced_stop_reason,
                "should_continue": False,
            }
            append_jsonl(progress_path, row)
            progress.append(row)
            break
        rounds_run_this_call += 1
        cumulative_round = previous_rounds + rounds_run_this_call
        scheduled = run_batch_scheduler(
            manifest,
            output_dir=output_dir,
            max_batches_per_round=max_batches_per_round,
            max_additional_results_per_batch=max_additional_results_per_batch,
            retry_failed=retry_failed,
            strategy=scheduler_strategy,
            start_time=start_time,
            time_budget_seconds=time_budget_seconds,
            graceful_stop_buffer_seconds=graceful_stop_buffer_seconds,
            scope_profile=scope_profile,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )
        if scheduled.get("time_budget_reached"):
            forced_stop_reason = "time_budget_reached"
        current_unique = int((scheduled.get("merge") or {}).get("unique_records") or _raw_unique_count(output_dir))
        new_unique = max(0, current_unique - previous_unique)
        low_growth_rounds = low_growth_rounds + 1 if new_unique < min_new_unique_records_per_round else 0
        manifest = list(scheduled.get("batch_manifest") or _latest_manifest(Path(output_dir) / "batch_manifest.jsonl").values())
        completion = check_collection_completion(
            manifest,
            round_index=rounds_run_this_call,
            low_growth_rounds=low_growth_rounds,
            max_rounds=max_rounds,
            stop_if_no_growth_rounds=stop_if_no_growth_rounds,
        )
        row = {
            "round": cumulative_round,
            "scheduled_batches": scheduled.get("scheduled_batches", 0),
            "selected_batches": scheduled.get("selected_batches", []),
            "new_unique_records": new_unique,
            "unique_records": current_unique,
            "low_growth_rounds": low_growth_rounds,
            "stop_reason": forced_stop_reason or completion.get("stop_reason", ""),
            "should_continue": False if forced_stop_reason else completion.get("should_continue", False),
        }
        append_jsonl(progress_path, row)
        progress.append(row)
        previous_unique = current_unique
        if forced_stop_reason:
            break
    merged = merge_batch_outputs(output_dir, scope_profile=scope_profile, include_terms=include_terms, exclude_terms=exclude_terms)
    final = build_final_corpus_manifests(output_dir) if finalize_on_stop else {"corpus_manifest_count": 0, "metadata_only_manifest_count": 0, "downloaded_pdfs_manifest_count": 0, "planned_legal_oa_count": 0, "artifacts": {}}
    latest = list(_latest_manifest(Path(output_dir) / "batch_manifest.jsonl").values())
    status_counts = Counter(str(row.get("status")) for row in latest)
    stop_reason = forced_stop_reason or completion.get("stop_reason", "")
    summary = {
        "status": "partial" if status_counts.get("partial") or status_counts.get("failed") or status_counts.get("pending") else "ok",
        "rounds_run": previous_rounds + rounds_run_this_call,
        "rounds_run_this_call": rounds_run_this_call,
        "previous_rounds": previous_rounds,
        "stop_reason": stop_reason,
        "total_batches": len(latest),
        "completed_batches": status_counts.get("completed", 0),
        "partial_batches": status_counts.get("partial", 0),
        "failed_batches": status_counts.get("failed", 0),
        "pending_batches": status_counts.get("pending", 0),
        "total_raw_records": merged.get("total_raw_records", 0),
        "unique_records": merged.get("unique_records", 0),
        "corpus_manifest_records": final.get("corpus_manifest_count", 0),
        "metadata_only_records": final.get("metadata_only_manifest_count", 0),
        "downloaded_pdf_records": final.get("downloaded_pdfs_manifest_count", 0),
        "planned_legal_oa_count": final.get("planned_legal_oa_count", 0),
        "actual_downloads": 0,
        "needs_llm_diagnosis": completion.get("needs_llm_diagnosis", False),
        "scheduler_strategy": scheduler_strategy,
        "time_budget_seconds": time_budget_seconds,
        "graceful_stop_buffer_seconds": graceful_stop_buffer_seconds,
        "resume_supported": True,
        "resume_detected": resume_detected,
        "checkpoint_every_batch": checkpoint_every_batch,
        "finalize_on_stop": finalize_on_stop,
        "previous_raw_records": previous_raw_records,
        "previous_unique_records": previous_unique_records,
        "remaining_pending_batches": status_counts.get("pending", 0),
        "remaining_partial_batches": status_counts.get("partial", 0),
        "artifacts": {
            "batch_manifest": str((Path(output_dir) / "batch_manifest.jsonl").resolve()),
            "full_collection_progress": str(progress_path.resolve()),
            **(merged.get("artifacts") or {}),
            **(final.get("artifacts") or {}),
        },
    }
    dataset_card = Path(output_dir) / "dataset_card.md"
    if dataset_card.exists():
        text = dataset_card.read_text(encoding="utf-8", errors="replace").rstrip()
        text += (
            "\n\n## Long Task Handling\n"
            f"- time_budget_seconds: {time_budget_seconds}\n"
            f"- graceful_stop_buffer_seconds: {graceful_stop_buffer_seconds}\n"
            "- resume_supported: True\n"
            f"- checkpoint_every_batch: {checkpoint_every_batch}\n"
            f"- finalize_on_stop: {finalize_on_stop}\n"
            "\n## Current Collection State\n"
            f"- completed_batches: {status_counts.get('completed', 0)}\n"
            f"- partial_batches: {status_counts.get('partial', 0)}\n"
            f"- pending_batches: {status_counts.get('pending', 0)}\n"
            f"- failed_batches: {status_counts.get('failed', 0)}\n"
            f"- can_resume: {bool(status_counts.get('pending', 0) or status_counts.get('partial', 0))}\n"
            "- next_recommended_action: resume_crispr_broad_full_collection\n"
            "\nThis collection can be resumed safely. Metadata is the primary database layer; legal OA PDFs are an optional full-text layer.\n"
        )
        dataset_card.write_text(text + "\n", encoding="utf-8")
    report_path = _write_full_collection_report(output_dir, summary, progress)
    summary["artifacts"]["full_collection_report"] = report_path
    return summary
