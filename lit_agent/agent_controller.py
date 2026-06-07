"""Legacy rule-based controller loop above the deterministic pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .agent_task import LiteratureTask
from .pipeline import PipelineOptions, run_pipeline


@dataclass(slots=True)
class ControllerOptions:
    output_dir: str = "agent_runs"
    allow_network_metadata: bool = False
    allow_network_rank: bool = False
    plan_downloads: bool = True
    download: bool = False
    allow_download: bool = False
    yes: bool = False
    max_results: int | None = None
    max_downloads: int | None = None
    max_attempts: int = 2
    retry_multiplier: int = 3

    def to_pipeline_options(self) -> PipelineOptions:
        return PipelineOptions(
            output_dir=self.output_dir,
            allow_network_metadata=self.allow_network_metadata,
            allow_network_rank=self.allow_network_rank,
            plan_downloads=self.plan_downloads,
            download=self.download,
            allow_download=self.allow_download,
            yes=self.yes,
            max_results=self.max_results,
            max_downloads=self.max_downloads,
        )


def _last_step(summary: dict[str, Any], step_name: str) -> dict[str, Any]:
    for step in reversed(summary.get("steps") or []):
        if step.get("step") == step_name:
            payload = step.get("summary")
            return payload if isinstance(payload, dict) else {}
    return {}


def _diagnose(summary: dict[str, Any], options: ControllerOptions) -> dict[str, Any]:
    metadata = _last_step(summary, "metadata_rank")
    legality = _last_step(summary, "legality_check")
    plan = _last_step(summary, "download_plan")
    download = _last_step(summary, "download")

    metadata_written = int(metadata.get("metadata_results_written") or 0)
    topic_removed = int(metadata.get("topic_guard_removed_count") or 0)
    rank_removed = int(metadata.get("rank_removed_count") or 0)
    rank_input = int(metadata.get("rank_input_count") or 0)
    legality_allowed = int(legality.get("allowed_for_future_download") or 0)
    planned = int(plan.get("planned_downloads") or 0)
    actual_downloads = int(download.get("actual_downloads") or 0)

    reasons: list[str] = []
    actions: list[str] = []
    retryable = False

    if actual_downloads > 0:
        reasons.append("At least one legal OA PDF was downloaded.")
    elif planned > 0:
        reasons.append("Legal OA download candidates were planned, but real download was not executed or did not complete.")
        if options.download and not options.allow_download:
            actions.append("Use --allow-download after reviewing the plan.")
    elif legality_allowed > 0:
        reasons.append("Legal OA records exist, but no download plan was produced.")
        actions.append("Run the download-planning step again.")
        retryable = True
    elif metadata_written == 0:
        if not options.allow_network_metadata:
            reasons.append("No metadata was written because network metadata lookup is disabled or only dry-run mock data was available.")
            actions.append("Rerun with --allow-network-metadata.")
        elif topic_removed > 0:
            reasons.append("Records were found but removed by the CRISPR detection topic guard.")
            actions.append("Keep the topic guard; increase max_results or inspect metadata rather than broadening to generic detection.")
            retryable = True
        elif rank_input > 0 and rank_removed >= rank_input:
            reasons.append("All records were removed by journal rank or impact-factor filtering.")
            if not options.allow_network_rank:
                actions.append("Rerun with --allow-network-rank so missing LetPub cache entries can be filled.")
            else:
                actions.append("Inspect journal_rank_cache.jsonl and consider whether the IF/JCR filter is too strict.")
            retryable = True
        else:
            reasons.append("No matching metadata survived the current search.")
            actions.append("Retry with a larger max_results while keeping CRISPR-specific keywords.")
            retryable = True
    else:
        reasons.append("Metadata was found, but no legal OA downloadable PDF was confirmed.")
        actions.append("Inspect candidates and failures; retry with more results or additional legal metadata sources.")
        retryable = True

    return {
        "metadata_results_written": metadata_written,
        "topic_guard_removed_count": topic_removed,
        "rank_removed_count": rank_removed,
        "rank_input_count": rank_input,
        "allowed_for_future_download": legality_allowed,
        "planned_downloads": planned,
        "actual_downloads": actual_downloads,
        "reasons": reasons,
        "recommended_actions": actions,
        "retryable": retryable,
    }


def _compact_step(summary: dict[str, Any], step_name: str) -> dict[str, Any]:
    step = _last_step(summary, step_name)
    keys = [
        "metadata_results_written",
        "candidates_written",
        "legality_checked",
        "allowed_for_future_download",
        "candidate_needs_confirmation",
        "oa_checked",
        "confirmed_oa_candidates",
        "missing_doi",
        "no_oa",
        "planned_downloads",
        "actual_downloads",
        "topic_guard_enabled",
        "topic_guard_removed_count",
        "candidate_topic_guard_removed_count",
        "oa_topic_guard_removed_count",
        "rank_check",
        "rank_input_count",
        "rank_removed_count",
        "rank_filtered_count",
        "network_calls",
        "downloads",
        "status",
    ]
    return {key: step.get(key) for key in keys if key in step}


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "request": summary.get("request"),
        "config_path": summary.get("config_path"),
        "queries_path": summary.get("queries_path"),
        "dry_run": summary.get("dry_run"),
        "allow_network_metadata": summary.get("allow_network_metadata"),
        "allow_network_rank": summary.get("allow_network_rank"),
        "download_requested": summary.get("download_requested"),
        "download_executed": summary.get("download_executed"),
        "report_en": summary.get("report_en"),
        "report_zh": summary.get("report_zh"),
        "steps": {
            "metadata_rank": _compact_step(summary, "metadata_rank"),
            "oa_check": _compact_step(summary, "oa_check"),
            "legality_check": _compact_step(summary, "legality_check"),
            "download_plan": _compact_step(summary, "download_plan"),
            "download": _compact_step(summary, "download"),
        },
    }


def _rule_decision(diagnosis: dict[str, Any]) -> dict[str, Any]:
    action = "retry" if diagnosis.get("retryable") else "stop"
    return {"source": "rules", "action": action, "reason": "legacy_rule_based_controller"}


def _next_max_results(current: int | None, task: LiteratureTask, options: ControllerOptions) -> int:
    base = current or options.max_results or task.max_results or 20
    return min(max(base + 1, base * max(2, options.retry_multiplier)), 200)


def run_agent_controller(
    task: LiteratureTask,
    *,
    request_text: str,
    options: ControllerOptions,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    _ = llm_client
    attempts: list[dict[str, Any]] = []
    working_options = ControllerOptions(**asdict(options))
    max_attempts = max(1, int(options.max_attempts or 1))

    for attempt_index in range(1, max_attempts + 1):
        summary = run_pipeline(task, request_text=request_text, options=working_options.to_pipeline_options())
        diagnosis = _diagnose(summary, working_options)
        decision = _rule_decision(diagnosis)
        attempts.append(
            {
                "attempt": attempt_index,
                "max_results": working_options.max_results or task.max_results,
                "summary": _compact_summary(summary),
                "diagnosis": diagnosis,
                "decision": decision,
            }
        )
        if attempt_index >= max_attempts:
            break
        if decision.get("action") not in {"retry", "retry_more_results"}:
            break
        if not diagnosis.get("retryable"):
            break
        working_options.max_results = _next_max_results(working_options.max_results, task, options)

    final = attempts[-1] if attempts else {}
    return {
        "request": request_text,
        "controller": "legacy_rule_based",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "final_diagnosis": final.get("diagnosis", {}),
        "final_summary": final.get("summary", {}),
    }
