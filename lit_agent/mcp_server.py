"""MCP server wrapper for the legal literature agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP

from .agent_controller import ControllerOptions, run_agent_controller
from .agent_graph import run_crispr_detection_corpus_agent as _run_crispr_detection_corpus_agent
from .agent_graph import run_crispr_broad_corpus_agent as _run_crispr_broad_corpus_agent
from .agent_graph import run_crispr_broad_full_collection as _run_crispr_broad_full_collection
from .agent_graph import run_crispr_broad_recall_calibration as _run_crispr_broad_recall_calibration
from .agent_graph import run_literature_graph_agent
from .agent_graph import run_literature_acquisition_agent as _run_literature_acquisition_agent
from .agent_task import LiteratureTask, parse_task, write_task_config
from .crispr_scope import BROAD_CRISPR_CORPUS_QUERIES
from .count_summary import debug_openalex_count_query as _debug_openalex_count_query
from .count_summary import summarize_literature_counts as _summarize_literature_counts
from .downloader import execute_download_plan, plan_downloads_from_legality_audit
from .llm_config import AgentConfig
from .manifest import read_jsonl
from .pipeline import PipelineOptions, run_pipeline
from . import tool_registry as tr


ARTIFACTS = {
    "metadata_results": Path("metadata_results.jsonl"),
    "candidates": Path("candidates.jsonl"),
    "legality_audit": Path("legality_audit.jsonl"),
    "download_plan": Path("download_plan.jsonl"),
    "manifest": Path("manifest.jsonl"),
    "failures": Path("failures.jsonl"),
    "search_log": Path("search_log.jsonl"),
    "report_en": Path("agent_runs/reports/report_en.md"),
    "report_zh": Path("agent_runs/reports/report_zh.md"),
}
RUNTIME_JSONL_ARTIFACTS = [
    "metadata_results",
    "candidates",
    "legality_audit",
    "download_plan",
    "manifest",
    "failures",
    "search_log",
]


mcp = FastMCP(
    "Legal Literature Agent",
    instructions=(
        "This server exposes both low-level literature tools and the high-level LangGraph agent. "
        "The high-level agent uses explicit LangGraph LLM nodes for request understanding and diagnosis. "
        "Never bypass paywalls, login, VPN, cookies, CAPTCHA, Sci-Hub, LibGen, or Z-Library. "
        "All downloads must be legal OA only."
    ),
    json_response=True,
)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_request(
    request: str,
    *,
    provider: str = "",
    llm_config: str = "",
    env_file: str = ".env",
    no_llm: bool = False,
    max_results: int | None = None,
    max_downloads: int | None = None,
) -> LiteratureTask:
    _ = (provider, llm_config, env_file, no_llm)
    task = parse_task(request, AgentConfig())
    if max_results:
        task.max_results = max_results
    if max_downloads:
        task.max_downloads = max_downloads
    return task


def _cleanup_runtime_artifacts() -> list[str]:
    removed: list[str] = []
    for name in RUNTIME_JSONL_ARTIFACTS:
        path = ARTIFACTS[name]
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def _artifact_path(name: str) -> Path:
    key = name.strip().lower().replace(".jsonl", "")
    if key not in ARTIFACTS:
        allowed = ", ".join(sorted(ARTIFACTS))
        raise ValueError(f"Unknown artifact {name!r}. Allowed: {allowed}")
    return ARTIFACTS[key]


def _tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _download_files() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    downloads = Path("downloads")
    if not downloads.exists():
        return output
    for path in sorted(downloads.glob("*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        output.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return output


def _artifact_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, path in ARTIFACTS.items():
        if path.suffix == ".jsonl":
            counts[name] = len(read_jsonl(path))
        else:
            counts[name] = 1 if path.exists() else 0
    return counts


@mcp.tool()
def search_openalex(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, allow_network: bool = False) -> dict[str, Any]:
    """Search OpenAlex metadata through the standardized tool registry."""

    return tr.search_openalex_tool(query, year_from=year_from, year_to=year_to, max_results=max_results, allow_network=allow_network)


@mcp.tool()
def search_pubmed(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, allow_network: bool = False) -> dict[str, Any]:
    """Search PubMed metadata through the standardized tool registry."""

    return tr.search_pubmed_tool(query, year_from=year_from, year_to=year_to, max_results=max_results, allow_network=allow_network)


@mcp.tool()
def search_crossref(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, allow_network: bool = False) -> dict[str, Any]:
    """Search Crossref metadata through the standardized tool registry."""

    return tr.search_crossref_tool(query, year_from=year_from, year_to=year_to, max_results=max_results, allow_network=allow_network)


@mcp.tool()
def search_europe_pmc(query: str, year_from: int | None = None, year_to: int | None = None, max_results: int = 20, allow_network: bool = False) -> dict[str, Any]:
    """Search Europe PMC metadata through the standardized tool registry."""

    return tr.search_europe_pmc_tool(query, year_from=year_from, year_to=year_to, max_results=max_results, allow_network=allow_network)


@mcp.tool()
def check_unpaywall(doi: str, allow_network: bool = False) -> dict[str, Any]:
    """Check Unpaywall OA evidence through the standardized tool registry."""

    return tr.check_unpaywall_tool(doi, allow_network=allow_network)


@mcp.tool()
def check_europe_pmc_oa(record: dict[str, Any]) -> dict[str, Any]:
    """Check Europe PMC OA evidence through the standardized tool registry."""

    return tr.check_europe_pmc_oa_tool(record)


@mcp.tool()
def plan_legal_downloads(records: list[dict[str, Any]], max_downloads: int = 5) -> dict[str, Any]:
    """Plan legal OA downloads through the standardized tool registry."""

    return tr.plan_legal_downloads_tool(records, max_downloads=max_downloads)


@mcp.tool()
def download_legal_pdf(download_request: dict[str, Any], allow_download: bool = False, yes: bool = False, output_dir: str = "downloads") -> dict[str, Any]:
    """Download one confirmed legal OA PDF through the standardized tool registry."""

    return tr.download_legal_pdf_tool(download_request, allow_download=allow_download, yes=yes, output_dir=output_dir)


@mcp.tool()
def write_manifest(records: list[dict[str, Any]] | dict[str, Any], path: str = "manifest.jsonl") -> dict[str, Any]:
    """Write manifest records through the standardized tool registry."""

    return tr.write_manifest_tool(records, path=path)


@mcp.tool()
def write_report(summary: dict[str, Any], output_dir: str = "agent_runs") -> dict[str, Any]:
    """Write Markdown reports through the standardized tool registry."""

    return tr.write_report_tool(summary, output_dir=output_dir)


@mcp.tool()
def run_literature_graph_agent_tool(
    request: str,
    allow_network_metadata: bool = True,
    allow_network_rank: bool = True,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results: int = 30,
    max_downloads: int = 5,
    use_llm_diagnosis: bool = True,
    output_dir: str = "agent_runs",
) -> dict[str, Any]:
    """Run the high-level LangGraph literature agent."""

    return run_literature_graph_agent(
        request,
        allow_network_metadata=allow_network_metadata,
        allow_network_rank=allow_network_rank,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results=max_results,
        max_downloads=max_downloads,
        use_llm_diagnosis=use_llm_diagnosis,
        output_dir=output_dir,
    )


@mcp.tool()
def run_literature_acquisition_agent(
    request: str,
    allow_download: bool = False,
    yes: bool = False,
    output_dir: str | None = None,
    max_rounds: int = 20,
    max_batches_per_round: int = 12,
    max_results_per_batch: int = 300,
    max_additional_results_per_batch: int = 300,
    time_budget_seconds: int | None = 3300,
    graceful_stop_buffer_seconds: int = 180,
) -> dict[str, Any]:
    """Run the general natural-language legal literature acquisition agent."""

    return _run_literature_acquisition_agent(
        request=request,
        allow_download=allow_download,
        yes=yes,
        output_dir=output_dir,
        max_rounds=max_rounds,
        max_batches_per_round=max_batches_per_round,
        max_results_per_batch=max_results_per_batch,
        max_additional_results_per_batch=max_additional_results_per_batch,
        time_budget_seconds=time_budget_seconds,
        graceful_stop_buffer_seconds=graceful_stop_buffer_seconds,
    )


@mcp.tool()
def summarize_literature_counts(
    topic: str,
    query: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    output_dir: str | None = None,
    include_openalex: bool = True,
    include_pubmed: bool = True,
    include_local: bool = True,
) -> dict[str, Any]:
    """Return OpenAlex, PubMed, and local corpus counts without collection or downloads."""

    return _summarize_literature_counts(
        topic=topic,
        query=query,
        year_from=year_from,
        year_to=year_to,
        output_dir=output_dir,
        include_openalex=include_openalex,
        include_pubmed=include_pubmed,
        include_local=include_local,
    )


@mcp.tool()
def debug_openalex_count_query(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    mode: str = "title_and_abstract_filter",
    sort: str | None = None,
) -> dict[str, Any]:
    """Debug OpenAlex count API params and counts without downloading full text."""

    return _debug_openalex_count_query(query=query, year_from=year_from, year_to=year_to, mode=mode, sort=sort)


@mcp.tool()
def run_crispr_detection_corpus_agent(
    year_from: int,
    year_to: int,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results_per_query: int = 200,
    max_downloads: int = 1000,
    exhaustive_mode: bool = True,
    include_reviews: bool = False,
    google_scholar_reference_count: int = 23200,
    pubmed_reference_count: int = 2700,
    output_dir: str = "agent_runs/crispr_detection_corpus",
) -> dict[str, Any]:
    """Collect a near-exhaustive CRISPR detection corpus with legal OA-only downloads."""

    return _run_crispr_detection_corpus_agent(
        year_from=year_from,
        year_to=year_to,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results_per_query=max_results_per_query,
        max_downloads=max_downloads,
        exhaustive_mode=exhaustive_mode,
        include_reviews=include_reviews,
        google_scholar_reference_count=google_scholar_reference_count,
        pubmed_reference_count=pubmed_reference_count,
        output_dir=output_dir,
    )


@mcp.tool()
def run_crispr_broad_corpus_agent(
    year_from: int,
    year_to: int,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results_per_query: int = 5000,
    max_downloads: int = 2000,
    exhaustive_mode: bool = True,
    include_reviews: bool = True,
    conservative_scope_guard: bool = True,
    output_dir: str = "agent_runs/crispr_broad_corpus",
) -> dict[str, Any]:
    """Collect a broad CRISPR corpus while excluding pure editing applications."""

    return _run_crispr_broad_corpus_agent(
        year_from=year_from,
        year_to=year_to,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results_per_query=max_results_per_query,
        max_downloads=max_downloads,
        exhaustive_mode=exhaustive_mode,
        include_reviews=include_reviews,
        conservative_scope_guard=conservative_scope_guard,
        output_dir=output_dir,
    )


@mcp.tool()
def run_crispr_broad_recall_calibration(
    year_from: int,
    year_to: int,
    max_results_per_source: int = 5000,
    output_dir: str = "agent_runs/crispr_broad_recall_calibration",
) -> dict[str, Any]:
    """Run metadata-only broad CRISPR recall calibration without PDF downloads."""

    return _run_crispr_broad_recall_calibration(
        year_from=year_from,
        year_to=year_to,
        max_results_per_source=max_results_per_source,
        output_dir=output_dir,
    )


@mcp.tool()
def run_crispr_broad_batch_metadata_search(
    year_from: int,
    year_to: int,
    max_results_per_batch: int = 500,
    resume: bool = True,
    retry_failed: bool = False,
    output_dir: str = "agent_runs/crispr_broad_batch_metadata_search",
) -> dict[str, Any]:
    """Run queue-style broad CRISPR metadata batches without downloading PDFs."""

    return tr.run_corpus_batch_search_tool(
        queries=BROAD_CRISPR_CORPUS_QUERIES,
        sources=["pubmed", "openalex", "crossref", "europe_pmc"],
        year_from=year_from,
        year_to=year_to,
        max_results_per_batch=max_results_per_batch,
        output_dir=output_dir,
        resume=resume,
        retry_failed=retry_failed,
    )


@mcp.tool()
def continue_crispr_broad_partial_batches(
    output_dir: str = "agent_runs/test_crispr_broad_batch_metadata_2025_2026",
    max_additional_results_per_batch: int = 500,
    max_batches: int | None = 10,
    sources: list[str] | None = None,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    """Continue only partial broad CRISPR metadata batches without downloading PDFs."""

    return tr.continue_partial_batches_tool(
        output_dir=output_dir,
        max_additional_results_per_batch=max_additional_results_per_batch,
        max_batches=max_batches,
        sources=sources,
        queries=queries,
    )


@mcp.tool()
def finalize_crispr_broad_corpus_manifests(
    output_dir: str = "agent_runs/test_crispr_broad_batch_metadata_2025_2026",
) -> dict[str, Any]:
    """Finalize broad CRISPR corpus metadata/full-text layer manifests without downloading PDFs."""

    result = tr.build_final_corpus_manifest_tool(output_dir)
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "actual_downloads": data.get("actual_downloads", 0),
        "corpus_manifest_count": data.get("corpus_manifest_count", 0),
        "metadata_only_manifest_count": data.get("metadata_only_manifest_count", 0),
        "downloaded_pdfs_manifest_count": data.get("downloaded_pdfs_manifest_count", 0),
    }


@mcp.tool()
def run_crispr_broad_full_collection(
    year_from: int | None,
    year_to: int | None,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
    max_results_per_batch: int = 500,
    max_additional_results_per_batch: int = 500,
    max_rounds: int = 20,
    max_batches_per_round: int = 20,
    stop_if_no_growth_rounds: int = 2,
    min_new_unique_records_per_round: int = 10,
    retry_failed: bool = False,
    time_budget_seconds: int | None = 3300,
    graceful_stop_buffer_seconds: int = 180,
    resume: bool = True,
    checkpoint_every_batch: bool = True,
    finalize_on_stop: bool = True,
    output_dir: str = "agent_runs/crispr_broad_full_collection",
) -> dict[str, Any]:
    """Run one-call deterministic broad CRISPR full metadata collection orchestration."""

    return _run_crispr_broad_full_collection(
        year_from=year_from,
        year_to=year_to,
        download=download,
        allow_download=allow_download,
        yes=yes,
        max_results_per_batch=max_results_per_batch,
        max_additional_results_per_batch=max_additional_results_per_batch,
        max_rounds=max_rounds,
        max_batches_per_round=max_batches_per_round,
        stop_if_no_growth_rounds=stop_if_no_growth_rounds,
        min_new_unique_records_per_round=min_new_unique_records_per_round,
        retry_failed=retry_failed,
        time_budget_seconds=time_budget_seconds,
        graceful_stop_buffer_seconds=graceful_stop_buffer_seconds,
        resume=resume,
        checkpoint_every_batch=checkpoint_every_batch,
        finalize_on_stop=finalize_on_stop,
        output_dir=output_dir,
    )


@mcp.tool()
def resume_crispr_broad_full_collection(
    output_dir: str,
    time_budget_seconds: int = 3300,
    graceful_stop_buffer_seconds: int = 180,
    max_rounds: int = 20,
    max_batches_per_round: int = 12,
    max_additional_results_per_batch: int = 300,
    retry_failed: bool = False,
    download: bool = False,
    allow_download: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Resume an existing broad CRISPR full collection output directory without downloading PDFs."""

    _ = (download, allow_download, yes)
    return _run_crispr_broad_full_collection(
        year_from=2025,
        year_to=2026,
        download=False,
        allow_download=False,
        yes=False,
        max_results_per_batch=max_additional_results_per_batch,
        max_additional_results_per_batch=max_additional_results_per_batch,
        max_rounds=max_rounds,
        max_batches_per_round=max_batches_per_round,
        stop_if_no_growth_rounds=3,
        min_new_unique_records_per_round=5,
        retry_failed=retry_failed,
        time_budget_seconds=time_budget_seconds,
        graceful_stop_buffer_seconds=graceful_stop_buffer_seconds,
        resume=True,
        checkpoint_every_batch=True,
        finalize_on_stop=True,
        output_dir=output_dir,
    )


@mcp.tool()
def pre_download_qa_for_legal_oa_candidates(
    output_dir: str = "agent_runs/crispr_broad_full_collection_2025_2026_medium",
) -> dict[str, Any]:
    """Run pre-download QA for legal OA candidates without downloading PDFs."""

    result = tr.pre_download_qa_for_legal_oa_candidates_tool(output_dir)
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "approved_for_download_count": data.get("approved_for_download_count", 0),
        "needs_manual_download_review_count": data.get("needs_manual_download_review_count", 0),
        "rejected_before_download_count": data.get("rejected_before_download_count", 0),
        "actual_downloads": data.get("actual_downloads", 0),
    }


@mcp.tool()
def download_approved_legal_oa_pdfs(
    output_dir: str = "agent_runs/crispr_broad_full_collection_2025_2026_medium",
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    """Download only pre-approved legal OA PDFs after explicit allow_download and yes gates."""

    result = tr.download_approved_legal_oa_pdfs_tool(
        output_dir=output_dir,
        approved_file=approved_file,
        allow_download=allow_download,
        yes=yes,
        max_downloads=max_downloads,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "attempted_downloads": data.get("attempted_downloads", 0),
        "actual_downloads": data.get("actual_downloads", 0),
        "failed_downloads": data.get("failed_downloads", 0),
        "skipped_downloads": data.get("skipped_downloads", 0),
    }


@mcp.tool()
def retry_failed_legal_oa_downloads(
    output_dir: str,
    failed_file: str = "failed_downloads.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_downloads: int | None = None,
) -> dict[str, Any]:
    """Retry only retryable failed legal OA downloads using legal candidate fallback."""

    result = tr.retry_failed_legal_oa_downloads_tool(
        output_dir=output_dir,
        failed_file=failed_file,
        allow_download=allow_download,
        yes=yes,
        max_downloads=max_downloads,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "retry_attempted": data.get("retry_attempted", 0),
        "retry_success": data.get("retry_success", 0),
        "retry_failed": data.get("retry_failed", 0),
        "actual_downloads": data.get("actual_downloads", 0),
    }


@mcp.tool()
def collect_approved_legal_fulltexts(
    output_dir: str = "agent_runs/crispr_broad_full_collection_2025_2026_medium",
    approved_file: str = "approved_for_download.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
    prefer_formats: list[str] | None = None,
) -> dict[str, Any]:
    """Collect approved legal OA full text files without bypassing access controls."""

    result = tr.collect_approved_legal_fulltexts_tool(
        output_dir=output_dir,
        approved_file=approved_file,
        allow_download=allow_download,
        yes=yes,
        max_items=max_items,
        prefer_formats=prefer_formats,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "attempted_items": data.get("attempted_items", 0),
        "actual_fulltexts": data.get("actual_fulltexts", 0),
        "pdf_count": data.get("pdf_count", 0),
        "xml_count": data.get("xml_count", 0),
        "html_count": data.get("html_count", 0),
        "repository_count": data.get("repository_count", 0),
        "preprint_count": data.get("preprint_count", 0),
        "failed_fulltext_fetches": data.get("failed_fulltext_fetches", 0),
        "needs_manual_fulltext_review": data.get("needs_manual_fulltext_review", 0),
        "actual_downloads": data.get("actual_downloads", 0),
    }


@mcp.tool()
def retry_failed_legal_fulltext_fetches(
    output_dir: str,
    failed_file: str = "failed_fulltext_fetches.jsonl",
    allow_download: bool = False,
    yes: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Retry only retryable failed legal OA fulltext fetches without bypassing access controls."""

    result = tr.retry_failed_legal_fulltext_fetches_tool(
        output_dir=output_dir,
        failed_file=failed_file,
        allow_download=allow_download,
        yes=yes,
        max_items=max_items,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "retry_attempted": data.get("retry_attempted", 0),
        "retry_success": data.get("retry_success", 0),
        "retry_failed": data.get("retry_failed", 0),
        "actual_fulltexts": data.get("actual_fulltexts", 0),
        "new_pdf_count": data.get("new_pdf_count", 0),
        "new_xml_count": data.get("new_xml_count", 0),
        "new_html_count": data.get("new_html_count", 0),
        "rejected_candidates": data.get("rejected_candidates", 0),
        "blocked_by_server_count": data.get("blocked_by_server_count", 0),
        "actual_downloads": data.get("actual_downloads", 0),
    }


@mcp.tool()
def calibrate_openalex_oa_counts(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    output_dir: str = "agent_runs/openalex_oa_calibration",
) -> dict[str, Any]:
    """Calibrate OpenAlex OA metadata counts without downloading or saving full text."""

    result = tr.calibrate_openalex_oa_counts_tool(
        query=query,
        year_from=year_from,
        year_to=year_to,
        output_dir=output_dir,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    return {
        **result,
        "total_count": data.get("total_count", 0),
        "oa_count": data.get("oa_count", 0),
        "oa_ratio": data.get("oa_ratio", 0.0),
        "has_oa_url_count": data.get("has_oa_url_count", 0),
        "has_pdf_url_count": data.get("has_pdf_url_count", 0),
        "has_license_count": data.get("has_license_count", 0),
        "actual_downloads": data.get("actual_downloads", 0),
    }


@mcp.tool()
def parse_literature_request(
    request: str,
    provider: str = "",
    no_llm: bool = False,
    max_results: int = 0,
    max_downloads: int = 0,
) -> dict[str, Any]:
    """Parse a natural-language literature request into the agent's structured task."""

    task = _parse_request(
        request,
        provider=provider,
        no_llm=no_llm,
        max_results=max_results or None,
        max_downloads=max_downloads or None,
    )
    config_path, queries_path = write_task_config(task)
    return {
        "task": task.to_dict(),
        "config_path": str(config_path.resolve()),
        "queries_path": str(queries_path.resolve()),
    }


@mcp.tool()
def run_literature_agent(
    request: str,
    provider: str = "",
    no_llm: bool = False,
    allow_network_metadata: bool = True,
    allow_network_rank: bool = True,
    download: bool = False,
    allow_download: bool = False,
    max_results: int = 30,
    max_downloads: int = 5,
    max_attempts: int = 2,
    clean_artifacts: bool = True,
    output_dir: str = "agent_runs",
) -> dict[str, Any]:
    """Run the LLM-assisted controller as an MCP tool and return structured status."""

    if clean_artifacts:
        removed = _cleanup_runtime_artifacts()
    else:
        removed = []
    task = _parse_request(
        request,
        provider=provider,
        no_llm=no_llm,
        max_results=max_results,
        max_downloads=max_downloads,
    )
    summary = run_agent_controller(
        task,
        request_text=request,
        options=ControllerOptions(
            output_dir=output_dir,
            allow_network_metadata=allow_network_metadata,
            allow_network_rank=allow_network_rank,
            plan_downloads=True,
            download=download,
            allow_download=allow_download,
            yes=bool(download and allow_download),
            max_results=max_results,
            max_downloads=max_downloads,
            max_attempts=max_attempts,
        ),
    )
    return {
        "summary": summary,
        "removed_artifacts": removed,
        "artifact_counts": _artifact_counts(),
        "downloaded_pdfs": _download_files(),
    }


@mcp.tool()
def run_literature_pipeline(
    request: str,
    provider: str = "",
    no_llm: bool = False,
    allow_network_metadata: bool = True,
    allow_network_rank: bool = True,
    download: bool = False,
    allow_download: bool = False,
    max_results: int = 30,
    max_downloads: int = 5,
    clean_artifacts: bool = True,
    output_dir: str = "agent_runs",
) -> dict[str, Any]:
    """Run one deterministic pipeline pass without the controller retry loop."""

    if clean_artifacts:
        removed = _cleanup_runtime_artifacts()
    else:
        removed = []
    task = _parse_request(
        request,
        provider=provider,
        no_llm=no_llm,
        max_results=max_results,
        max_downloads=max_downloads,
    )
    summary = run_pipeline(
        task,
        request_text=request,
        options=PipelineOptions(
            output_dir=output_dir,
            allow_network_metadata=allow_network_metadata,
            allow_network_rank=allow_network_rank,
            plan_downloads=True,
            download=download,
            allow_download=allow_download,
            yes=bool(download and allow_download),
            max_results=max_results,
            max_downloads=max_downloads,
        ),
    )
    return {
        "summary": summary,
        "removed_artifacts": removed,
        "artifact_counts": _artifact_counts(),
        "downloaded_pdfs": _download_files(),
    }


@mcp.tool()
def continue_downloads_from_last_audit(
    max_downloads: int = 5,
    allow_download: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Plan or execute downloads from the existing legality_audit.jsonl file."""

    requests = plan_downloads_from_legality_audit(max_downloads=max_downloads)
    if dry_run or not allow_download:
        return {
            "planned": [request.to_dict() for request in requests],
            "executed": False,
            "downloaded_pdfs": _download_files(),
        }
    results = execute_download_plan(requests, dry_run=False, allow_download=True)
    return {
        "planned": [request.to_dict() for request in requests],
        "results": [result.to_dict() for result in results],
        "executed": True,
        "downloaded_pdfs": _download_files(),
    }


@mcp.tool()
def list_literature_agent_outputs() -> dict[str, Any]:
    """List runtime artifact counts and downloaded PDF files."""

    return {
        "artifact_counts": _artifact_counts(),
        "artifacts": {name: str(path.resolve()) for name, path in ARTIFACTS.items() if path.exists()},
        "downloaded_pdfs": _download_files(),
    }


@mcp.tool()
def read_literature_agent_artifact(name: str, max_lines: int = 80) -> str:
    """Read an allowed runtime artifact such as manifest, failures, search_log, or report_zh."""

    return _tail_text(_artifact_path(name), max_lines=max_lines)


@mcp.resource("artifact://{name}")
def artifact_resource(name: str) -> str:
    """Read a runtime artifact by name, for example artifact://manifest."""

    return _tail_text(_artifact_path(name), max_lines=200)


@mcp.resource("downloads://list")
def downloads_resource() -> str:
    """Return downloaded PDF files as JSON."""

    return _json(_download_files())


@mcp.prompt()
def legal_literature_download_prompt(topic: str, year: int = 2026, max_downloads: int = 5) -> str:
    """Create a safe literature-download prompt for this MCP server."""

    return (
        f"Use the Legal Literature Agent MCP tools to find legal open-access papers about {topic} "
        f"published in {year}. Confirm OA evidence through Unpaywall/PMC/Europe PMC or publisher OA pages, "
        f"do not bypass paywalls or authentication, and download at most {max_downloads} PDFs."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCP server for the legal literature agent.")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mount-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport, mount_path=args.mount_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
