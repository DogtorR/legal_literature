"""One-shot conversational pipeline orchestration."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any

from . import cli
from .agent_task import LiteratureTask, write_task_config
from .report import write_reports


@dataclass(slots=True)
class PipelineOptions:
    output_dir: str = "agent_runs"
    allow_network_metadata: bool = False
    allow_network_rank: bool = False
    plan_downloads: bool = True
    download: bool = False
    allow_download: bool = False
    yes: bool = False
    max_results: int | None = None
    max_downloads: int | None = None


def _run_cli(args: list[str]) -> dict[str, Any]:
    stdout = StringIO()
    with redirect_stdout(stdout):
        exit_code = cli.main(args)
    text = stdout.getvalue().strip()
    if exit_code != 0:
        return {"exit_code": exit_code, "stdout": text}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"exit_code": exit_code, "stdout": text}


def _confirm_download(options: PipelineOptions) -> bool:
    return bool(options.download and options.yes)


def run_pipeline(task: LiteratureTask, *, request_text: str, options: PipelineOptions) -> dict[str, Any]:
    config_path, queries_path = write_task_config(task, options.output_dir)
    common = ["--config", str(config_path), "--metadata-only", "--dry-run"]
    if options.max_results:
        common.extend(["--max-results", str(options.max_results)])
    if options.allow_network_metadata:
        common.append("--allow-network-metadata")
    if options.allow_network_rank:
        common.append("--allow-network-rank")
    if task.min_impact_factor is not None:
        common.extend(["--rank-check", "--min-impact-factor", str(task.min_impact_factor)])
    if task.require_jcr_q1:
        common.extend(["--rank-check", "--require-jcr-q1"])

    steps: list[dict[str, Any]] = []
    metadata_summary = _run_cli(common)
    steps.append({"step": "metadata_rank", "summary": metadata_summary})

    oa_args = ["--config", str(config_path), "--source", "unpaywall", "--oa-check", "--metadata-only", "--dry-run"]
    if options.max_results:
        oa_args.extend(["--max-results", str(options.max_results)])
    if options.allow_network_metadata:
        oa_args.append("--allow-network-metadata")
    oa_summary = _run_cli(oa_args)
    steps.append({"step": "oa_check", "summary": oa_summary})

    legality_args = ["--config", str(config_path), "--legality-check", "--metadata-only", "--dry-run"]
    legality_summary = _run_cli(legality_args)
    steps.append({"step": "legality_check", "summary": legality_summary})

    plan_summary: dict[str, Any] = {}
    if options.plan_downloads or options.download:
        plan_args = ["--config", str(config_path), "--plan-downloads", "--dry-run"]
        if options.max_downloads:
            plan_args.extend(["--max-downloads", str(options.max_downloads)])
        plan_summary = _run_cli(plan_args)
        steps.append({"step": "download_plan", "summary": plan_summary})

    download_summary: dict[str, Any] = {}
    planned_count = int(plan_summary.get("planned_downloads") or 0)
    proceed_download = _confirm_download(options) if planned_count else False
    if proceed_download and options.allow_download:
        download_args = ["--config", str(config_path), "--download", "--allow-download"]
        if options.max_downloads:
            download_args.extend(["--max-downloads", str(options.max_downloads)])
        download_summary = _run_cli(download_args)
        steps.append({"step": "download", "summary": download_summary})

    summary = {
        "request": request_text,
        "config_path": str(config_path),
        "queries_path": str(queries_path),
        "dry_run": not (proceed_download and options.allow_download),
        "allow_network_metadata": options.allow_network_metadata,
        "allow_network_rank": options.allow_network_rank,
        "download_requested": options.download,
        "download_executed": bool(download_summary),
        "steps": steps,
    }
    en_report, zh_report = write_reports(summary, output_dir=options.output_dir)
    summary["report_en"] = str(en_report)
    summary["report_zh"] = str(zh_report)
    return summary
