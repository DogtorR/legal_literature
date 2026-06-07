"""Markdown reports for pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import read_jsonl


def _count_status(records: list[dict[str, Any]], field: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field) or "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _sample_lines(records: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for record in records[:limit]:
        title = str(record.get("title") or "(untitled)").replace("\n", " ")
        doi = str(record.get("doi") or "")
        journal = str(record.get("journal") or "")
        status = str(record.get("status") or record.get("decision") or record.get("reason") or "")
        lines.append(f"- {title} | {journal} | {doi} | {status}")
    return lines or ["- No records."]


def _load_outputs(base_dir: str | Path = ".") -> dict[str, list[dict[str, Any]]]:
    root = Path(base_dir)
    return {
        "metadata": read_jsonl(root / "metadata_results.jsonl"),
        "candidates": read_jsonl(root / "candidates.jsonl"),
        "legality": read_jsonl(root / "legality_audit.jsonl"),
        "download_plan": read_jsonl(root / "download_plan.jsonl"),
        "manifest": read_jsonl(root / "manifest.jsonl"),
        "failures": read_jsonl(root / "failures.jsonl"),
        "rank_cache": read_jsonl(root / "journal_rank_cache.jsonl"),
    }


def render_english_report(summary: dict[str, Any], *, base_dir: str | Path = ".") -> str:
    outputs = _load_outputs(base_dir)
    manifest_counts = _count_status(outputs["manifest"])
    plan_counts = _count_status(outputs["download_plan"], field="planned_status")
    legality_counts = _count_status(outputs["legality"], field="decision")
    lines = [
        "# Literature Agent Report",
        "",
        "## Task",
        f"- Request: {summary.get('request', '')}",
        f"- Config: {summary.get('config_path', '')}",
        f"- Dry run: {summary.get('dry_run', True)}",
        f"- Network metadata: {summary.get('allow_network_metadata', False)}",
        f"- Network LetPub rank lookup: {summary.get('allow_network_rank', False)}",
        "",
        "## Counts",
        f"- Metadata records: {len(outputs['metadata'])}",
        f"- Candidate records: {len(outputs['candidates'])}",
        f"- Legality audit records: {len(outputs['legality'])}",
        f"- Download plan records: {len(outputs['download_plan'])}",
        f"- Manifest records: {len(outputs['manifest'])}",
        f"- Failure records: {len(outputs['failures'])}",
        f"- Journal rank cache records: {len(outputs['rank_cache'])}",
        "",
        "## Status Breakdowns",
        f"- Legality decisions: {legality_counts}",
        f"- Download plan statuses: {plan_counts}",
        f"- Manifest statuses: {manifest_counts}",
        "",
        "## Sample Metadata",
        *_sample_lines(outputs["metadata"]),
        "",
        "## Sample Download Plan",
        *_sample_lines(outputs["download_plan"]),
        "",
        "## Safety Note",
        "The agent does not use Sci-Hub, institutional logins, cookies, VPNs, CAPTCHA bypass, or paywall bypass. Real downloads require legal OA evidence and final preflight.",
    ]
    return "\n".join(lines) + "\n"


def render_chinese_report(summary: dict[str, Any], *, base_dir: str | Path = ".") -> str:
    outputs = _load_outputs(base_dir)
    manifest_counts = _count_status(outputs["manifest"])
    plan_counts = _count_status(outputs["download_plan"], field="planned_status")
    legality_counts = _count_status(outputs["legality"], field="decision")
    lines = [
        "# 文献 Agent 报告",
        "",
        "## 任务",
        f"- 用户请求：{summary.get('request', '')}",
        f"- 配置文件：{summary.get('config_path', '')}",
        f"- Dry-run：{summary.get('dry_run', True)}",
        f"- 联网检索 metadata：{summary.get('allow_network_metadata', False)}",
        f"- 联网查询 LetPub：{summary.get('allow_network_rank', False)}",
        "",
        "## 数量统计",
        f"- Metadata 记录：{len(outputs['metadata'])}",
        f"- 候选记录：{len(outputs['candidates'])}",
        f"- 合法性审计记录：{len(outputs['legality'])}",
        f"- 下载计划记录：{len(outputs['download_plan'])}",
        f"- Manifest 记录：{len(outputs['manifest'])}",
        f"- 失败记录：{len(outputs['failures'])}",
        f"- 期刊指标缓存记录：{len(outputs['rank_cache'])}",
        "",
        "## 状态分布",
        f"- 合法性判断：{legality_counts}",
        f"- 下载计划状态：{plan_counts}",
        f"- Manifest 状态：{manifest_counts}",
        "",
        "## Metadata 示例",
        *_sample_lines(outputs["metadata"]),
        "",
        "## 下载计划示例",
        *_sample_lines(outputs["download_plan"]),
        "",
        "## 安全说明",
        "Agent 不使用 Sci-Hub、机构登录、cookie、VPN、验证码绕过或付费墙绕过。真实下载必须有合法 OA 证据，并通过最终 preflight。",
    ]
    return "\n".join(lines) + "\n"


def write_reports(summary: dict[str, Any], *, output_dir: str | Path, base_dir: str | Path = ".") -> tuple[Path, Path]:
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    en_path = report_dir / "report_en.md"
    zh_path = report_dir / "report_zh.md"
    en_path.write_text(render_english_report(summary, base_dir=base_dir), encoding="utf-8")
    zh_path.write_text(render_chinese_report(summary, base_dir=base_dir), encoding="utf-8")
    return en_path, zh_path
