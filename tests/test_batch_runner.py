from __future__ import annotations

import json
from pathlib import Path

from lit_agent.batch_runner import (
    build_corpus_search_batches,
    continue_partial_batches,
    merge_batch_outputs,
    run_corpus_batch_search,
    run_search_batch,
)
from lit_agent.crispr_scope import BROAD_CRISPR_CORPUS_QUERIES


SOURCES = ["pubmed", "openalex", "crossref", "europe_pmc"]


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def test_build_corpus_search_batches() -> None:
    batches = build_corpus_search_batches(BROAD_CRISPR_CORPUS_QUERIES, SOURCES, 2025, 2026, 500)

    assert len(BROAD_CRISPR_CORPUS_QUERIES) == 21
    assert len(batches) == 84
    assert {batch["source"] for batch in batches} == set(SOURCES)


def test_batch_manifest_schema() -> None:
    batch = build_corpus_search_batches(["CRISPR detection"], ["pubmed"], 2025, 2026, 100)[0]

    for key in ["batch_id", "query", "source", "year_from", "year_to", "max_results", "status"]:
        assert key in batch


def test_partial_batch_continuation_schema() -> None:
    batch = build_corpus_search_batches(["CRISPR detection"], ["pubmed"], 2025, 2026, 100)[0]

    for key in ["next_cursor", "next_offset", "retstart", "has_more", "stop_reason", "continuation_supported"]:
        assert key in batch


def test_resume_skips_completed_batch(tmp_path: Path, monkeypatch) -> None:
    batch = build_corpus_search_batches(["CRISPR detection"], ["pubmed"], 2025, 2026, 10)[0]
    manifest = tmp_path / "batch_manifest.jsonl"
    manifest.write_text(json.dumps({**batch, "status": "completed", "retrieved_count": 0}) + "\n", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_search_batch(_batch, _output_dir, timeout_seconds=30):
        calls["count"] += 1
        return {**_batch, "status": "completed", "retrieved_count": 0}

    monkeypatch.setattr("lit_agent.batch_runner.run_search_batch", fake_run_search_batch)

    result = run_corpus_batch_search(
        ["CRISPR detection"],
        ["pubmed"],
        2025,
        2026,
        max_results_per_batch=10,
        output_dir=str(tmp_path),
        resume=True,
        retry_failed=False,
    )

    assert calls["count"] == 0
    assert result["completed_batches"] == 1
    assert (tmp_path / "batch_report.md").exists()


def test_failed_batch_recorded(tmp_path: Path) -> None:
    batch = {
        "batch_id": "bad_source_batch",
        "query": "CRISPR detection",
        "source": "bad_source",
        "year_from": 2025,
        "year_to": 2026,
        "max_results": 10,
        "status": "pending",
    }

    result = run_search_batch(batch, str(tmp_path), timeout_seconds=1)

    assert result["status"] == "failed"
    assert "Unsupported source" in result["error"]
    assert (tmp_path / "batch_manifest.jsonl").exists()
    assert _jsonl_count(tmp_path / "batch_manifest.jsonl") == 1


def test_merge_batch_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_records_by_batch"
    raw_dir.mkdir(parents=True)
    records = [
        {
            "title": "CRISPR detection assay with Cas12",
            "doi": "10.0000/example-1",
            "publication_year": 2025,
            "abstract": "A CRISPR Cas12 detection assay for nucleic acid diagnostics.",
            "source": "pubmed",
            "matched_queries": ["CRISPR detection"],
        },
        {
            "title": "CRISPR detection assay with Cas12",
            "doi": "10.0000/example-1",
            "publication_year": 2025,
            "abstract": "Duplicate metadata from another source.",
            "source": "openalex",
            "matched_queries": ["Cas12 detection"],
        },
    ]
    (raw_dir / "a.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    (tmp_path / "batch_manifest.jsonl").write_text(
        json.dumps({"batch_id": "a", "query": "CRISPR detection", "source": "pubmed", "retrieved_count": 2, "status": "completed"}) + "\n",
        encoding="utf-8",
    )

    result = merge_batch_outputs(str(tmp_path))

    assert result["total_raw_records"] == 2
    assert result["unique_records"] == 1
    assert (tmp_path / "raw_records_merged.jsonl").exists()
    assert (tmp_path / "metadata_all.jsonl").exists()
    assert (tmp_path / "batch_report.md").exists()


def test_continue_partial_batches_filters_partial_only(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"batch_id": "done", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "completed"},
        {"batch_id": "bad", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "failed"},
        {"batch_id": "part", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "partial", "retrieved_count": 1},
    ]
    (tmp_path / "batch_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    calls = []

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        calls.append(batch["batch_id"])
        return {**batch, "status": "completed", "retrieved_count": 2, "has_more": False, "output_file": str(tmp_path / "raw_records_by_batch" / "part.jsonl")}

    monkeypatch.setattr("lit_agent.batch_runner.run_search_batch", fake_run_search_batch)
    result = continue_partial_batches(str(tmp_path), max_additional_results_per_batch=10)

    assert calls == ["part"]
    assert result["continued_batches"] == 1
    assert result["actual_downloads"] == 0


def test_continue_partial_batches_respects_max_batches(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"batch_id": f"part{i}", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "partial"}
        for i in range(4)
    ]
    (tmp_path / "batch_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    calls = []

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        calls.append(batch["batch_id"])
        return {**batch, "status": "partial", "retrieved_count": 1, "has_more": True}

    monkeypatch.setattr("lit_agent.batch_runner.run_search_batch", fake_run_search_batch)
    result = continue_partial_batches(str(tmp_path), max_additional_results_per_batch=10, max_batches=2)

    assert calls == ["part0", "part1"]
    assert result["continued_batches"] == 2


def test_continue_partial_batches_no_download(tmp_path: Path, monkeypatch) -> None:
    row = {"batch_id": "part", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "partial"}
    (tmp_path / "batch_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        return {**batch, "status": "completed", "retrieved_count": 0, "has_more": False}

    monkeypatch.setattr("lit_agent.batch_runner.run_search_batch", fake_run_search_batch)
    result = continue_partial_batches(str(tmp_path), max_additional_results_per_batch=10)

    assert result["actual_downloads"] == 0
    assert not (tmp_path / "downloads").exists()


def test_continuation_report_created(tmp_path: Path, monkeypatch) -> None:
    row = {"batch_id": "part", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "partial"}
    (tmp_path / "batch_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        return {**batch, "status": "completed", "retrieved_count": 0, "has_more": False}

    monkeypatch.setattr("lit_agent.batch_runner.run_search_batch", fake_run_search_batch)
    result = continue_partial_batches(str(tmp_path), max_additional_results_per_batch=10)

    assert (tmp_path / "continuation_report.md").exists()
    assert "continuation_report" in result["artifacts"]
