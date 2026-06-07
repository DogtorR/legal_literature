from __future__ import annotations

from pathlib import Path

from lit_agent.collection_orchestrator import _write_full_collection_report, check_collection_completion, select_batches_for_round, run_full_collection_loop
from lit_agent.crispr_scope import PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY
from lit_agent.manifest import read_jsonl


def test_full_collection_from_empty_dir_creates_batch_manifest(tmp_path: Path, monkeypatch) -> None:
    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        return {**batch, "status": "completed", "retrieved_count": 1, "total_count": 1, "has_more": False}

    monkeypatch.setattr("lit_agent.collection_orchestrator.run_search_batch", fake_run_search_batch)
    result = run_full_collection_loop(
        query_list=["CRISPR detection"],
        sources=["pubmed"],
        year_from=2025,
        year_to=2026,
        max_rounds=1,
        max_batches_per_round=1,
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "batch_manifest.jsonl").exists()
    assert result["total_batches"] == 1


def test_full_collection_auto_loops_partial_batch(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return {**batch, "status": "partial", "retrieved_count": 1, "total_count": 2, "has_more": True, "continuation_supported": True}
        return {**batch, "status": "completed", "retrieved_count": 2, "total_count": 2, "has_more": False, "continuation_supported": True}

    monkeypatch.setattr("lit_agent.collection_orchestrator.run_search_batch", fake_run_search_batch)
    result = run_full_collection_loop(
        query_list=["CRISPR detection"],
        sources=["pubmed"],
        year_from=2025,
        year_to=2026,
        max_rounds=3,
        max_batches_per_round=1,
        min_new_unique_records_per_round=0,
        output_dir=str(tmp_path),
    )

    assert calls["count"] >= 2
    assert result["rounds_run"] >= 2


def test_completion_checker_continues_when_total_count_gt_retrieved_count() -> None:
    decision = check_collection_completion(
        [{"status": "partial", "has_more": True, "total_count": 10, "retrieved_count": 5, "continuation_supported": True}],
        round_index=1,
        low_growth_rounds=0,
        max_rounds=5,
    )

    assert decision["should_continue"] is True


def test_completion_checker_stops_when_all_completed() -> None:
    decision = check_collection_completion([{"status": "completed", "has_more": False}], round_index=1, low_growth_rounds=0, max_rounds=5)

    assert decision["done"] is True
    assert decision["stop_reason"] == "all_batches_completed"


def test_completion_checker_stops_on_low_growth() -> None:
    decision = check_collection_completion([{"status": "partial", "has_more": True, "continuation_supported": True}], round_index=1, low_growth_rounds=2, max_rounds=5, stop_if_no_growth_rounds=2)

    assert decision["done"] is True
    assert decision["stop_reason"] == "low_growth_stop"


def test_completion_checker_stops_on_max_rounds() -> None:
    decision = check_collection_completion([{"status": "pending"}], round_index=5, low_growth_rounds=0, max_rounds=5)

    assert decision["done"] is True
    assert decision["stop_reason"] == "max_rounds_reached"


def test_full_collection_download_false_no_download(tmp_path: Path, monkeypatch) -> None:
    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        return {**batch, "status": "completed", "retrieved_count": 0, "total_count": 0, "has_more": False}

    monkeypatch.setattr("lit_agent.collection_orchestrator.run_search_batch", fake_run_search_batch)
    result = run_full_collection_loop(
        query_list=["CRISPR detection"],
        sources=["pubmed"],
        year_from=2025,
        year_to=2026,
        max_rounds=1,
        max_batches_per_round=1,
        output_dir=str(tmp_path),
    )

    assert result["actual_downloads"] == 0
    assert not (tmp_path / "downloads").exists()
    assert read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl") == []


def test_select_batches_prefers_pending_over_partial() -> None:
    rows = [
        {"batch_id": "partial_primary_pubmed", "query": PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY, "source": "pubmed", "status": "partial"},
        {"batch_id": "pending_detection_pubmed", "query": "CRISPR detection", "source": "pubmed", "status": "pending"},
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=1)

    assert selected[0]["batch_id"] == "pending_detection_pubmed"


def test_select_batches_prioritizes_primary_query() -> None:
    rows = [
        {"batch_id": "pending_detection_pubmed", "query": "CRISPR detection", "source": "pubmed", "status": "pending"},
        {"batch_id": "pending_primary_pubmed", "query": PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY, "source": "pubmed", "status": "pending"},
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=1)

    assert selected[0]["batch_id"] == "pending_primary_pubmed"
    assert "primary_broad_query" in selected[0]["selected_reason"]


def test_select_batches_balances_sources() -> None:
    rows = [
        {"batch_id": f"primary_{source}", "query": PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY, "source": source, "status": "pending"}
        for source in ["crossref", "europe_pmc", "openalex", "pubmed"]
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=4)

    assert [row["source"] for row in selected] == ["pubmed", "openalex", "europe_pmc", "crossref"]


def test_select_batches_skips_completed() -> None:
    rows = [
        {"batch_id": "done", "query": PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY, "source": "pubmed", "status": "completed"},
        {"batch_id": "todo", "query": PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY, "source": "openalex", "status": "pending"},
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=2)

    assert [row["batch_id"] for row in selected] == ["todo"]


def test_select_batches_respects_max_batches_per_round() -> None:
    rows = [
        {"batch_id": f"batch_{index}", "query": "CRISPR detection", "source": "pubmed", "status": "pending"}
        for index in range(5)
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=2)

    assert len(selected) == 2


def test_full_collection_graceful_time_budget_stop(tmp_path: Path) -> None:
    result = run_full_collection_loop(
        query_list=["CRISPR detection"],
        sources=["pubmed"],
        year_from=2025,
        year_to=2026,
        max_rounds=3,
        max_batches_per_round=1,
        time_budget_seconds=0,
        graceful_stop_buffer_seconds=1,
        output_dir=str(tmp_path),
    )

    assert result["stop_reason"] == "time_budget_reached"
    assert (tmp_path / "full_collection_report.md").exists()
    assert (tmp_path / "corpus_manifest.jsonl").exists()


def test_full_collection_resume_existing_output_dir(tmp_path: Path, monkeypatch) -> None:
    completed = {"batch_id": "done", "query": "CRISPR detection", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "completed"}
    pending = {"batch_id": "todo", "query": "CRISPR diagnostics", "source": "pubmed", "year_from": 2025, "year_to": 2026, "max_results": 10, "status": "pending"}
    (tmp_path / "batch_manifest.jsonl").write_text(__import__("json").dumps(completed) + "\n" + __import__("json").dumps(pending) + "\n", encoding="utf-8")
    calls = []

    def fake_run_search_batch(batch, output_dir, timeout_seconds=30, resume=False):
        calls.append(batch["batch_id"])
        return {**batch, "status": "completed", "retrieved_count": 0, "total_count": 0, "has_more": False}

    monkeypatch.setattr("lit_agent.collection_orchestrator.run_search_batch", fake_run_search_batch)
    result = run_full_collection_loop(
        query_list=["CRISPR detection", "CRISPR diagnostics"],
        sources=["pubmed"],
        year_from=2025,
        year_to=2026,
        max_rounds=1,
        max_batches_per_round=2,
        output_dir=str(tmp_path),
    )

    assert result["resume_detected"] is True
    assert "done" not in calls
    assert "todo" in calls


def test_scheduler_flags_overrepresented_queries(tmp_path: Path) -> None:
    rows = [
        {"batch_id": "holmes", "query": "HOLMES", "source": "pubmed", "status": "partial", "retrieved_count": 100},
        {"batch_id": "structure", "query": "CRISPR structure", "source": "pubmed", "status": "pending", "retrieved_count": 0},
    ]
    (tmp_path / "batch_manifest.jsonl").write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
    _write_full_collection_report(str(tmp_path), {"total_batches": 2, "pending_batches": 1, "partial_batches": 1}, [])
    text = (tmp_path / "full_collection_report.md").read_text(encoding="utf-8")

    assert "overrepresented_queries" in text
    assert "HOLMES" in text


def test_scheduler_boosts_undercovered_structure_queries() -> None:
    rows = [
        {"batch_id": "holmes", "query": "HOLMES", "source": "pubmed", "status": "pending", "retrieved_count": 100},
        {"batch_id": "structure", "query": "CRISPR structure", "source": "pubmed", "status": "pending", "retrieved_count": 0},
    ]

    selected = select_batches_for_round(rows, max_batches_per_round=1)

    assert selected[0]["batch_id"] == "structure"


def test_no_annotation_pipeline_added() -> None:
    root = Path("lit_agent")
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in root.glob("*.py"))
    for banned in ["annotation_pipeline", "auto_label", "training_data_extraction"]:
        assert banned not in text


def test_resume_mcp_tool_exists() -> None:
    import lit_agent.mcp_server as ms

    assert hasattr(ms, "resume_crispr_broad_full_collection")
