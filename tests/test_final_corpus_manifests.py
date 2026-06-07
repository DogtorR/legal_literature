from __future__ import annotations

import json
from pathlib import Path

from lit_agent.batch_runner import build_final_corpus_manifests
from lit_agent.manifest import read_jsonl
from lit_agent.mcp_server import finalize_crispr_broad_corpus_manifests


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_metadata_only_retained_when_non_oa(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "Closed CRISPR detection paper", "doi": "10.1/nonoa", "is_oa": False, "scope_status": "included"}])
    build_final_corpus_manifests(str(tmp_path))

    rows = read_jsonl(tmp_path / "metadata_only_manifest.jsonl")
    assert rows[0]["doi"] == "10.1/nonoa"
    assert rows[0]["full_text_status"] == "metadata_only_non_oa"
    assert rows[0]["can_use_metadata"] is True


def test_metadata_only_retained_when_no_pdf_url(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "OA without PDF", "doi": "10.1/nopdf", "is_oa": True, "evidence": "publisher OA", "scope_status": "included"}])
    build_final_corpus_manifests(str(tmp_path))

    rows = read_jsonl(tmp_path / "metadata_only_manifest.jsonl")
    assert rows[0]["full_text_status"] == "metadata_only_no_pdf_url"


def test_planned_legal_oa_status(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "oa_audit.jsonl",
        [{"title": "Legal OA CRISPR", "doi": "10.1/oa", "is_oa": True, "can_download": True, "legal_pdf_url": "https://example.org/a.pdf", "evidence": "Unpaywall legal OA"}],
    )
    build_final_corpus_manifests(str(tmp_path))

    rows = read_jsonl(tmp_path / "corpus_manifest.jsonl")
    assert rows[0]["full_text_status"] == "planned_legal_oa"
    assert rows[0]["can_use_full_text"] is False


def test_downloaded_pdf_status(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "Downloaded CRISPR", "doi": "10.1/done", "is_oa": True, "can_download": True, "legal_pdf_url": "https://example.org/d.pdf", "evidence": "PMC"}])
    _write_jsonl(tmp_path / "download_results.jsonl", [{"doi": "10.1/done", "status": "downloaded", "local_path": "downloads/done.pdf"}])
    build_final_corpus_manifests(str(tmp_path))

    rows = read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")
    assert rows[0]["full_text_status"] == "downloaded_pdf"
    assert rows[0]["can_use_full_text"] is True


def test_corpus_manifest_contains_metadata_and_pdf_layers(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "Layered CRISPR", "doi": "10.1/layers", "is_oa": False}])
    build_final_corpus_manifests(str(tmp_path))

    row = read_jsonl(tmp_path / "corpus_manifest.jsonl")[0]
    assert "can_use_metadata" in row
    assert "can_use_full_text" in row
    assert row["can_use_metadata"] is True


def test_failures_do_not_remove_metadata(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "Needs evidence", "doi": "10.1/evidence", "is_oa": True, "legal_pdf_url": "https://example.org/e.pdf"}])
    _write_jsonl(tmp_path / "scope_excluded_records.jsonl", [{"title": "Pure editing", "doi": "10.1/editing", "scope_reason": "pure_gene_editing"}])
    build_final_corpus_manifests(str(tmp_path))

    metadata_ids = {row["record_id"] for row in read_jsonl(tmp_path / "metadata_only_manifest.jsonl")}
    excluded_ids = {row["record_id"] for row in [{"record_id": "10_1_editing"}]}
    for failure in read_jsonl(tmp_path / "failures.jsonl"):
        assert failure["trace_location"] in {"metadata_only_manifest.jsonl", "scope_excluded_records.jsonl"}
        assert failure["record_id"] in metadata_ids or failure["record_id"] in excluded_ids


def test_finalize_crispr_broad_corpus_manifests_no_download(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "oa_audit.jsonl", [{"title": "No download finalize", "doi": "10.1/finalize", "is_oa": False}])

    result = finalize_crispr_broad_corpus_manifests(output_dir=str(tmp_path))

    assert result["actual_downloads"] == 0
    assert (tmp_path / "corpus_manifest.jsonl").exists()
    assert not (tmp_path / "downloads").exists()
