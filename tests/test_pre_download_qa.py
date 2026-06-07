from __future__ import annotations

import json
from pathlib import Path

from lit_agent.batch_runner import pre_download_qa_for_legal_oa_candidates
from lit_agent.manifest import read_jsonl
from lit_agent.mcp_server import pre_download_qa_for_legal_oa_candidates as mcp_pre_download_qa


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_no_legal_pdf_url_not_approved(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "title": "Paper", "full_text_status": "planned_legal_oa", "oa_evidence": "Unpaywall", "source_provenance": ["unpaywall"], "license": "cc-by"}])

    pre_download_qa_for_legal_oa_candidates(str(tmp_path))

    assert read_jsonl(tmp_path / "approved_for_download.jsonl") == []
    assert read_jsonl(tmp_path / "rejected_before_download.jsonl")[0]["rejection_reason"] == "missing_legal_pdf_url"


def test_no_oa_evidence_not_approved(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "title": "Paper", "pdf_url": "https://example.org/p.pdf", "source_provenance": ["publisher"], "license": "cc-by"}])

    pre_download_qa_for_legal_oa_candidates(str(tmp_path))

    assert read_jsonl(tmp_path / "approved_for_download.jsonl") == []
    assert read_jsonl(tmp_path / "rejected_before_download.jsonl")[0]["rejection_reason"] == "missing_oa_evidence"


def test_forbidden_source_not_approved(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "title": "Paper", "pdf_url": "https://sci-hub.example/p.pdf", "oa_evidence": "OA", "source_provenance": ["unpaywall"], "license": "cc-by"}])

    pre_download_qa_for_legal_oa_candidates(str(tmp_path))

    assert read_jsonl(tmp_path / "approved_for_download.jsonl") == []
    assert read_jsonl(tmp_path / "rejected_before_download.jsonl")[0]["rejection_reason"] == "forbidden_source_or_bypass_risk"


def test_missing_license_with_evidence_needs_manual_review(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "doi": "10.1/a", "title": "Paper", "pdf_url": "https://example.org/p.pdf", "oa_evidence": "Unpaywall confirmed OA", "source_provenance": ["unpaywall"]}])

    pre_download_qa_for_legal_oa_candidates(str(tmp_path))

    assert read_jsonl(tmp_path / "approved_for_download.jsonl") == []
    assert read_jsonl(tmp_path / "needs_manual_download_review.jsonl")[0]["review_reason"] == "missing_license_with_oa_evidence"


def test_legal_oa_pdf_evidence_doi_title_approved(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "doi": "10.1/a", "title": "Paper", "journal": "J", "year": 2026, "pdf_url": "https://example.org/p.pdf", "oa_evidence": "Unpaywall confirmed OA", "source_provenance": ["unpaywall"], "license": "cc-by", "matched_queries": ["CRISPR detection"]}])

    pre_download_qa_for_legal_oa_candidates(str(tmp_path))

    row = read_jsonl(tmp_path / "approved_for_download.jsonl")[0]
    assert row["doi"] == "10.1/a"
    assert row["approval_reason"]


def test_pre_download_qa_does_not_download_and_writes_report(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "corpus_manifest.jsonl", [{"record_id": "r1", "doi": "10.1/a", "title": "Paper", "pdf_url": "https://example.org/p.pdf", "oa_evidence": "OA", "source_provenance": ["publisher"], "license": "cc-by"}])

    result = mcp_pre_download_qa(output_dir=str(tmp_path))

    assert result["actual_downloads"] == 0
    assert not (tmp_path / "downloads").exists()
    assert (tmp_path / "pre_download_qa_report.md").exists()
