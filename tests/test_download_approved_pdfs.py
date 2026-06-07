from __future__ import annotations

import json
from pathlib import Path

from lit_agent.batch_runner import download_approved_legal_oa_pdfs, resolve_legal_oa_pdf_candidates
from lit_agent.manifest import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _approved(url: str, record_id: str = "r1") -> dict:
    return {
        "record_id": record_id,
        "doi": "10.1/test",
        "title": "Legal OA PDF",
        "journal": "J",
        "year": 2026,
        "pdf_url": url,
        "oa_source": "unpaywall",
        "license": "cc-by",
        "oa_evidence": "Unpaywall confirmed OA",
        "source_provenance": ["unpaywall"],
        "matched_queries": ["CRISPR detection"],
    }


def test_allow_download_false_does_not_download(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nx")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=False, yes=False, max_downloads=1)

    assert result["actual_downloads"] == 0
    assert result["skipped_downloads"] == 1


def test_yes_false_does_not_download(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nx")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=False, max_downloads=1)

    assert result["actual_downloads"] == 0
    assert result["skipped_downloads"] == 1


def test_only_approved_file_is_downloaded(tmp_path: Path) -> None:
    approved_pdf = tmp_path / "approved.pdf"
    manual_pdf = tmp_path / "manual.pdf"
    rejected_pdf = tmp_path / "rejected.pdf"
    approved_pdf.write_bytes(b"%PDF-1.4\napproved")
    manual_pdf.write_bytes(b"%PDF-1.4\nmanual")
    rejected_pdf.write_bytes(b"%PDF-1.4\nrejected")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(approved_pdf.as_uri(), "approved")])
    _write_jsonl(tmp_path / "needs_manual_download_review.jsonl", [_approved(manual_pdf.as_uri(), "manual")])
    _write_jsonl(tmp_path / "rejected_before_download.jsonl", [_approved(rejected_pdf.as_uri(), "rejected")])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_downloads"] == 1
    assert read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")[0]["record_id"] == "approved"


def test_forbidden_source_not_downloaded(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved("https://sci-hub.example/p.pdf")])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_downloads"] == 0
    assert read_jsonl(tmp_path / "failed_downloads.jsonl")[0]["failure_reason"] == "forbidden_source_or_bypass_risk"


def test_success_record_contains_file_size_and_checksum(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nchecksum")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    row = read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")[0]
    assert row["file_size"] > 0
    assert len(row["checksum_sha256"]) == 64


def test_download_failure_written(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved((tmp_path / "missing.pdf").as_uri())])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    assert result["failed_downloads"] == 1
    assert read_jsonl(tmp_path / "failed_downloads.jsonl")[0]["retryable"] is True


def test_max_downloads_respected(tmp_path: Path) -> None:
    rows = []
    for idx in range(3):
        pdf = tmp_path / f"source_{idx}.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + bytes([idx]))
        rows.append(_approved(pdf.as_uri(), f"r{idx}"))
    _write_jsonl(tmp_path / "approved_for_download.jsonl", rows)

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True, max_downloads=2)

    assert result["actual_downloads"] == 2
    assert len(read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")) == 2


def test_403_like_failure_tries_next_legal_candidate(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.pdf"
    fallback.write_bytes(b"%PDF-1.4\nfallback")
    row = _approved((tmp_path / "missing.pdf").as_uri())
    row["oa_locations"] = [{"pdf_url": fallback.as_uri(), "source": "pmc", "evidence": "PMC OA", "license": "cc-by"}]
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_downloads"] == 1
    downloaded = read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")[0]
    assert downloaded["candidate_rank"] == 2
    assert downloaded["selected_candidate_source"] == "pmc"
    assert len(downloaded["download_attempts"]) == 2


def test_forbidden_candidate_is_not_attempted(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.pdf"
    fallback.write_bytes(b"%PDF-1.4\nfallback")
    row = _approved("https://sci-hub.example/p.pdf")
    row["oa_locations"] = [{"pdf_url": fallback.as_uri(), "source": "pmc", "evidence": "PMC OA", "license": "cc-by"}]

    candidates = resolve_legal_oa_pdf_candidates(row)

    assert candidates[0]["rejected"] is True


def test_all_candidates_failed_writes_attempted_candidates(tmp_path: Path) -> None:
    row = _approved((tmp_path / "missing1.pdf").as_uri())
    row["oa_locations"] = [{"pdf_url": (tmp_path / "missing2.pdf").as_uri(), "source": "pmc", "evidence": "PMC OA", "license": "cc-by"}]
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    assert result["failed_downloads"] == 1
    failed = read_jsonl(tmp_path / "failed_downloads.jsonl")[0]
    assert len(failed["attempted_candidates"]) == 2
    assert failed["final_failure_reason"]


def test_success_record_contains_candidate_source_rank_checksum(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\ncandidate")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    download_approved_legal_oa_pdfs(str(tmp_path), allow_download=True, yes=True)

    row = read_jsonl(tmp_path / "downloaded_pdfs_manifest.jsonl")[0]
    assert row["selected_candidate_source"] == "unpaywall"
    assert row["candidate_rank"] == 1
    assert len(row["checksum_sha256"]) == 64


def test_downloader_does_not_use_cookies_login_or_bypass() -> None:
    source = Path("lit_agent/batch_runner.py").read_text(encoding="utf-8")

    assert '"Cookie"' not in source
    assert "captcha solver" not in source.lower()
