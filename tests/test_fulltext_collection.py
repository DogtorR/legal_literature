from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.error import HTTPError

import lit_agent.batch_runner as br
from lit_agent.batch_runner import (
    collect_approved_legal_fulltexts,
    is_safe_legal_fulltext_candidate,
    retry_failed_legal_fulltext_fetches,
    resolve_legal_fulltext_candidates,
)
from lit_agent.manifest import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _approved(url: str, record_id: str = "r1") -> dict:
    return {
        "record_id": record_id,
        "doi": "10.1/fulltext",
        "title": "Legal OA full text",
        "journal": "J",
        "year": 2026,
        "pdf_url": url,
        "oa_source": "unpaywall",
        "license": "cc-by",
        "oa_evidence": "Unpaywall confirmed OA",
        "source_provenance": ["unpaywall"],
        "matched_queries": ["CRISPR detection"],
    }


def test_allow_download_false_does_not_collect_fulltext(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nx")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=False, yes=False, max_items=1)

    assert result["actual_fulltexts"] == 0
    assert result["skipped_items"] == 1
    assert read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl") == []


def test_yes_false_does_not_collect_fulltext(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nx")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=False, max_items=1)

    assert result["actual_fulltexts"] == 0
    assert result["skipped_items"] == 1


def test_only_approved_file_is_used_for_fulltext_collection(tmp_path: Path) -> None:
    approved = tmp_path / "approved.pdf"
    manual = tmp_path / "manual.pdf"
    rejected = tmp_path / "rejected.pdf"
    approved.write_bytes(b"%PDF-1.4\napproved")
    manual.write_bytes(b"%PDF-1.4\nmanual")
    rejected.write_bytes(b"%PDF-1.4\nrejected")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(approved.as_uri(), "approved")])
    _write_jsonl(tmp_path / "needs_manual_download_review.jsonl", [_approved(manual.as_uri(), "manual")])
    _write_jsonl(tmp_path / "rejected_before_download.jsonl", [_approved(rejected.as_uri(), "rejected")])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_fulltexts"] == 1
    assert read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")[0]["record_id"] == "approved"


def test_pdf_collects_to_pdf_folder(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nchecksum")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [_approved(pdf.as_uri())])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    row = read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")[0]
    assert result["pdf_count"] == 1
    assert row["full_text_type"] == "pdf"
    assert row["file_size"] > 0
    assert len(row["checksum_sha256"]) == 64
    assert "\\fulltexts\\pdf\\" in row["local_path"] or "/fulltexts/pdf/" in row["local_path"]


def test_pdf_failure_can_fallback_to_xml_success(tmp_path: Path) -> None:
    xml = tmp_path / "article.xml"
    xml.write_text("<article>legal oa</article>", encoding="utf-8")
    row = _approved((tmp_path / "missing.pdf").as_uri())
    row["fullTextXML"] = xml.as_uri()

    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])
    result = collect_approved_legal_fulltexts(
        str(tmp_path),
        allow_download=True,
        yes=True,
        prefer_formats=["pdf", "europe_pmc_xml"],
    )

    manifest = read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")[0]
    assert result["actual_fulltexts"] == 1
    assert result["xml_count"] == 1
    assert manifest["full_text_type"] == "europe_pmc_xml"
    assert len(manifest["attempts"]) == 2


def test_forbidden_fulltext_candidate_rejected(tmp_path: Path) -> None:
    row = _approved("https://sci-hub.example/full.pdf")
    candidates = resolve_legal_fulltext_candidates(row)

    assert candidates[0]["rejected"] is True
    assert is_safe_legal_fulltext_candidate(candidates[0])[0] is False


def test_missing_oa_evidence_goes_to_manual_or_failed(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nx")
    row = _approved(pdf.as_uri())
    row["oa_evidence"] = ""
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_fulltexts"] == 0
    assert result["needs_manual_fulltext_review"] == 1
    assert read_jsonl(tmp_path / "failed_fulltext_fetches.jsonl")[0]["final_failure_reason"] == "no_safe_legal_fulltext_candidate"


def test_failed_fulltext_fetch_records_attempted_candidates(tmp_path: Path) -> None:
    row = _approved((tmp_path / "missing.pdf").as_uri())
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["failed_fulltext_fetches"] == 1
    failed = read_jsonl(tmp_path / "failed_fulltext_fetches.jsonl")[0]
    assert failed["attempted_candidates"]
    assert failed["final_failure_reason"]
    assert "recommended_next_action" in failed


def test_pdf_failure_tries_unpaywall_fallback(tmp_path: Path, monkeypatch) -> None:
    fallback = tmp_path / "fallback.pdf"
    fallback.write_bytes(b"%PDF-1.4\nunpaywall")
    row = _approved((tmp_path / "missing.pdf").as_uri())
    row["force_network_fallback"] = True

    from lit_agent.sources import unpaywall

    def fake_check_doi(*_args, **_kwargs):
        return {
            "is_legal_oa_candidate": True,
            "unpaywall_best_oa_location": {"url_for_pdf": fallback.as_uri(), "source": "repository", "license": "cc-by"},
            "oa_evidence": "Unpaywall confirmed OA",
            "license": "cc-by",
        }

    monkeypatch.setattr(unpaywall, "check_doi", fake_check_doi)
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_fulltexts"] == 1
    manifest = read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")[0]
    assert manifest["source"] == "repository"


def test_publisher_cookie_risk_falls_back_to_legal_location(tmp_path: Path) -> None:
    safe_dir = Path(tempfile.mkdtemp(prefix="safe_fulltext_"))
    fallback = safe_dir / "fallback.xml"
    fallback.write_text("<article/>", encoding="utf-8")
    row = _approved("https://publisher.example/cookies_not_supported/article.pdf")
    row["fullTextXML"] = fallback.as_uri()
    row["skip_network_fallback"] = True
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True, prefer_formats=["pdf", "europe_pmc_xml"])

    assert result["actual_fulltexts"] == 1
    manifest = read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")[0]
    assert manifest["full_text_type"] == "europe_pmc_xml"
    manual = read_jsonl(tmp_path / "needs_manual_fulltext_review.jsonl")
    assert manual


def test_biorxiv_403_marks_blocked_by_server_retryable(tmp_path: Path, monkeypatch) -> None:
    row = _approved("https://www.biorxiv.org/content/biorxiv/early/2025/04/09/2025.04.05.647395.full.pdf")
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError("https://www.biorxiv.org/x", 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(br, "urlopen", fake_urlopen)
    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["failed_fulltext_fetches"] == 1
    failed = read_jsonl(tmp_path / "failed_fulltext_fetches.jsonl")[0]
    assert failed["retryable"] is True
    assert failed["recommended_next_action"] == "blocked_by_server"


def test_final_misuse_error_url_is_not_saved(tmp_path: Path, monkeypatch) -> None:
    row = _approved("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1")
    row["skip_network_fallback"] = True
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [row])

    class Response:
        url = "https://misuse.ncbi.nlm.nih.gov/error/abuse.shtml?orig_args=/pmc/articles/PMC1"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"<html>error</html>"

    monkeypatch.setattr(br, "urlopen", lambda *_args, **_kwargs: Response())
    result = collect_approved_legal_fulltexts(str(tmp_path), allow_download=True, yes=True)

    assert result["actual_fulltexts"] == 0
    assert read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl") == []
    assert read_jsonl(tmp_path / "failed_fulltext_fetches.jsonl")[0]["final_failure_reason"] == "forbidden_or_bypass_risk_final_url"


def test_pmcid_generates_pmc_xml_candidate() -> None:
    row = _approved("")
    row["pmcid"] = "PMC123456"
    candidates = resolve_legal_fulltext_candidates(row)

    assert any(candidate["full_text_type"] == "pmc_xml" and "PMC123456" in candidate["url"] for candidate in candidates)


def test_doi_can_use_unpaywall_candidate_fields(tmp_path: Path) -> None:
    pdf = tmp_path / "u.pdf"
    pdf.write_bytes(b"%PDF")
    row = _approved("")
    row["unpaywall_best_oa_location"] = {"url_for_pdf": pdf.as_uri(), "source": "repository", "license": "cc-by"}
    candidates = resolve_legal_fulltext_candidates(row)

    assert any(candidate["source"] == "repository" and candidate["url"] == pdf.as_uri() for candidate in candidates)


def test_retry_failed_only_processes_retryable(tmp_path: Path) -> None:
    retry_pdf = tmp_path / "retry.pdf"
    retry_pdf.write_bytes(b"%PDF retry")
    skip_pdf = tmp_path / "skip.pdf"
    skip_pdf.write_bytes(b"%PDF skip")
    retry_row = _approved(retry_pdf.as_uri(), "retry")
    skip_row = _approved(skip_pdf.as_uri(), "skip")
    skip_row["doi"] = "10.1/fulltext-skip"
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [retry_row, skip_row])
    _write_jsonl(
        tmp_path / "failed_fulltext_fetches.jsonl",
        [
            {"record_id": "retry", "doi": "10.1/fulltext", "title": "retry", "retryable": True},
            {"record_id": "skip", "doi": "10.1/fulltext-skip", "title": "skip", "retryable": False},
        ],
    )

    result = retry_failed_legal_fulltext_fetches(str(tmp_path), allow_download=True, yes=True)

    assert result["retry_attempted"] == 1
    assert result["retry_success"] == 1
    assert len(read_jsonl(tmp_path / "collected_fulltexts_manifest.jsonl")) == 1


def test_fulltext_collection_report_created(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "approved_for_download.jsonl", [])

    collect_approved_legal_fulltexts(str(tmp_path), allow_download=False, yes=False)

    assert (tmp_path / "fulltext_collection_report.md").exists()


def test_no_annotation_or_training_pipeline_added() -> None:
    source = Path("lit_agent").glob("*.py")
    joined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in source)
    for banned in ["annotation_pipeline", "auto_label", "training_data_extraction"]:
        assert banned not in joined
