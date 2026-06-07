import json
from pathlib import Path

from lit_agent.downloader import DownloadRequest, execute_download_plan


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/pdf"}
    url = "https://example.org/manifest.pdf"

    def iter_content(self, chunk_size=65536):
        yield b"%PDF-1.7\nbody"


class FakeSession:
    def head(self, *args, **kwargs):
        return FakeResponse()

    def get(self, *args, **kwargs):
        return FakeResponse()


def test_downloaded_manifest_record_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = DownloadRequest(
        input_id="q1",
        title="Manifest Test",
        doi="10.1000/manifest",
        pmid="123",
        openalex_id="W1",
        publication_year=2024,
        publication_type="journal-article",
        journal="Journal",
        authors=["A. Author"],
        keywords_matched=["carbapenemase"],
        source="unpaywall",
        pdf_url="https://example.org/manifest.pdf",
        landing_url="https://example.org/manifest",
        license="cc-by",
        oa_status="gold",
        legality_decision="allowed_for_future_download",
        evidence_sources=["unpaywall"],
        output_dir="downloads",
    )
    result = execute_download_plan([request], dry_run=False, allow_download=True, session=FakeSession())[0]
    record = json.loads(Path("manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    required = {
        "input_id",
        "title",
        "doi",
        "pmid",
        "openalex_id",
        "publication_year",
        "publication_type",
        "journal",
        "authors",
        "keywords_matched",
        "source",
        "pdf_url",
        "landing_url",
        "license",
        "oa_status",
        "oa_evidence",
        "is_legal_oa",
        "local_path",
        "sha256",
        "downloaded_at",
        "status",
        "reason",
    }
    assert required <= set(record)
    assert record["status"] == "downloaded"
    assert record["is_legal_oa"] is True
    assert record["sha256"]
    assert Path(record["local_path"]).exists()
    assert result.status == "downloaded"
