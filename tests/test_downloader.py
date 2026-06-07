from lit_agent.downloader import DownloadRequest, download_pdf


def test_downloader_dry_run_skips_without_download():
    request = DownloadRequest(
        source="unpaywall",
        pdf_url="https://example.org/legal.pdf",
        legality_decision="allowed_for_future_download",
        evidence_sources=["unpaywall"],
    )
    result = download_pdf(request, dry_run=True)
    assert result.status == "skipped_dry_run"
    assert result.local_path == ""
    assert result.sha256 == ""


def test_downloader_blocks_without_legality_decision():
    result = download_pdf(DownloadRequest(pdf_url="https://example.org/legal.pdf"), dry_run=False)
    assert result.status == "blocked"
    assert result.reason == "legality_audit_decision_not_allowed"


def test_downloader_round_002_still_blocks_real_downloads():
    result = download_pdf(DownloadRequest(pdf_url="https://example.org/legal.pdf"), dry_run=False)
    assert result.status == "blocked"
