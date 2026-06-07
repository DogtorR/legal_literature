from pathlib import Path

from lit_agent.downloader import DownloadRequest, download_pdf, execute_download_plan


class FakeResponse:
    def __init__(self, content=b"%PDF-1.7\nbody", status_code=200, headers=None, url="https://example.org/a.pdf"):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/pdf"}
        self.url = url

    def iter_content(self, chunk_size=65536):
        yield self.content


class FakeSession:
    def __init__(self, head_response=None, get_response=None):
        self.head_response = head_response or FakeResponse(content=b"")
        self.get_response = get_response or FakeResponse()
        self.head_calls = 0
        self.get_calls = 0

    def head(self, *args, **kwargs):
        self.head_calls += 1
        return self.head_response

    def get(self, *args, **kwargs):
        self.get_calls += 1
        return self.get_response


def allowed_request(output_dir="downloads", pdf_url="https://example.org/a.pdf"):
    return DownloadRequest(
        input_id="q1",
        title="Legal OA Test",
        doi="10.1000/legal",
        source="unpaywall",
        pdf_url=pdf_url,
        landing_url="https://example.org/a",
        license="cc-by",
        oa_status="gold",
        legality_decision="allowed_for_future_download",
        evidence_sources=["unpaywall"],
        output_dir=output_dir,
    )


def test_download_without_allow_download_does_not_call_network(tmp_path):
    session = FakeSession()
    result = download_pdf(allowed_request(str(tmp_path)), dry_run=True, allow_download=False, session=session)
    assert result.status == "skipped_dry_run"
    assert session.head_calls == 0
    assert session.get_calls == 0


def test_download_without_allow_flag_does_not_call_network(tmp_path):
    session = FakeSession()
    result = download_pdf(allowed_request(str(tmp_path)), dry_run=False, allow_download=False, session=session)
    assert result.status == "skipped_dry_run"
    assert session.head_calls == 0
    assert session.get_calls == 0


def test_allow_download_preflight_fail_does_not_download(tmp_path):
    session = FakeSession(head_response=FakeResponse(status_code=403))
    result = download_pdf(allowed_request(str(tmp_path)), dry_run=False, allow_download=True, session=session)
    assert result.status == "blocked"
    assert result.reason == "access_requires_auth_or_paywall"
    assert session.get_calls == 0


def test_successful_fake_download_writes_file_manifest_and_sha256(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = FakeSession()
    result = execute_download_plan([allowed_request("downloads")], dry_run=False, allow_download=True, session=session)[0]
    assert result.status == "downloaded"
    assert result.sha256
    assert Path(result.local_path).exists()
    assert Path(result.local_path).read_bytes().startswith(b"%PDF")
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
    assert '"status": "downloaded"' in manifest
    assert '"sha256":' in manifest


def test_non_pdf_content_is_removed_and_not_manifested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = FakeSession(get_response=FakeResponse(content=b"not pdf", headers={"Content-Type": "text/html"}))
    result = execute_download_plan([allowed_request("downloads")], dry_run=False, allow_download=True, session=session)[0]
    assert result.status == "failed"
    assert not list((tmp_path / "downloads").glob("*.pdf"))
    manifest_path = tmp_path / "manifest.jsonl"
    assert not manifest_path.exists() or '"status": "downloaded"' not in manifest_path.read_text(encoding="utf-8")


def test_max_downloads_three_is_honored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = [allowed_request("downloads", f"https://example.org/{idx}.pdf") for idx in range(5)]
    results = execute_download_plan(requests[:3], dry_run=True, allow_download=False)
    assert len(results) == 3
