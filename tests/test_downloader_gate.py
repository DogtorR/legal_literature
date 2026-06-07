from lit_agent.downloader import DownloadRequest, download_pdf, final_preflight_check


def allowed_request(source="unpaywall", evidence_sources=None, pdf_url="https://example.org/a.pdf"):
    return DownloadRequest(
        source=source,
        pdf_url=pdf_url,
        landing_url="https://example.org/a",
        legality_decision="allowed_for_future_download",
        evidence_sources=evidence_sources or [source],
        license="cc-by",
    )


def test_allowed_unpaywall_and_europe_pmc_plan_dry_run():
    assert download_pdf(allowed_request("unpaywall"), dry_run=True).status == "skipped_dry_run"
    assert download_pdf(allowed_request("europe_pmc"), dry_run=True).status == "skipped_dry_run"


def test_openalex_crossref_pubmed_only_are_blocked():
    assert final_preflight_check(allowed_request("openalex", ["openalex"])).status == "blocked"
    assert final_preflight_check(allowed_request("crossref", ["crossref"])).status == "blocked"
    assert final_preflight_check(allowed_request("pubmed", ["pubmed"])).status == "blocked"


def test_forbidden_login_missing_and_doi_urls_blocked():
    assert final_preflight_check(allowed_request(pdf_url="https://sci-hub.example/a.pdf")).status == "blocked"
    request = allowed_request()
    request.risk_flags = ["cookie reuse"]
    assert final_preflight_check(request).status == "blocked"
    assert final_preflight_check(allowed_request(pdf_url="")).reason == "missing_pdf_url"
    assert final_preflight_check(allowed_request(pdf_url="https://doi.org/10.1000/a")).reason == "doi_landing_page_is_not_pdf_url"


def test_nature_cookie_not_supported_query_is_not_access_risk():
    request = allowed_request(pdf_url="https://www.nature.com/articles/s41467-026-73286-8_reference.pdf?error=cookies_not_supported&code=abc")
    result = final_preflight_check(request)
    assert result.status == "preflight_passed"


def test_dry_run_and_allow_download_false_do_not_call_network_session():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    result = download_pdf(allowed_request(), dry_run=True, allow_download=False, session=FailingSession())
    assert result.status == "skipped_dry_run"
