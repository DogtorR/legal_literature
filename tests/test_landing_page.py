from lit_agent.landing_page import classify_landing_candidate, extract_pdf_candidates_from_html, normalize_doi_url, resolve_landing_page, select_landing_page_records


def test_normalize_doi_url():
    assert normalize_doi_url("https://doi.org/10.1000/ABC") == "https://doi.org/10.1000/abc"


def test_extracts_citation_pdf_url_and_alternate_pdf():
    html = """
    <html><head>
    <meta name="citation_pdf_url" content="/article.pdf">
    <link rel="alternate" type="application/pdf" href="https://example.org/alt.pdf">
    </head></html>
    """
    urls = extract_pdf_candidates_from_html(html, "https://example.org/page")
    assert "https://example.org/article.pdf" in urls
    assert "https://example.org/alt.pdf" in urls


def test_extracts_relative_anchor_pdf():
    html = '<a href="../pdf/file.pdf">PDF</a>'
    assert extract_pdf_candidates_from_html(html, "https://example.org/articles/page") == ["https://example.org/pdf/file.pdf"]


def test_forbidden_url_is_blocked():
    candidate = classify_landing_candidate("https://sci-hub.example/a.pdf", {"doi": "10.1000/a"})
    assert candidate["planned_status"] == "blocked"


def test_landing_candidate_requires_confirmation_without_oa():
    candidate = classify_landing_candidate("https://example.org/a.pdf", {"doi": "10.1000/a"})
    assert candidate["decision"] == "doi_landing_candidate_needs_confirmation"


def test_landing_candidate_with_oa_evidence_goes_to_plan():
    candidate = classify_landing_candidate("https://example.org/a.pdf", {"doi": "10.1000/a", "decision": "allowed_for_future_download", "source": "unpaywall"})
    assert candidate["planned_status"] == "candidate_direct_pdf_discovered_not_downloaded_round_011"


def test_dry_run_does_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    result = resolve_landing_page({"doi": "10.1000/a"}, dry_run=True, allow_network=False, session=FailingSession())
    assert result.network_calls == 0
    assert result.status == "planned"


def test_allow_network_false_does_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    result = resolve_landing_page({"doi": "10.1000/a"}, dry_run=False, allow_network=False, session=FailingSession())
    assert result.network_calls == 0


def test_direct_pdf_landing_url_is_blocked_without_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("direct PDF landing URL should not be fetched")

    result = resolve_landing_page({"doi": "10.1234/a", "landing_url": "https://publisher.test/article.pdf"}, dry_run=False, allow_network=True, session=FailingSession())
    assert result.status == "blocked"
    assert result.reason == "doi_landing_direct_pdf_url_not_fetched"
    assert result.network_calls == 0


def test_landing_redirect_to_pdf_is_blocked_before_body_read():
    class RedirectResponse:
        status_code = 302
        url = "https://doi.org/10.1234/a"
        headers = {"Location": "https://publisher.test/article.pdf"}

    class FakeSession:
        def get(self, *args, **kwargs):
            return RedirectResponse()

    result = resolve_landing_page({"doi": "10.1234/a"}, dry_run=False, allow_network=True, session=FakeSession())
    assert result.status == "blocked"
    assert result.reason == "doi_landing_redirect_to_pdf_not_fetched"


def test_login_paywall_text_blocks():
    class FakeResponse:
        status_code = 200
        url = "https://example.org/a"
        text = "<html>institutional login required</html>"

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    result = resolve_landing_page({"doi": "10.1000/a"}, dry_run=False, allow_network=True, session=FakeSession())
    assert result.status == "blocked"
    assert result.reason == "doi_landing_blocked_requires_login_or_paywall"


def test_select_landing_records_prioritizes_real_oa_and_excludes_mock():
    records = [
        {
            "decision": "allowed_for_future_download",
            "doi": "10.1000/mock",
            "title": "mock article",
            "source": "unpaywall",
            "landing_url": "https://example.org/mock",
        },
        {
            "doi": "10.5555/needs-confirmation",
            "source": "crossref",
            "landing_url": "https://publisher.test/articles/needs-confirmation",
        },
        {
            "decision": "allowed_for_future_download",
            "doi": "10.1038/s41586-020-2649-2",
            "source": "unpaywall",
            "landing_url": "https://www.nature.com/articles/s41586-020-2649-2",
            "evidence_sources": ["unpaywall"],
        },
        {
            "doi": "10.1093/nar/gkaa123",
            "source": "metadata",
            "unpaywall_is_oa": True,
            "landing_url": "https://academic.oup.com/nar/article/landing",
        },
    ]

    selected, stats = select_landing_page_records(records, max_results=3)

    assert [record["doi"] for record in selected] == [
        "10.1038/s41586-020-2649-2",
        "10.1093/nar/gkaa123",
        "10.5555/needs-confirmation",
    ]
    assert stats["excluded_mock_count"] == 1
    assert stats["priority_breakdown"] == {
        "allowed_oa_audit": 1,
        "metadata_oa_status": 1,
        "other_needs_confirmation": 1,
    }
