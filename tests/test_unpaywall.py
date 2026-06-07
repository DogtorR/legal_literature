from lit_agent.legality import assess_unpaywall_oa
from lit_agent.sources import common, unpaywall


def oa_payload():
    return {
        "doi": "10.1000/UPW",
        "is_oa": True,
        "oa_status": "gold",
        "best_oa_location": {
            "url": "https://example.org/article",
            "url_for_pdf": "https://example.org/article.pdf",
            "license": "cc-by",
            "host_type": "publisher",
            "version": "publishedVersion",
            "evidence": "oa journal",
        },
        "oa_locations": [{"url": "https://example.org/article", "url_for_pdf": "https://example.org/article.pdf"}],
    }


def test_unpaywall_url_and_params():
    assert unpaywall.build_unpaywall_url("https://doi.org/10.1000/UPW").endswith("10.1000%2Fupw")
    assert unpaywall.build_unpaywall_params({"contact_email": "author@example.org"}) == {"email": "author@example.org"}


def test_dry_run_and_allow_network_false_do_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    result = unpaywall.check_doi("10.1000/upw", {}, dry_run=True, session=FailingSession())
    assert result["reason"] == "unpaywall_dry_run_not_checked"
    result = unpaywall.check_doi("10.1000/upw", {}, dry_run=False, allow_network=False, session=FailingSession())
    assert result["reason"] == "unpaywall_dry_run_not_checked"


def test_parse_oa_true_response_and_candidate_no_download():
    result = unpaywall.parse_unpaywall_response(oa_payload(), query_id="q")
    assert result["doi"] == "10.1000/upw"
    assert result["is_legal_oa_candidate"] is True
    assert result["is_download_allowed_now"] is False
    candidate = unpaywall.candidate_from_result(result)
    assert candidate["pdf_url_candidate"] == "https://example.org/article.pdf"
    assert candidate["is_download_allowed_now"] is False


def test_parse_pmcid_from_unpaywall_oa_locations():
    payload = oa_payload()
    payload["best_oa_location"] = {
        "url": "https://doi.org/10.1126/sciadv.aed1757",
        "license": "cc-by-nc",
        "host_type": "publisher",
        "version": "publishedVersion",
    }
    payload["oa_locations"] = [
        {"url": "https://doi.org/10.1126/sciadv.aed1757"},
        {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13196775/", "pmh_id": "oai:pubmedcentral.nih.gov:13196775"},
    ]
    result = unpaywall.parse_unpaywall_response(payload, query_id="q")
    candidate = unpaywall.candidate_from_result(result, {"pmid": "42172311"})
    assert result["pmcid"] == "PMC13196775"
    assert candidate["pmcid"] == "PMC13196775"


def test_parse_oa_false_response():
    result = unpaywall.parse_unpaywall_response({"doi": "10.1000/no", "is_oa": False, "oa_status": "closed"}, query_id="q")
    assert result["is_legal_oa_candidate"] is False
    assert result["reason"] == "unpaywall_no_confirmed_oa_location"


def test_missing_doi_candidate():
    candidate = unpaywall.missing_doi_candidate({"title": "No DOI"})
    assert candidate["reason"] == "missing_doi_for_unpaywall_check"
    assert candidate["is_download_allowed_now"] is False


def test_forbidden_url_detection_in_unpaywall_assessment():
    decision = assess_unpaywall_oa({"is_oa": True, "url_for_pdf": "https://sci-hub.example/test.pdf"})
    assert decision["reason"] == "forbidden_url_detected_in_unpaywall_result"
    assert decision["is_download_allowed_now"] is False


def test_http_error_records_failure(monkeypatch):
    failures = []

    class ErrorResponse:
        status_code = 500

        def json(self):
            return {}

    class ErrorSession:
        def get(self, *args, **kwargs):
            return ErrorResponse()

    monkeypatch.setattr(common.time, "sleep", lambda *_: None)
    monkeypatch.setattr(common, "write_failure", lambda reason, context: failures.append({"reason": reason, "context": context}))
    monkeypatch.setattr(unpaywall, "write_failure", lambda reason, context: failures.append({"reason": reason, "context": context}))
    result = unpaywall.check_doi("10.1000/upw", {}, dry_run=False, allow_network=True, session=ErrorSession())
    assert result["reason"] == "unpaywall_check_failed"
    assert any(item["reason"] == "unpaywall oa check failed" for item in failures)
