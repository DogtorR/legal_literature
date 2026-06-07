from lit_agent.query import QuerySpec
from lit_agent.sources import common, crossref


def crossref_item():
    return {
        "DOI": "10.1000/ABC",
        "title": ["Rapid carbapenemase detection"],
        "container-title": ["Mock Journal"],
        "published-print": {"date-parts": [[2024, 1, 1]]},
        "type": "journal-article",
        "author": [{"given": "A", "family": "Chen"}],
        "URL": "https://doi.org/10.1000/abc",
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        "link": [{"URL": "https://example.org/article.pdf", "content-type": "application/pdf"}],
        "score": 3.2,
    }


def test_crossref_query_params_include_years_and_rows():
    spec = QuerySpec(keywords=["carbapenemase detection"], year_from=2020, year_to=2025, max_results=5)
    url, params = crossref.build_crossref_request(spec)
    assert url == crossref.CROSSREF_WORKS_ENDPOINT
    assert params["query.bibliographic"] == "carbapenemase detection"
    assert params["rows"] == "5"
    assert "from-pub-date:2020-01-01" in params["filter"]
    assert "until-pub-date:2025-12-31" in params["filter"]


def test_crossref_doi_uses_work_endpoint():
    url, params = crossref.build_crossref_request(QuerySpec(doi="https://doi.org/10.1000/ABC"))
    assert url.endswith("/10.1000/abc")
    assert params == {}


def test_crossref_item_to_metadata_and_candidate():
    record = crossref.crossref_item_to_metadata(crossref_item(), QuerySpec(input_id="q"))
    assert record["source"] == "crossref"
    assert record["doi"] == "10.1000/abc"
    assert record["publication_year"] == 2024
    assert record["authors"] == ["A Chen"]
    candidate = crossref.candidates_from_records([record])[0]
    assert candidate["reason"] == "crossref_pdf_link_requires_confirmation"


def test_crossref_dry_run_does_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    records = crossref.search(QuerySpec(input_id="dry", keywords=["carbapenemase"], max_results=2), {"max_results_per_source": 2}, dry_run=True, session=FailingSession())
    assert records
    assert crossref.LAST_SEARCH_INFO["network_calls"] == 0


def test_crossref_http_error_records_failure(monkeypatch):
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
    records = crossref.search(QuerySpec(input_id="err", keywords=["carbapenemase"], max_results=1), {"max_results_per_source": 1}, dry_run=False, allow_network=True, session=ErrorSession())
    assert records == []
    assert failures[0]["reason"] == "crossref metadata request failed"
