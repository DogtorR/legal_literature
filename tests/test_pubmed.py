from lit_agent.query import QuerySpec
from lit_agent.sources import common, pubmed


def pubmed_summary():
    return {
        "uid": "12345",
        "title": "Carbapenemase detection assay",
        "pubdate": "2024 Jan",
        "pubtype": ["Journal Article"],
        "fulljournalname": "Mock PubMed Journal",
        "authors": [{"name": "A Chen"}],
        "articleids": [
            {"idtype": "doi", "value": "10.1000/pubmed"},
            {"idtype": "pmc", "value": "PMC12345"},
        ],
    }


def test_pubmed_esearch_params_include_terms_and_years():
    params = pubmed.build_pubmed_esearch_params(QuerySpec(keywords=["carbapenemase detection"], year_from=2020, year_to=2025, max_results=5))
    assert "carbapenemase detection" in params["term"]
    assert '"2020"[PDAT] : "2025"[PDAT]' in params["term"]
    assert params["retmax"] == "5"


def test_pubmed_summary_to_metadata_and_candidate():
    record = pubmed.pubmed_summary_to_metadata(pubmed_summary(), QuerySpec(input_id="q"))
    assert record["source"] == "pubmed"
    assert record["doi"] == "10.1000/pubmed"
    assert record["pmcid"] == "PMC12345"
    candidate = pubmed.candidates_from_records([record])[0]
    assert candidate["reason"] == "pubmed_pmcid_requires_pmc_confirmation"


def test_pubmed_dry_run_does_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    records = pubmed.search(QuerySpec(input_id="dry", keywords=["carbapenemase"], max_results=2), {"max_results_per_source": 2}, dry_run=True, session=FailingSession())
    assert records
    assert pubmed.LAST_SEARCH_INFO["network_calls"] == 0


def test_pubmed_http_error_records_failure(monkeypatch):
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
    records = pubmed.search(QuerySpec(input_id="err", keywords=["carbapenemase"], max_results=1), {"max_results_per_source": 1}, dry_run=False, allow_network=True, session=ErrorSession())
    assert records == []
    assert failures[0]["reason"] == "pubmed metadata request failed"
