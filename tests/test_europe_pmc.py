from lit_agent.query import QuerySpec
from lit_agent.sources import common, europe_pmc


def europe_pmc_item():
    return {
        "id": "12345",
        "source": "MED",
        "pmid": "12345",
        "pmcid": "PMC12345",
        "doi": "10.1000/epmc",
        "title": "Carbapenemase detection in clinical isolates",
        "pubYear": "2024",
        "pubType": "research article",
        "journalTitle": "Mock Europe PMC Journal",
        "authorString": "A Chen, B Wang",
        "abstractText": "Carbapenemase detection benchmark.",
        "isOpenAccess": "Y",
        "fullTextUrlList": {"fullTextUrl": [{"url": "https://europepmc.org/articles/PMC12345?pdf=render"}]},
    }


def test_europe_pmc_params_include_query_and_years():
    params = europe_pmc.build_europe_pmc_params(QuerySpec(keywords=["carbapenemase detection"], year_from=2020, year_to=2025, max_results=5))
    assert "carbapenemase detection" in params["query"]
    assert "FIRST_PDATE:[2020-01-01 TO 2025-12-31]" in params["query"]
    assert params["pageSize"] == "5"


def test_europe_pmc_item_to_metadata_and_candidate():
    record = europe_pmc.europe_pmc_item_to_metadata(europe_pmc_item(), QuerySpec(input_id="q"))
    assert record["source"] == "europe_pmc"
    assert record["doi"] == "10.1000/epmc"
    assert record["pmcid"] == "PMC12345"
    assert record["oa_status"] == "open"
    candidate = europe_pmc.candidates_from_records([record])[0]
    assert candidate["reason"] == "europe_pmc_fulltext_requires_download_round_confirmation"


def test_europe_pmc_dry_run_does_not_call_network():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    records = europe_pmc.search(QuerySpec(input_id="dry", keywords=["carbapenemase"], max_results=2), {"max_results_per_source": 2}, dry_run=True, session=FailingSession())
    assert records
    assert europe_pmc.LAST_SEARCH_INFO["network_calls"] == 0


def test_europe_pmc_http_error_records_failure(monkeypatch):
    failures = []

    class ErrorResponse:
        status_code = 429

        def json(self):
            return {}

    class ErrorSession:
        def get(self, *args, **kwargs):
            return ErrorResponse()

    monkeypatch.setattr(common.time, "sleep", lambda *_: None)
    monkeypatch.setattr(common, "write_failure", lambda reason, context: failures.append({"reason": reason, "context": context}))
    records = europe_pmc.search(QuerySpec(input_id="err", keywords=["carbapenemase"], max_results=1), {"max_results_per_source": 1}, dry_run=False, allow_network=True, session=ErrorSession())
    assert records == []
    assert failures[0]["reason"] == "europe_pmc metadata request failed"
