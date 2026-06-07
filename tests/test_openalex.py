import pytest

from lit_agent.filters import apply_query_filters
from lit_agent.query import QuerySpec
from lit_agent.sources import openalex


def sample_work():
    return {
        "id": "https://openalex.org/W123",
        "ids": {"doi": "https://doi.org/10.1000/ABC", "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345"},
        "title": "Rapid carbapenemase detection",
        "publication_year": 2024,
        "type": "article",
        "authorships": [{"author": {"display_name": "A Chen"}}],
        "abstract_inverted_index": {"Rapid": [0], "detection": [2], "carbapenemase": [1]},
        "primary_location": {
            "landing_page_url": "https://example.org/work",
            "pdf_url": None,
            "license": None,
            "source": {"display_name": "Mock Journal"},
        },
        "best_oa_location": {
            "landing_page_url": "https://example.org/oa",
            "pdf_url": "https://example.org/oa.pdf",
            "license": "cc-by",
        },
        "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://example.org/oa"},
        "relevance_score": 12.5,
    }


def test_build_openalex_params_includes_search_years_type_and_limit():
    spec = QuerySpec(keywords=["carbapenemase detection"], year_from=2020, year_to=2025, publication_type="journal-article", max_results=5)
    params = openalex.build_openalex_params(spec)
    assert params["search"] == "carbapenemase detection"
    assert "from_publication_date:2020-01-01" in params["filter"]
    assert "to_publication_date:2025-12-31" in params["filter"]
    assert "type:article" in params["filter"]
    assert params["per-page"] == "5"


def test_build_openalex_params_prefers_doi_filter():
    spec = QuerySpec(doi="https://doi.org/10.1000/ABC", keywords=["ignored"])
    params = openalex.build_openalex_params(spec)
    assert params["filter"] == "doi:10.1000/abc"
    assert "search" not in params


def test_restore_abstract_from_inverted_index():
    assert openalex.restore_abstract({"world": [1], "hello": [0]}) == "hello world"


def test_work_to_metadata_normalizes_fields_and_candidate_url():
    spec = QuerySpec(input_id="q1", keywords=["carbapenemase detection"])
    record = openalex.openalex_work_to_metadata(sample_work(), spec)
    assert record["source"] == "openalex"
    assert record["doi"] == "10.1000/abc"
    assert record["pmid"] == "12345"
    assert record["openalex_id"] == "https://openalex.org/W123"
    assert record["abstract"] == "Rapid carbapenemase detection"
    assert record["journal"] == "Mock Journal"
    assert record["pdf_url_candidate"] == "https://example.org/oa.pdf"


def test_openalex_record_filters_keywords_include_exclude_and_years():
    spec = QuerySpec(year_from=2020, year_to=2025, keywords=["carbapenemase"], include_terms=["detection"], exclude_terms=["review"])
    keep = openalex.openalex_work_to_metadata(sample_work(), spec)
    drop = dict(keep, title="Review of carbapenemase detection", abstract="review")
    filtered = apply_query_filters([keep, drop], spec)
    assert len(filtered) == 1
    assert filtered[0]["matched_keywords"] == ["carbapenemase", "detection"]


def test_best_oa_location_becomes_candidate_not_download():
    spec = QuerySpec(input_id="q1")
    record = openalex.openalex_work_to_metadata(sample_work(), spec)
    candidate = openalex.candidate_from_record(record)
    assert candidate is not None
    assert candidate["candidate_url"] == "https://example.org/oa.pdf"
    assert candidate["reason"] == "openalex_oa_location_requires_confirmation"


def test_dry_run_does_not_call_network_session():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    records = openalex.search(QuerySpec(input_id="dry", keywords=["carbapenemase"], max_results=2), {"max_results_per_source": 2}, dry_run=True, session=FailingSession())
    assert records
    assert openalex.LAST_SEARCH_INFO["network_calls"] == 0


def test_http_error_is_written_to_failures(monkeypatch):
    failures = []

    class ErrorResponse:
        status_code = 500

        def json(self):
            return {}

    class ErrorSession:
        def get(self, *args, **kwargs):
            return ErrorResponse()

    monkeypatch.setattr(openalex.time, "sleep", lambda *_: None)
    monkeypatch.setattr(openalex, "write_failure", lambda reason, context: failures.append({"reason": reason, "context": context}))
    records = openalex.search(QuerySpec(input_id="err", keywords=["carbapenemase"], max_results=1), {"max_results_per_source": 1}, dry_run=False, allow_network=True, session=ErrorSession())
    assert records == []
    assert failures[0]["reason"] == "openalex metadata request failed"
    assert "OpenAlex HTTP 500" in failures[0]["context"]["error"]
