import argparse

from lit_agent.cli import _selected_sources
from lit_agent.metadata import deduplicate_records


def test_source_all_expands_to_allowed_round_004_sources():
    args = argparse.Namespace(source=["all"])
    assert _selected_sources(args, {}) == ["openalex", "crossref", "europe_pmc", "pubmed", "unpaywall"]


def test_multisource_deduplicates_doi_and_preserves_sources():
    records = [
        {"title": "A", "doi": "10.1000/shared", "source": "openalex", "sources": ["openalex"]},
        {"title": "A", "doi": "https://doi.org/10.1000/shared", "source": "crossref", "sources": ["crossref"], "license": "license-url"},
        {"title": "A", "doi": "10.1000/shared", "source": "europe_pmc", "sources": ["europe_pmc"], "pmcid": "PMC1"},
        {"title": "B", "pmid": "123", "source": "pubmed", "sources": ["pubmed"]},
    ]
    unique = deduplicate_records(records)
    assert len(unique) == 2
    shared = next(record for record in unique if record.get("doi") == "10.1000/shared")
    assert shared["sources"] == ["openalex", "crossref", "europe_pmc"]
    assert shared["license"] == "license-url"
    assert shared["pmcid"] == "PMC1"
