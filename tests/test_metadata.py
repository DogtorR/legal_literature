from lit_agent.metadata import deduplicate_records, normalize_doi, normalize_title


def test_normalize_doi_and_title():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi:10.1000/XYZ") == "10.1000/xyz"
    assert normalize_title(" A Study:  of Detection! ") == "a study of detection"


def test_deduplicate_records_merges_doi_sources_and_oa_evidence():
    records = [
        {"title": "A", "doi": "10.1000/a", "source": "openalex", "oa_evidence": ["one"]},
        {"title": "A duplicate", "doi": "https://doi.org/10.1000/A", "source": "crossref", "oa_evidence": ["two"]},
        {"title": "Same Title", "publication_year": 2024, "source": "pubmed"},
        {"title": "Same Title", "publication_year": 2024, "source": "doaj"},
    ]
    unique = deduplicate_records(records)
    assert len(unique) == 2
    doi_record = next(item for item in unique if item.get("doi") == "10.1000/a")
    assert doi_record["sources"] == ["openalex", "crossref"]
    assert doi_record["oa_evidence"] == ["one", "two"]
