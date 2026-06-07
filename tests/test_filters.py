from lit_agent.filters import apply_query_filters, apply_record_scope_filters, exclude_terms_block, include_terms_match, publication_date_in_range, record_matches_scope, text_matches_keywords, year_in_range
from lit_agent.query import QuerySpec
from lit_agent.topic_guard import apply_topic_guard


def test_year_range_filter_keeps_missing_year_with_reason():
    spec = QuerySpec(year_from=2020, year_to=2025)
    assert year_in_range({"publication_year": 2022}, spec) == (True, None)
    assert year_in_range({"publication_year": 2019}, spec) == (False, "year_before_range")
    assert year_in_range({"publication_year": None}, spec) == (True, "missing_year")


def test_keywords_include_and_exclude_filters():
    spec = QuerySpec(keywords=["carbapenemase detection"], include_terms=["detection"], exclude_terms=["review"])
    record = {"title": "Carbapenemase detection assay", "abstract": "Fast detection", "journal": "Diagnostics"}
    blocked = {"title": "Review of carbapenemase detection", "abstract": "review", "journal": "Reviews"}
    assert text_matches_keywords(record, spec)
    assert include_terms_match(record, spec)
    assert not exclude_terms_block(record, spec)
    assert exclude_terms_block(blocked, spec)


def test_apply_query_filters_combines_rules_and_preserves_missing_year():
    spec = QuerySpec(year_from=2020, year_to=2025, keywords=["carbapenemase"], include_terms=["detection"], exclude_terms=["review"])
    records = [
        {"title": "Carbapenemase detection", "publication_year": 2021},
        {"title": "Carbapenemase review detection", "publication_year": 2021},
        {"title": "Carbapenemase detection without year", "publication_year": None},
        {"title": "Unrelated", "publication_year": 2021},
    ]
    filtered = apply_query_filters(records, spec)
    assert len(filtered) == 2
    assert filtered[1]["filter_reasons"] == ["missing_year"]


def test_topic_guard_mode_can_relax_exact_keyword_phrase_filter():
    spec = QuerySpec(keywords=["CRISPR detection"], include_terms=["CRISPR", "detection"])
    records = [
        {"title": "CRISPR-based environmental detection of Burkholderia pseudomallei"},
        {"title": "CRISPR base editor screening identifies MEN1 mutations"},
        {"title": "CRISPR genome editing in cells"},
        {"title": "Rapid pathogen detection by PCR"},
    ]
    filtered = apply_query_filters(records, spec, require_keyword_match=False)
    kept = apply_topic_guard(filtered, {"enabled": True, "type": "crispr_detection"})
    assert [record["title"] for record in kept] == [
        "CRISPR-based environmental detection of Burkholderia pseudomallei",
    ]


def test_publication_month_and_journal_scope_are_strict():
    spec = QuerySpec(publication_date_from="2026-05-01", publication_date_to="2026-05-31")
    assert publication_date_in_range({"publication_date": "2026-05-12"}, spec) == (True, None)
    assert publication_date_in_range({"publication_date": "2026-06-01"}, spec) == (False, "publication_date_after_range")
    assert publication_date_in_range({"publication_year": 2026}, spec) == (False, "publication_date_precision_too_low")

    records = [
        {"title": "A", "journal": "Nature Communications", "publication_date": "2026-05-12"},
        {"title": "B", "journal": "Nature", "publication_date": "2026-05-12"},
        {"title": "C", "journal": "Nature Communications", "publication_date": "2026-06-01"},
    ]
    kept, stats = apply_record_scope_filters(
        records,
        {
            "journals": ["Nature Communications"],
            "publication_date_from": "2026-05-01",
            "publication_date_to": "2026-05-31",
            "require_journal_match": True,
            "require_date_match": True,
        },
    )
    assert [record["title"] for record in kept] == ["A"]
    assert stats["scope_removed_count"] == 2


def test_nature_family_scope_matches_portfolio_journals():
    scope = {
        "journals": ["Nature", "Nature Communications", "Communications Chemistry"],
        "journal_family": "nature",
        "require_journal_family_match": True,
    }

    assert record_matches_scope({"journal": "Nature Communications"}, scope)[0] is True
    assert record_matches_scope({"journal": "Communications Chemistry"}, scope)[0] is True
    assert record_matches_scope({"journal": "npj Quantum Information"}, scope)[0] is True
    assert record_matches_scope({"journal": "Science Advances"}, scope) == (False, "journal_family_not_in_scope")
