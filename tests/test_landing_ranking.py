from lit_agent.landing_page import select_landing_page_records


def test_landing_ranking_excludes_known_failed_and_mock_records():
    records = [
        {
            "decision": "allowed_for_future_download",
            "doi": "10.1000/mock",
            "title": "mock placeholder",
            "source": "unpaywall",
            "landing_url": "https://example.org/mock",
        },
        {
            "decision": "allowed_for_future_download",
            "doi": "10.1371/journal.pone.0245290",
            "source": "unpaywall",
            "landing_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245290",
        },
        {
            "decision": "allowed_for_future_download",
            "doi": "10.5812/jjm-146081",
            "source": "unpaywall",
            "landing_url": "https://brieflands.com/articles/jjm-146081.pdf",
        },
    ]
    failures = [
        {"reason": "doi landing page blocked direct pdf", "context": {"doi": "10.5812/jjm-146081", "reason": "doi_landing_direct_pdf_url_not_fetched"}}
    ]

    selected, stats = select_landing_page_records(records, max_results=5, failure_records=failures)

    assert [record["doi"] for record in selected] == ["10.1371/journal.pone.0245290"]
    assert stats["excluded_mock_count"] == 1
    assert stats["excluded_known_failed_count"] == 1
    assert stats["known_failure_breakdown"]["direct_pdf_guard_blocked"] == 1


def test_landing_ranking_prefers_recent_duplicate_with_same_priority():
    records = [
        {"decision": "allowed_for_future_download", "doi": "10.1/a", "source": "unpaywall", "landing_url": "https://old.publisher/a"},
        {"decision": "allowed_for_future_download", "doi": "10.1/a", "source": "unpaywall", "landing_url": "https://new.publisher/a"},
    ]

    selected, stats = select_landing_page_records(records, max_results=1)

    assert selected[0]["landing_url"] == "https://new.publisher/a"
    assert stats["duplicate_doi_count"] == 1
