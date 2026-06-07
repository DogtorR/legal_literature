from lit_agent.failure_analysis import classify_failure, summarize_dedupe, summarize_failures


def test_classifies_common_failure_categories():
    assert classify_failure({"reason": "round_012 task downloads check command failed", "context": {"error": "ValueError: Unacceptable pattern: ''"}}) == "task_command_defect"
    assert classify_failure({"reason": "doi landing page blocked direct pdf", "context": {"reason": "doi_landing_direct_pdf_url_not_fetched"}}) == "landing_direct_pdf_guard"
    assert classify_failure({"reason": "download preflight or retrieval failed", "context": {"reason": "missing_pdf_url", "pdf_url": ""}}) == "missing_pdf_url"
    assert classify_failure({"reason": "unpaywall oa check failed", "context": {"doi": "10.1234/a", "error": "HTTP 404"}}) == "unpaywall_404"


def test_summarize_failures_includes_all_categories():
    summary = summarize_failures(
        [
            {"reason": "doi landing page blocked direct pdf", "context": {"doi": "10.1234/a"}},
            {"reason": "doi landing page blocked direct pdf", "context": {"doi": "10.1234/b"}},
        ]
    )

    assert summary["total_rows"] == 2
    assert summary["categories"]["landing_direct_pdf_guard"]["count"] == 2
    assert "recommended_action" in summary["categories"]["landing_direct_pdf_guard"]


def test_dedupe_summary_counts_duplicate_rows():
    records = [
        {"doi": "10.1/a", "candidate_url": "https://publisher/a", "reason": "missing_pdf_url", "source": "unpaywall"},
        {"doi": "10.1/a", "candidate_url": "https://publisher/a", "reason": "missing_pdf_url", "source": "unpaywall"},
        {"doi": "10.1/b", "candidate_url": "https://publisher/b", "reason": "other", "source": "unpaywall"},
    ]

    summary = summarize_dedupe(records, key_name="candidate")

    assert summary["total_rows"] == 3
    assert summary["unique_keys"] == 2
    assert summary["duplicate_rows"] == 1
