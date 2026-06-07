from lit_agent.agent_graph import coverage_check_node
from lit_agent.tool_registry import diagnose_pubmed_query_recall, search_pubmed_tool


def test_search_tool_returns_recall_envelope():
    result = search_pubmed_tool("CRISPR detection", year_from=2025, year_to=2026, max_results=5, allow_network=False)
    assert result["tool"] == "search_pubmed"
    assert result["query"] == "CRISPR detection"
    assert result["source"] == "pubmed"
    data = result["data"]
    for key in ["records", "total_count", "retrieved_count", "page_count", "has_more", "next_cursor", "query_used", "date_filter"]:
        assert key in data


def test_diagnose_pubmed_query_recall_schema(monkeypatch):
    def fake_search(query, year_from=None, year_to=None, max_results=20, **kwargs):
        return {
            "status": "partial",
            "tool": "search_pubmed",
            "query": query,
            "source": "pubmed",
            "data": {
                "records": [{"title": "first"}, {"title": "last"}],
                "total_count": 10,
                "retrieved_count": 2,
                "page_count": 1,
                "has_more": True,
                "source_info": {"errors": []},
            },
            "error": "",
        }

    monkeypatch.setattr("lit_agent.tool_registry.search_pubmed_tool", fake_search)
    result = diagnose_pubmed_query_recall("CRISPR detection", 2025, 2026, max_results=5000)
    assert result["query"] == "CRISPR detection"
    assert result["total_count"] == 10
    assert result["retrieved_count"] == 2
    assert result["page_count"] == 1
    assert result["has_more"] is True
    assert result["first_5_titles"] == ["first", "last"]


def test_coverage_partial_when_total_exceeds_retrieved():
    state = coverage_check_node(
        {
            "query_list": ["CRISPR detection"],
            "search_log": [
                {
                    "query": "CRISPR detection",
                    "source": "pubmed",
                    "status": "partial",
                    "total_count": 2700,
                    "retrieved_count": 500,
                    "result_count": 500,
                    "page_count": 5,
                    "has_more": True,
                    "max_results_per_query": 500,
                }
            ],
            "topic_records": [],
            "oa_audit_records": [],
            "failures": [],
            "search_attempt_count": 1,
            "max_search_attempts": 1,
            "oa_attempt_count": 0,
            "max_oa_attempts": 1,
        }
    )
    assert state["coverage"]["status"] == "partial"
    assert "pagination_or_limit_insufficient" in state["coverage"]["reasons"]
