from lit_agent.agent_graph import query_expansion_node
from lit_agent.crispr_scope import PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY
from lit_agent.mcp_server import run_crispr_broad_corpus_agent
from lit_agent import tool_registry as tr


def test_primary_broad_query_present():
    state = query_expansion_node({"user_request": "broad CRISPR corpus", "task": {}, "options": {"mode": "crispr_broad_corpus"}})
    query = state["query_list"][0]
    assert query == PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY
    for term in ["CRISPR", "Cas12", "Cas13", "structure", "mechanism", "detection"]:
        assert term in query


def test_scope_guard_includes_detection():
    result = tr.scope_guard_records_tool(
        [{"title": "CRISPR Cas12 detection assay for viral nucleic acid", "abstract": "A biosensor diagnostic platform."}]
    )
    assert len(result["data"]["included_records"]) == 1


def test_scope_guard_includes_structure():
    result = tr.scope_guard_records_tool(
        [{"title": "Structural mechanism of Cas12a collateral cleavage", "abstract": "Cryo-EM reveals PAM recognition by CRISPR-Cas."}]
    )
    assert len(result["data"]["included_records"]) == 1


def test_scope_guard_excludes_pure_editing():
    result = tr.scope_guard_records_tool(
        [{"title": "CRISPR crop genome editing knockout application", "abstract": "Plant breeding by gene editing."}]
    )
    assert len(result["data"]["excluded_records"]) == 1


def test_scope_guard_retains_detection_even_with_editing_terms():
    result = tr.scope_guard_records_tool(
        [
            {
                "title": "CRISPR gene editing detection assay",
                "abstract": "A Cas12 biosensor detects genome editing outcomes in nucleic acid samples.",
            }
        ]
    )
    assert len(result["data"]["included_records"]) == 1
    record = result["data"]["included_records"][0]
    assert record["scope_reason"] == "mixed_editing_with_detection_retained"
    assert record["is_detection_related"] is True
    assert record["matched_exclude_terms"]


def test_scope_guard_manual_review_when_mixed():
    result = tr.scope_guard_records_tool(
        [{"title": "CRISPR gene editing mechanism in Cas9 structure", "abstract": "Domain mechanism and guide RNA specificity."}]
    )
    assert len(result["data"]["excluded_records"]) == 0
    assert result["data"]["included_records"] or result["data"]["needs_manual_scope_review"]


def test_broad_corpus_no_download_by_default():
    result = run_crispr_broad_corpus_agent(
        year_from=2025,
        year_to=2026,
        download=False,
        allow_download=False,
        yes=False,
        max_results_per_query=2,
        max_downloads=0,
        exhaustive_mode=False,
        include_reviews=True,
        output_dir="agent_runs/test_broad_no_download_default",
    )
    assert result.get("actual_downloads", 0) == 0


def test_download_gate_still_strict():
    no_evidence = {"title": "No evidence", "legal_pdf_url": "https://example.org/a.pdf"}
    legal = {
        "title": "Legal OA",
        "legal_pdf_url": "https://example.org/a.pdf",
        "legality_decision": "allowed_for_future_download",
        "license": "cc-by",
        "evidence_sources": ["europe_pmc"],
    }
    assert tr.download_legal_pdf_tool(no_evidence, allow_download=True, yes=True)["status"] == "blocked"
    assert tr.download_legal_pdf_tool(legal, allow_download=False, yes=True)["status"] == "blocked"
    assert tr.download_legal_pdf_tool(legal, allow_download=True, yes=False)["status"] == "blocked"
