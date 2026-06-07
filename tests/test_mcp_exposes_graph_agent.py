from pathlib import Path


def test_mcp_exposes_low_level_tools_and_graph_agent():
    src = Path("lit_agent/mcp_server.py").read_text(encoding="utf-8")
    assert "run_literature_graph_agent" in src
    assert "run_literature_graph_agent_tool" in src
    for tool in [
        "search_openalex",
        "search_pubmed",
        "search_crossref",
        "search_europe_pmc",
        "check_unpaywall",
        "download_legal_pdf",
    ]:
        assert tool in src
    for banned in ["llm_client.complete", ".complete(", "build_llm_client(", "load_llm_config("]:
        assert banned not in src
