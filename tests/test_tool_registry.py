from lit_agent import tool_registry as tr


def test_tool_registry_contains_standard_tools():
    for name in [
        "search_openalex_tool",
        "search_pubmed_tool",
        "search_crossref_tool",
        "search_europe_pmc_tool",
        "check_unpaywall_tool",
        "check_europe_pmc_oa_tool",
        "plan_legal_downloads_tool",
        "download_legal_pdf_tool",
        "write_manifest_tool",
        "write_report_tool",
    ]:
        assert hasattr(tr, name)
