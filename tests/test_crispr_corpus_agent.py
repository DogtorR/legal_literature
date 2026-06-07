from pathlib import Path

from lit_agent.agent_graph import query_expansion_node, run_crispr_detection_corpus_agent, run_literature_graph_agent
from lit_agent import tool_registry as tr


def test_query_expansion_crispr_detection():
    state = query_expansion_node({"user_request": "Collect CRISPR detection papers", "task": {"keywords": []}})
    query_list = state["query_list"]
    for query in ["CRISPR detection", "Cas12 detection", "Cas13 detection", "SHERLOCK", "DETECTR", "HOLMES"]:
        assert query in query_list


def test_deduplicate_records():
    records = [
        {"doi": "10.1000/ABC", "title": "CRISPR detection assay", "source": "openalex", "query": "CRISPR detection"},
        {"doi": "https://doi.org/10.1000/abc", "title": "CRISPR detection assay", "source": "pubmed", "query": "Cas12 detection"},
        {"pmid": "123", "title": "Cas13 detection", "source": "crossref"},
        {"pmid": "123", "title": "Cas13 detection duplicate", "source": "europe_pmc"},
        {"title": "SHERLOCK nucleic acid detection", "source": "openalex"},
        {"title": "SHERLOCK nucleic acid detection", "source": "pubmed"},
    ]
    result = tr.deduplicate_records_tool(records)
    unique = result["data"]["unique_records"]
    assert result["status"] == "ok"
    assert len(unique) == 3
    doi_record = next(item for item in unique if item.get("doi") == "10.1000/abc")
    assert set(doi_record["seen_sources"]) == {"openalex", "pubmed"}


def test_topic_guard_crispr_detection():
    records = [
        {"title": "CRISPR Cas12 detection assay for viral nucleic acid", "abstract": "A biosensor assay."},
        {"title": "Generic glucose sensor", "abstract": "No gene editing term."},
        {"title": "CRISPR detection editorial", "publication_type": "editorial"},
    ]
    result = tr.topic_guard_records_tool(records)
    assert result["status"] == "ok"
    assert len(result["data"]["topic_records"]) == 1
    assert len(result["data"]["excluded_records"]) == 2


def test_oa_audit_schema():
    records = [
        {
            "title": "CRISPR Cas12 detection assay",
            "doi": "10.0000/test",
            "source": "europe_pmc",
            "pmcid": "PMC123",
            "landing_url": "https://europepmc.org/article/PMC/PMC123",
        }
    ]
    result = tr.audit_oa_records_tool(records, allow_network=False)
    audited = result["data"]["oa_audit_records"]
    assert audited
    for key in ["is_oa", "legal_pdf_url", "license", "evidence", "can_download"]:
        assert key in audited[0]


def test_download_policy_strict():
    legal_request = {
        "title": "OA paper",
        "legal_pdf_url": "https://example.org/paper.pdf",
        "legality_decision": "allowed_for_future_download",
        "license": "cc-by",
        "evidence_sources": ["europe_pmc"],
    }
    no_evidence = {"title": "No OA evidence", "legal_pdf_url": "https://example.org/paper.pdf"}
    assert tr.download_legal_pdf_tool(legal_request, allow_download=False, yes=True)["status"] == "blocked"
    assert tr.download_legal_pdf_tool(legal_request, allow_download=True, yes=False)["status"] == "blocked"
    assert tr.download_legal_pdf_tool(no_evidence, allow_download=True, yes=True)["status"] == "blocked"
    graph_result = run_literature_graph_agent(
        "Collect CRISPR detection papers. Do not download.",
        allow_network_metadata=False,
        download=False,
        allow_download=False,
        yes=False,
        max_results=2,
        max_downloads=0,
        use_llm_diagnosis=False,
        output_dir="agent_runs/test_download_false_strict",
    )
    assert graph_result["actual_downloads"] == 0


def test_corpus_export_files(tmp_path):
    state = {
        "unique_records": [{"doi": "10.1/a", "title": "CRISPR detection assay", "source": "openalex"}],
        "oa_audit_records": [{"doi": "10.1/a", "title": "CRISPR detection assay", "is_oa": False, "failure_reason": "no_oa"}],
        "download_plan": [],
        "download_results": [],
        "failures": [{"reason": "no_oa"}],
        "excluded_records": [],
        "coverage": {"status": "ok"},
        "query_list": ["CRISPR detection"],
    }
    manifest = tr.export_corpus_manifest_tool(state, output_dir=str(tmp_path))
    coverage = tr.write_coverage_report_tool(state, output_dir=str(tmp_path))
    assert manifest["status"] == "ok"
    assert coverage["status"] == "ok"
    for name in ["metadata_all.jsonl", "failures.jsonl", "coverage_report.md"]:
        assert (tmp_path / name).exists()


def test_graph_corpus_dry_run():
    result = run_crispr_detection_corpus_agent(
        year_from=2025,
        year_to=2026,
        download=False,
        allow_download=False,
        yes=False,
        max_results_per_query=2,
        max_downloads=0,
        exhaustive_mode=False,
        output_dir="agent_runs/test_graph_corpus_dry_run",
    )
    assert isinstance(result, dict)
    assert "status" in result
    assert result.get("actual_downloads", 0) == 0
