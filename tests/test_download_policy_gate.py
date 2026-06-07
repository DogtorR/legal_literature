from lit_agent import tool_registry as tr
from lit_agent.agent_graph import run_literature_graph_agent


def test_download_tool_blocks_without_allow_download():
    result = tr.download_legal_pdf_tool({"legality_decision": "allowed_for_future_download"}, allow_download=False, yes=True)
    assert result["status"] == "blocked"


def test_download_tool_blocks_without_yes():
    result = tr.download_legal_pdf_tool({"legality_decision": "allowed_for_future_download"}, allow_download=True, yes=False)
    assert result["status"] == "blocked"


def test_download_tool_requires_legal_oa_evidence():
    result = tr.download_legal_pdf_tool({"pdf_url": "https://example.org/test.pdf"}, allow_download=True, yes=True)
    assert result["status"] == "blocked"


def test_download_tool_can_execute_only_after_policy_and_oa_evidence(monkeypatch):
    calls = []

    class FakeResult:
        def to_dict(self):
            return {"status": "blocked", "reason": "fake_no_network"}

    def fake_execute(requests, *, dry_run=True, allow_download=False, session=None):
        calls.append({"requests": requests, "dry_run": dry_run, "allow_download": allow_download, "session": session})
        return [FakeResult()]

    monkeypatch.setattr(tr, "execute_download_plan", fake_execute)
    result = tr.download_legal_pdf_tool(
        {
            "title": "OA paper",
            "source": "europe_pmc",
            "pdf_url": "https://europepmc.org/articles/test.pdf",
            "legality_decision": "allowed_for_future_download",
            "license": "cc-by",
            "evidence_sources": ["europe_pmc"],
        },
        allow_download=True,
        yes=True,
    )
    assert calls
    assert calls[0]["dry_run"] is False
    assert calls[0]["allow_download"] is True
    assert result["status"] == "partial"


def test_graph_blocks_download_without_allow_download():
    result = run_literature_graph_agent(
        "Find legal open-access CRISPR detection papers, max 3, download.",
        allow_network_metadata=False,
        allow_network_rank=False,
        download=True,
        allow_download=False,
        yes=False,
        max_results=3,
        max_downloads=3,
        use_llm_diagnosis=False,
        output_dir="agent_runs/test_gate_no_allow",
    )
    assert result["blocked"] is True
    assert result["actual_downloads"] == 0


def test_graph_blocks_or_skips_download_without_yes():
    result = run_literature_graph_agent(
        "Find legal open-access CRISPR detection papers, max 3, download.",
        allow_network_metadata=False,
        allow_network_rank=False,
        download=True,
        allow_download=True,
        yes=False,
        max_results=3,
        max_downloads=3,
        use_llm_diagnosis=False,
        output_dir="agent_runs/test_gate_no_yes",
    )
    assert result["blocked"] is True
    assert result["actual_downloads"] == 0
