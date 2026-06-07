from pathlib import Path

from lit_agent.agent_graph import build_agent_graph, run_literature_graph_agent


def test_agent_graph_imports_and_builds():
    graph = build_agent_graph()
    assert graph is not None
    assert callable(run_literature_graph_agent)


def test_agent_graph_contains_required_nodes():
    src = Path("lit_agent/agent_graph.py").read_text(encoding="utf-8")
    for required in [
        "StateGraph",
        "understand_request_node",
        "route_node",
        "search_metadata_node",
        "audit_oa_node",
        "plan_download_node",
        "human_or_policy_gate_node",
        "download_node",
        "diagnose_node",
        "report_node",
        "finish_node",
    ]:
        assert required in src
