from pathlib import Path


def test_llm_calls_are_not_in_legacy_or_mcp_paths():
    for file, banned_values in {
        "lit_agent/agent_task.py": ["llm_client.complete", 'response_format="json_object"'],
        "lit_agent/agent_controller.py": [".complete("],
        "lit_agent/mcp_server.py": [".complete("],
    }.items():
        src = Path(file).read_text(encoding="utf-8")
        for banned in banned_values:
            assert banned not in src


def test_llm_nodes_expose_explicit_node_adapters():
    src = Path("lit_agent/llm_nodes.py").read_text(encoding="utf-8")
    assert "understand_request_with_llm" in src
    assert "diagnose_with_llm" in src
