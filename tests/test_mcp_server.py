from pathlib import Path

from lit_agent import mcp_server


def test_mcp_output_listing_uses_known_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("manifest.jsonl").write_text('{"status":"downloaded"}\n', encoding="utf-8")
    Path("downloads").mkdir()
    (Path("downloads") / "paper.pdf").write_bytes(b"%PDF-1.4")

    result = mcp_server.list_literature_agent_outputs()

    assert result["artifact_counts"]["manifest"] == 1
    assert result["downloaded_pdfs"][0]["name"] == "paper.pdf"


def test_parse_tool_can_use_rule_parser(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("llm_config.yaml").write_text(
        """
active_provider: deepseek
providers:
  deepseek:
    type: openai_compatible
    base_url: https://api.deepseek.com
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
agent:
  max_results: 20
  max_downloads: 20
""".strip(),
        encoding="utf-8",
    )

    result = mcp_server.parse_literature_request(
        "Download 2026 Nature Communications CRISPR detection papers",
        no_llm=True,
        max_results=5,
        max_downloads=2,
    )

    assert result["task"]["topic"] == "CRISPR detection"
    assert result["task"]["journal_names"] == ["Nature Communications"]
    assert Path(result["config_path"]).exists()
