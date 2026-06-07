import csv

from lit_agent.agent_task import parse_task, write_task_config
from lit_agent.llm_nodes import understand_request_with_llm
from lit_agent.llm_config import AgentConfig
from datetime import date


def test_crispr_detection_task_uses_conservative_keywords():
    task = parse_task("帮我下载2024-2025年CRISPR检测文献", AgentConfig(), llm_client=None)
    assert task.topic == "CRISPR detection"
    assert task.topic_guard["enabled"] is True
    assert task.topic_guard["type"] == "crispr_detection"
    assert "CRISPR detection" in task.keywords
    assert "detection" not in [keyword.lower() for keyword in task.keywords]


def test_specific_journal_and_month_are_strict_scope():
    task = parse_task(
        "帮我下载2026年5月发表在Nature Communications上关于CRISPR检测的合法开放获取文献",
        AgentConfig(default_year_from=2024, default_year_to=2025),
        llm_client=None,
    )

    assert task.journal_filter_mode == "explicit_journals"
    assert task.journal_names == ["Nature Communications"]
    assert task.year_from == 2026
    assert task.year_to == 2026
    assert task.publication_date_from == "2026-05-01"
    assert task.publication_date_to == "2026-05-31"


def test_this_year_science_advances_month_is_strict_scope():
    task = parse_task(
        "帮我下载今年5月发表在Science Advances上关于CRISPR检测的合法开放获取文献",
        AgentConfig(default_year_from=2024, default_year_to=2025),
        llm_client=None,
    )
    year = date.today().year
    assert task.journal_filter_mode == "explicit_journals"
    assert task.journal_names == ["Science Advances"]
    assert task.year_from == year
    assert task.year_to == year
    assert task.publication_date_from == f"{year}-05-01"
    assert task.publication_date_to == f"{year}-05-31"


def test_nature_family_dna_nanostructure_task_expands_topic_and_queries(tmp_path):
    task = parse_task(
        "download 2026 Nature family DNA nanostructure legal OA papers",
        AgentConfig(default_year_from=2024, default_year_to=2025),
        llm_client=None,
    )

    assert task.topic == "DNA nanostructures"
    assert task.keywords == ["DNA"]
    assert "DNA origami" in task.include_terms
    assert task.journal_filter_mode == "journal_family"
    assert "Nature Nanotechnology" in task.journal_names
    assert "crossref" not in task.sources

    _, queries_path = write_task_config(task, tmp_path)
    with queries_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    journals = {row["journal"] for row in rows}
    assert "Nature" in journals
    assert "Nature Communications" in journals
    assert "Nature Nanotechnology" in journals
    assert "Nature Chemistry" in journals


def test_llm_node_parsed_dna_nanostructure_task_uses_fast_sources(monkeypatch):
    class FakeClient:
        def complete(self, *_args, **_kwargs):
            return (
                '{"topic":"DNA nanostructures","keywords":["DNA nanostructures"],'
                '"include_terms":["DNA origami"],"journal_filter_mode":"journal_family",'
                '"journal_names":["Nature","Nature Communications"],"year_from":2026,"year_to":2026}'
            )

    def fake_build_codex_client(**_kwargs):
        return AgentConfig(default_year_from=2024, default_year_to=2025), FakeClient()

    monkeypatch.setattr("lit_agent.llm_nodes.build_codex_client", fake_build_codex_client)
    state = understand_request_with_llm({"user_request": "download Nature family papers", "options": {}})
    task = state["task"]

    assert task["topic"] == "DNA nanostructures"
    assert task["keywords"] == ["DNA"]
    assert "crossref" not in task["sources"]
