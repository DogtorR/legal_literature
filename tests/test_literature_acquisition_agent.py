from __future__ import annotations

from pathlib import Path

from lit_agent.acquisition_request import parse_acquisition_request_rules_fallback
from lit_agent.acquisition_request import parse_acquisition_request_with_llm
from lit_agent.agent_graph import run_literature_acquisition_agent
from lit_agent.llm_config import AgentConfig
from lit_agent.query import QuerySpec
from lit_agent.sources import crossref, europe_pmc, openalex, pubmed


def test_parse_all_years_crispr_request():
    parsed = parse_acquisition_request_rules_fallback("抓取所有年份 CRISPR 相关合法开放全文")
    assert parsed.all_years is True
    assert parsed.year_from is None
    assert parsed.year_to is None
    assert parsed.query_profile == "crispr_broad"


def test_parse_year_range_request():
    parsed = parse_acquisition_request_rules_fallback("抓取 2020–2024 年 CRISPR detection 文献")
    assert parsed.year_from == 2020
    assert parsed.year_to == 2024
    assert parsed.query_profile == "crispr_broad"


def test_parse_recent_years_request():
    parsed = parse_acquisition_request_rules_fallback("近两年 Cas12 diagnostics")
    assert parsed.year_from is not None
    assert parsed.year_to is not None
    assert parsed.year_to - parsed.year_from == 1


def test_default_year_filter_is_all_years():
    parsed = parse_acquisition_request_rules_fallback("抓取 CRISPR 相关合法开放全文")
    assert parsed.all_years is True
    assert parsed.year_from is None
    assert parsed.year_to is None


def test_materials_broad_requires_clarification(monkeypatch, tmp_path):
    called = {"collection": False}

    def fake_collection(**kwargs):
        called["collection"] = True
        return {"status": "ok", "data": {"actual_downloads": 0}, "artifacts": {}}

    monkeypatch.setattr("lit_agent.llm_nodes.build_codex_client", lambda *args, **kwargs: (AgentConfig(), None))
    monkeypatch.setattr("lit_agent.tool_registry.run_full_collection_loop_tool", fake_collection)
    result = run_literature_acquisition_agent(
        "抓取所有材料相关文献",
        allow_download=False,
        yes=False,
        output_dir=str(tmp_path),
        max_rounds=1,
        max_batches_per_round=1,
    )
    assert result["status"] == "needs_clarification"
    assert result["clarification_question"]
    assert result["suggested_profiles"]
    assert called["collection"] is False


def test_specific_materials_topic_runs():
    parsed = parse_acquisition_request_rules_fallback("抓取二维材料传感器合法开放全文")
    assert parsed.needs_clarification is True
    assert parsed.query_profile == "llm_generated"
    assert parsed.clarification_question


class FakeLLMClient:
    def __init__(self, payload: str):
        self.payload = payload

    def complete(self, messages, *, response_format=None):
        return self.payload


def test_non_crispr_uses_llm_generated_query(monkeypatch):
    monkeypatch.setattr(
        "lit_agent.llm_nodes.build_codex_client",
        lambda *args, **kwargs: (
            AgentConfig(),
            FakeLLMClient('{"topic":"two-dimensional materials sensors","query_profile":"2d_materials","primary_query":"two-dimensional materials sensor","expansion_queries":["graphene sensor","MXene sensor"],"include_terms":["sensor"]}'),
        ),
    )
    parsed = parse_acquisition_request_with_llm("抓取二维材料传感器合法开放全文")
    assert parsed.query_profile == "llm_generated"
    assert parsed.primary_query == "two-dimensional materials sensor"
    assert parsed.expansion_queries == ["graphene sensor", "MXene sensor"]
    assert parsed.needs_clarification is False


def test_crispr_forces_crispr_broad_even_if_llm_returns_other_profile(monkeypatch):
    monkeypatch.setattr(
        "lit_agent.llm_nodes.build_codex_client",
        lambda *args, **kwargs: (
            AgentConfig(),
            FakeLLMClient('{"topic":"CRISPR detection","query_profile":"crispr_detection","primary_query":"CRISPR detection"}'),
        ),
    )
    parsed = parse_acquisition_request_with_llm("抓取 CRISPR detection 文献")
    assert parsed.query_profile == "crispr_broad"
    assert parsed.primary_query.startswith('(CRISPR OR "CRISPR-Cas"')


def test_crispr_non_gene_editing_rule_adds_exclude_terms():
    parsed = parse_acquisition_request_rules_fallback("\u67e5 CRISPR \u975e\u7eaf\u57fa\u56e0\u7f16\u8f91\u7684\u6587\u732e")
    assert parsed.query_profile == "crispr_broad"
    assert "gene editing" in parsed.exclude_terms
    assert "genome editing" in parsed.exclude_terms
    assert "editorial" in parsed.exclude_terms


def test_crispr_profile_rebuild_preserves_llm_exclude_terms(monkeypatch):
    monkeypatch.setattr(
        "lit_agent.llm_nodes.build_codex_client",
        lambda *args, **kwargs: (
            AgentConfig(),
            FakeLLMClient('{"topic":"CRISPR","query_profile":"crispr_detection","primary_query":"CRISPR diagnostics","exclude_terms":["gene editing","base editing"]}'),
        ),
    )
    parsed = parse_acquisition_request_with_llm("\u67e5 CRISPR \u975e\u7eaf\u57fa\u56e0\u7f16\u8f91\u7684\u6587\u732e")
    assert parsed.query_profile == "crispr_broad"
    assert parsed.primary_query.startswith('(CRISPR OR "CRISPR-Cas"')
    assert "gene editing" in parsed.exclude_terms
    assert "base editing" in parsed.exclude_terms
    assert "prime editing" in parsed.exclude_terms


def test_allow_download_false_no_fulltext(monkeypatch, tmp_path):
    calls = {"fulltext": 0}

    monkeypatch.setattr(
        "lit_agent.tool_registry.run_full_collection_loop_tool",
        lambda **kwargs: {"status": "ok", "data": {"status": "ok", "actual_downloads": 0}, "artifacts": {}},
    )
    monkeypatch.setattr(
        "lit_agent.tool_registry.pre_download_qa_for_legal_oa_candidates_tool",
        lambda output_dir: {"status": "ok", "data": {"approved_for_download": 1}, "artifacts": {}},
    )

    def fake_fulltext(**kwargs):
        calls["fulltext"] += 1
        return {"status": "ok", "data": {"actual_fulltexts": 1}, "artifacts": {}}

    monkeypatch.setattr("lit_agent.tool_registry.collect_approved_legal_fulltexts_tool", fake_fulltext)
    result = run_literature_acquisition_agent("抓取 CRISPR 合法开放全文", allow_download=False, yes=False, output_dir=str(tmp_path), max_rounds=1)
    assert result["actual_downloads"] == 0
    assert calls["fulltext"] == 0


def test_allow_download_true_yes_true_collects_fulltext(monkeypatch, tmp_path):
    calls = {"fulltext": 0}

    monkeypatch.setattr(
        "lit_agent.tool_registry.run_full_collection_loop_tool",
        lambda **kwargs: {"status": "ok", "data": {"status": "ok", "actual_downloads": 0}, "artifacts": {}},
    )
    monkeypatch.setattr(
        "lit_agent.tool_registry.pre_download_qa_for_legal_oa_candidates_tool",
        lambda output_dir: {"status": "ok", "data": {"approved_for_download": 1}, "artifacts": {}},
    )

    def fake_fulltext(**kwargs):
        calls["fulltext"] += 1
        assert kwargs["allow_download"] is True
        assert kwargs["yes"] is True
        return {"status": "ok", "data": {"actual_fulltexts": 1}, "artifacts": {}}

    monkeypatch.setattr("lit_agent.tool_registry.collect_approved_legal_fulltexts_tool", fake_fulltext)
    result = run_literature_acquisition_agent("抓取 CRISPR 合法开放全文", allow_download=True, yes=True, output_dir=str(tmp_path), max_rounds=1)
    assert result["actual_downloads"] == 1
    assert calls["fulltext"] == 1


def test_no_database_annotation_training_pipeline():
    banned = ["annotation_pipeline", "auto_label", "training_data_extraction", "duckdb", "sqlite3", "postgresql"]
    for path in Path("lit_agent").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in banned), path


def test_all_years_source_query_no_year_filter():
    spec = QuerySpec(keywords=["CRISPR"], year_from=None, year_to=None, max_results=5)
    pubmed_params = pubmed.build_pubmed_esearch_params(spec)
    assert "PDAT" not in pubmed_params["term"]
    openalex_params = openalex.build_openalex_params(spec)
    assert "publication_date" not in openalex_params.get("filter", "")
    _url, crossref_params = crossref.build_crossref_request(spec)
    assert "pub-date" not in crossref_params.get("filter", "")
    europe_params = europe_pmc.build_europe_pmc_params(spec)
    assert "FIRST_PDATE" not in europe_params["query"]
