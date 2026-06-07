from __future__ import annotations

import json
from pathlib import Path

from lit_agent.acquisition_request import parse_acquisition_request_rules_fallback
from lit_agent.agent_graph import run_literature_acquisition_agent
from lit_agent.count_summary import debug_openalex_count_query, summarize_literature_counts


class MockResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class OpenAlex429Session:
    def get(self, *args, **kwargs):
        return MockResponse(429, {})


class PubMedCountSession:
    def get(self, *args, **kwargs):
        return MockResponse(200, {"esearchresult": {"count": "12345", "idlist": []}})


class OpenAlexCountSession:
    def get(self, url, params=None, **kwargs):
        params = params or {}
        is_oa = "open_access.is_oa:true" in str(params.get("filter", ""))
        search_mode = "search" in params
        base = 110 if search_mode else 100
        return MockResponse(200, {"meta": {"count": 25 if is_oa else base}, "results": []})


def test_count_intent_detection():
    parsed = parse_acquisition_request_rules_fallback("有多少 CRISPR 文献")
    assert parsed.intent == "count_summary"


def test_count_summary_does_not_start_collection(monkeypatch, tmp_path):
    called = {"collection": 0, "fulltext": 0}

    def fake_collection(**kwargs):
        called["collection"] += 1
        return {"status": "ok", "data": {}, "artifacts": {}}

    def fake_fulltext(**kwargs):
        called["fulltext"] += 1
        return {"status": "ok", "data": {}, "artifacts": {}}

    monkeypatch.setattr("lit_agent.tool_registry.run_full_collection_loop_tool", fake_collection)
    monkeypatch.setattr("lit_agent.tool_registry.collect_approved_legal_fulltexts_tool", fake_fulltext)
    monkeypatch.setattr(
        "lit_agent.agent_graph.summarize_literature_counts",
        lambda **kwargs: {"status": "ok", "intent": "count_summary", "actual_downloads": 0, "artifacts": {}},
    )
    result = run_literature_acquisition_agent("有多少 CRISPR 文献", allow_download=True, yes=True, output_dir=str(tmp_path))
    assert result["intent"] == "count_summary"
    assert result["actual_downloads"] == 0
    assert called == {"collection": 0, "fulltext": 0}


def test_local_summary_counts_manifest_files(tmp_path):
    (tmp_path / "corpus_manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (tmp_path / "metadata_only_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "approved_for_download.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    (tmp_path / "collected_fulltexts_manifest.jsonl").write_text(
        json.dumps({"full_text_type": "pdf"}) + "\n" + json.dumps({"full_text_type": "europe_pmc_xml"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "failed_fulltext_fetches.jsonl").write_text("{}\n", encoding="utf-8")
    result = summarize_literature_counts("CRISPR", include_openalex=False, include_pubmed=False, output_dir=str(tmp_path))
    assert result["local"]["corpus_manifest_records"] == 2
    assert result["local"]["collected_fulltexts_count"] == 2
    assert result["local"]["pdf_count"] == 1
    assert result["local"]["xml_count"] == 1
    assert Path(result["artifacts"]["literature_count_summary_report"]).exists()


def test_openalex_rate_limited_not_zero(tmp_path):
    result = summarize_literature_counts(
        "CRISPR",
        include_pubmed=False,
        include_local=False,
        output_dir=str(tmp_path),
        openalex_session=OpenAlex429Session(),
    )
    assert result["openalex"]["status"] == "rate_limited"
    assert result["openalex"]["total_count"] is None
    assert result["openalex"]["oa_count"] is None


def test_debug_openalex_count_query_modes():
    query = "(CRISPR OR Cas12) AND detection"
    search = debug_openalex_count_query(query, mode="search", session=OpenAlexCountSession())
    title_abs = debug_openalex_count_query(query, mode="title_and_abstract_filter", session=OpenAlexCountSession())
    assert search["status"] == "ok"
    assert title_abs["status"] == "ok"
    assert search["params"]["search"] == query
    assert "title_and_abstract.search" in title_abs["params"]["filter"]
    assert search["total_count"] == 110
    assert title_abs["total_count"] == 100
    assert search["api_url"].startswith("https://api.openalex.org/works?")


def test_count_summary_report_includes_openalex_debug_params(tmp_path):
    result = summarize_literature_counts(
        "CRISPR",
        include_pubmed=False,
        include_local=False,
        output_dir=str(tmp_path),
        openalex_session=OpenAlexCountSession(),
    )
    report = Path(result["artifacts"]["literature_count_summary_report"]).read_text(encoding="utf-8")
    assert "OpenAlex Query Alignment Debug" in report
    assert "search parameter mode" in report
    assert "title_and_abstract.search filter mode" in report
    assert "api_url:" in report
    assert "params:" in report


def test_pubmed_count_query(tmp_path):
    result = summarize_literature_counts(
        "CRISPR",
        include_openalex=False,
        include_local=False,
        output_dir=str(tmp_path),
        pubmed_session=PubMedCountSession(),
    )
    assert result["pubmed"]["status"] == "ok"
    assert result["pubmed"]["total_count"] == 12345


def test_run_literature_acquisition_agent_count_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lit_agent.agent_graph.summarize_literature_counts",
        lambda **kwargs: {
            "status": "ok",
            "intent": "count_summary",
            "openalex": {"status": "ok", "total_count": 100, "oa_count": 25, "oa_ratio": 0.25},
            "pubmed": {"status": "ok", "total_count": 80},
            "local": {"status": "ok", "corpus_manifest_records": 2},
            "actual_downloads": 0,
            "artifacts": {},
        },
    )
    result = run_literature_acquisition_agent("OpenAlex 上有多少 CRISPR 文献，PubMed 上有多少，本地有多少", output_dir=str(tmp_path))
    assert result["intent"] == "count_summary"
    assert result["openalex"]["total_count"] == 100
    assert result["pubmed"]["total_count"] == 80
    assert result["actual_downloads"] == 0
