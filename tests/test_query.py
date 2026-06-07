from pathlib import Path

from lit_agent.config import load_search_config
from lit_agent.query import merge_cli_overrides, normalize_terms, query_specs_from_config, queries_from_config, queries_from_csv


ROOT = Path(__file__).resolve().parents[1]


def test_queries_csv_normalizes_required_fields():
    specs = queries_from_csv(ROOT / "examples" / "queries.csv")
    assert len(specs) == 1
    spec = specs[0]
    assert spec.input_id == "q001"
    assert spec.year_from == 2020
    assert spec.year_to == 2025
    assert spec.keywords == ["carbapenemase detection"]
    assert "carbapenemase" in spec.include_terms
    assert "review" in spec.exclude_terms
    assert "openalex" in spec.source_preference


def test_config_normalizes_to_query_spec():
    config = load_search_config(ROOT / "examples" / "search_config.yaml")
    spec = query_specs_from_config(config)[0]
    assert spec.input_id == "carbapenemase_detection_dry_run"
    assert spec.year_from == 2020
    assert spec.year_to == 2025
    assert spec.keywords == ["carbapenemase detection"]
    assert "detection" in spec.include_terms


def test_legacy_queries_from_config_alias():
    config = load_search_config(ROOT / "examples" / "search_config.yaml")
    assert queries_from_config(config)[0].input_id == "carbapenemase_detection_dry_run"


def test_normalize_terms_accepts_lists_and_delimiters():
    assert normalize_terms(["a", " b "]) == ["a", "b"]
    assert normalize_terms("a,b") == ["a", "b"]
    assert normalize_terms("a;b|c") == ["a", "b", "c"]


def test_cli_overrides_have_highest_priority():
    spec = queries_from_csv(ROOT / "examples" / "queries.csv")[0]
    overridden = merge_cli_overrides([spec], {"year_from": 2021, "year_to": 2024, "keywords": "override", "max_results": 5})[0]
    assert overridden.year_from == 2021
    assert overridden.year_to == 2024
    assert overridden.keywords == ["override"]
    assert overridden.max_results == 5
