from pathlib import Path

import pytest

from lit_agent.config import load_config, load_search_config, normalize_config


ROOT = Path(__file__).resolve().parents[1]


def test_search_config_loads_defaults_and_fields():
    config = load_search_config(ROOT / "examples" / "search_config.yaml")
    assert config["dry_run"] is True
    assert config["download_policy"] == "legal_oa_only"
    assert "openalex" in config["sources"]
    assert config["year_from"] == 2020
    assert config["year_to"] == 2025


def test_load_config_alias_and_defaults():
    config = load_config(ROOT / "examples" / "search_config.yaml")
    assert config["max_downloads"] == 0
    assert config["sources"] == ["openalex", "europe_pmc", "pubmed"]


def test_invalid_year_range_raises():
    with pytest.raises(ValueError):
        normalize_config({"year_from": 2025, "year_to": 2020})


def test_invalid_source_raises():
    with pytest.raises(ValueError):
        normalize_config({"sources": ["openalex", "unknown_source"]})


def test_forbidden_bypass_field_raises():
    with pytest.raises(ValueError):
        normalize_config({"use_cookie": True})


def test_mapping_download_policy_normalizes_to_safe_policy():
    config = normalize_config(
        {
            "download_policy": {
                "only_legal_open_access": True,
                "uncertain_to_candidates": True,
                "require_license_or_oa_evidence": True,
                "downloads_enabled": False,
            }
        }
    )
    assert config["download_policy"] == "legal_oa_only"
    assert config["download_policy_details"]["downloads_enabled"] is False


def test_simple_yaml_download_policy_fields_are_merged():
    config = normalize_config(
        {
            "download_policy": [],
            "only_legal_open_access": True,
            "uncertain_to_candidates": True,
            "require_license_or_oa_evidence": True,
            "downloads_enabled": False,
        }
    )
    assert config["download_policy"] == "legal_oa_only"
    assert config["download_policy_details"]["require_license_or_oa_evidence"] is True
