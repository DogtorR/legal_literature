import json

from lit_agent.cli import main
from lit_agent.metadata import deduplicate_records
from lit_agent.sources import unpaywall


def test_check_records_splits_missing_doi_and_doi_candidates():
    checked, candidates, info = unpaywall.check_records(
        [{"doi": "10.1000/a", "title": "A"}, {"title": "No DOI"}],
        {},
        dry_run=True,
        max_results=2,
    )
    assert len(checked) == 1
    assert len(candidates) == 2
    assert info["checked_doi_count"] == 1
    assert info["missing_doi_count"] == 1
    assert any(candidate["reason"] == "missing_doi_for_unpaywall_check" for candidate in candidates)


def test_unpaywall_result_merges_into_metadata_record():
    unique = deduplicate_records(
        [
            {"doi": "10.1000/a", "title": "A", "source": "openalex", "sources": ["openalex"]},
            {"doi": "10.1000/a", "source": "unpaywall", "sources": ["unpaywall"], "unpaywall_checked": True, "unpaywall_is_oa": True},
        ]
    )
    assert len(unique) == 1
    assert unique[0]["sources"] == ["openalex", "unpaywall"]
    assert unique[0]["unpaywall_checked"] is True


def test_cli_oa_check_dry_run_does_not_download(capsys):
    assert main(["--config", "examples/search_config.yaml", "--source", "unpaywall", "--oa-check", "--metadata-only", "--dry-run", "--max-results", "1"]) == 0
    output = capsys.readouterr().out
    assert '"oa_check": true' in output
    assert '"downloads": 0' in output


def test_cli_oa_check_applies_topic_guard_to_existing_candidates(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project_name: test",
                "query_name: crispr_detection",
                "keywords: [CRISPR detection]",
                "include_terms: [CRISPR, detection]",
                "sources: [unpaywall]",
                "max_results_per_source: 10",
                "dry_run: true",
                "topic_guard:",
                "  enabled: true",
                "  type: crispr_detection",
            ]
        ),
        encoding="utf-8",
    )
    candidates = [
        {"query_id": "q", "title": "CRISPR Cas12 assay for viral detection", "doi": "10.1000/on", "source": "openalex"},
        {"query_id": "q", "title": "Mitochondria serve as a holdout compartment", "doi": "10.1000/off", "source": "openalex"},
    ]
    (tmp_path / "candidates.jsonl").write_text("\n".join(json.dumps(record) for record in candidates) + "\n", encoding="utf-8")

    assert main(["--config", str(config), "--source", "unpaywall", "--oa-check", "--metadata-only", "--dry-run", "--max-results", "10"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["oa_checked"] == 1
    assert output["oa_topic_guard_removed_count"] == 1
