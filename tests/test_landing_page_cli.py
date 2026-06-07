import json
from pathlib import Path

from lit_agent.cli import main


def write_config(path: Path):
    path.write_text(
        """
project_name: test
query_name: test_query
keywords:
  - carbapenemase detection
year_from: 2020
year_to: 2025
include_terms:
  - carbapenemase
exclude_terms: []
max_results_per_source: 5
dry_run: true
sources:
  - openalex
download_policy: legal_oa_only
""".strip(),
        encoding="utf-8",
    )


def write_allowed_audit(path: Path):
    path.write_text(
        json.dumps(
            {
                "decision": "allowed_for_future_download",
                "doi": "10.1038/s41586-020-2649-2",
                "title": "A",
                "source": "unpaywall",
                "landing_url": "https://www.nature.com/articles/s41586-020-2649-2",
                "license": "cc-by",
                "evidence_sources": ["unpaywall"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_discover_landing_pages_dry_run_no_download(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    write_config(config)
    write_allowed_audit(tmp_path / "legality_audit.jsonl")
    assert main(["--config", str(config), "--discover-landing-pages", "--discover-pdf-urls", "--metadata-only", "--dry-run", "--max-results", "1"]) == 0
    output = capsys.readouterr().out
    assert '"landing_pages_checked": 1' in output
    assert '"downloads": 0' in output
    if Path("manifest.jsonl").exists():
        assert '"status": "downloaded"' not in Path("manifest.jsonl").read_text(encoding="utf-8")


def test_cli_discover_landing_pages_with_mocked_network_writes_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    write_config(config)
    write_allowed_audit(tmp_path / "legality_audit.jsonl")

    def fake_resolve(record, dry_run=True, allow_network=False):
        from lit_agent.landing_page import LandingDiscoveryResult

        candidate = {
            "doi": record["doi"],
            "title": record["title"],
            "source": "doi_landing_page",
            "candidate_url": "https://www.nature.com/articles/s41586-020-2649-2.pdf",
            "pdf_url_candidate": "https://www.nature.com/articles/s41586-020-2649-2.pdf",
            "landing_url": "https://www.nature.com/articles/s41586-020-2649-2",
            "decision": "doi_landing_candidate_with_existing_oa_evidence",
            "reason": "doi_landing_pdf_with_existing_oa_evidence_not_downloaded_round_011",
            "planned_status": "candidate_direct_pdf_discovered_not_downloaded_round_011",
        }
        return LandingDiscoveryResult(doi=record["doi"], landing_url="https://www.nature.com/articles/s41586-020-2649-2", status="completed", reason="ok", candidates=[candidate], candidate_count=1, with_existing_oa_evidence_count=1, network_calls=1, dry_run=False)

    monkeypatch.setattr("lit_agent.cli.resolve_landing_page", fake_resolve)
    assert main(["--config", str(config), "--discover-landing-pages", "--allow-network-metadata", "--metadata-only", "--dry-run", "--max-results", "1"]) == 0
    output = capsys.readouterr().out
    assert '"candidate_pdf_links_found": 1' in output
    assert Path("download_plan.jsonl").exists()
    assert not Path("manifest.jsonl").exists()
