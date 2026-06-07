import json

from lit_agent.cli import main
from lit_agent.downloader import execute_download_plan, plan_downloads_from_legality_audit


def test_plan_downloads_from_legality_audit_and_manifest_status(tmp_path, monkeypatch):
    audit = tmp_path / "legality_audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "decision": "allowed_for_future_download",
                "source": "unpaywall",
                "pdf_url_candidate": "https://example.org/a.pdf",
                "landing_url": "https://example.org/a",
                "evidence_sources": ["unpaywall"],
                "license": "cc-by",
                "doi": "10.1000/a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    requests = plan_downloads_from_legality_audit(audit)
    results = execute_download_plan(requests, dry_run=True, allow_download=False)
    assert results[0].status == "skipped_dry_run"
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest[0]["status"] == "planned_dry_run"
    assert manifest[0]["status"] != "downloaded"
    plan = [json.loads(line) for line in (tmp_path / "download_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    assert plan[0]["planned_status"] == "planned_dry_run"


def write_allowed_audit(path):
    path.write_text(
        json.dumps(
            {
                "decision": "allowed_for_future_download",
                "source": "unpaywall",
                "pdf_url_candidate": "https://example.org/a.pdf",
                "landing_url": "https://example.org/a",
                "evidence_sources": ["unpaywall"],
                "license": "cc-by",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_plan_and_download_dry_runs_do_not_download(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_allowed_audit(tmp_path / "legality_audit.jsonl")
    assert main(["--plan-downloads", "--dry-run", "--max-downloads", "1"]) == 0
    assert '"downloads": 0' in capsys.readouterr().out
    assert main(["--download", "--dry-run", "--max-downloads", "1"]) == 0
    assert '"downloads": 0' in capsys.readouterr().out
    assert main(["--download", "--max-downloads", "1"]) == 0
    assert '"allow_download": false' in capsys.readouterr().out
