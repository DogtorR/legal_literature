import json
from pathlib import Path

from lit_agent.cli import main
from lit_agent.manifest import write_legality_audit


def test_cli_legality_check_does_not_download(capsys):
    assert main(["--config", "examples/search_config.yaml", "--legality-check", "--metadata-only", "--dry-run", "--max-results", "1"]) == 0
    output = capsys.readouterr().out
    assert '"legality_check": true' in output
    assert '"downloads": 0' in output
    assert Path("legality_audit.jsonl").exists()


def test_legality_audit_jsonl_is_valid(tmp_path):
    path = tmp_path / "legality_audit.jsonl"
    write_legality_audit({"decision": "candidate_needs_confirmation", "doi": "10.1000/a"}, path=path)
    assert json.loads(path.read_text(encoding="utf-8"))["decision"] == "candidate_needs_confirmation"
