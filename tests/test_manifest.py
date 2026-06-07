import json
from datetime import datetime

from lit_agent.manifest import append_jsonl, read_jsonl, write_failure, write_search_log


def test_append_jsonl_writes_valid_json_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_jsonl(path, {"status": "skipped", "input_id": "q001"})
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["status"] == "skipped"
    assert read_jsonl(path)[0]["input_id"] == "q001"


def test_append_jsonl_serializes_supported_non_json_types(tmp_path):
    path = tmp_path / "search_log.jsonl"
    append_jsonl(path, {"time": datetime(2026, 1, 2, 3, 4), "path": tmp_path, "items": {"b", "a"}})
    record = read_jsonl(path)[0]
    assert record["time"] == "2026-01-02T03:04:00"
    assert record["items"] == ["a", "b"]


def test_failure_and_search_log_helpers(tmp_path):
    failures = tmp_path / "failures.jsonl"
    search_log = tmp_path / "search_log.jsonl"
    write_failure("bad input", {"field": "year_from"}, path=failures)
    write_search_log("cli_dry_run", {"downloads": 0}, path=search_log)
    assert read_jsonl(failures)[0]["reason"] == "bad input"
    assert read_jsonl(search_log)[0]["event"] == "cli_dry_run"
