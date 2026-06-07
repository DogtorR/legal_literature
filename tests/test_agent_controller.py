from lit_agent.agent_controller import ControllerOptions, run_agent_controller
from lit_agent.agent_task import LiteratureTask


def _summary(metadata_written=0, planned=0, downloads=0):
    steps = [
        {
            "step": "metadata_rank",
            "summary": {
                "metadata_results_written": metadata_written,
                "topic_guard_removed_count": 0,
                "rank_removed_count": 0,
                "rank_input_count": metadata_written,
            },
        },
        {
            "step": "legality_check",
            "summary": {"allowed_for_future_download": planned},
        },
        {
            "step": "download_plan",
            "summary": {"planned_downloads": planned},
        },
    ]
    if downloads:
        steps.append({"step": "download", "summary": {"actual_downloads": downloads}})
    return {"request": "x", "steps": steps}


def test_controller_retries_with_larger_max_results(monkeypatch):
    calls = []

    def fake_run_pipeline(task, *, request_text, options):
        calls.append(options.max_results)
        if len(calls) == 1:
            return _summary(metadata_written=0)
        return _summary(metadata_written=3, planned=1)

    monkeypatch.setattr("lit_agent.agent_controller.run_pipeline", fake_run_pipeline)
    result = run_agent_controller(
        LiteratureTask(original_request="x", topic="CRISPR detection", max_results=5),
        request_text="x",
        options=ControllerOptions(max_results=5, max_attempts=2, allow_network_metadata=True),
    )
    assert result["attempt_count"] == 2
    assert calls == [5, 15]
    assert result["final_diagnosis"]["planned_downloads"] == 1


def test_controller_stops_when_download_succeeds(monkeypatch):
    calls = []

    def fake_run_pipeline(task, *, request_text, options):
        calls.append(options.max_results)
        return _summary(metadata_written=3, planned=1, downloads=1)

    monkeypatch.setattr("lit_agent.agent_controller.run_pipeline", fake_run_pipeline)
    result = run_agent_controller(
        LiteratureTask(original_request="x", topic="CRISPR detection", max_results=5),
        request_text="x",
        options=ControllerOptions(max_results=5, max_attempts=3, allow_network_metadata=True),
    )
    assert result["attempt_count"] == 1
    assert calls == [5]
    assert result["final_diagnosis"]["actual_downloads"] == 1
