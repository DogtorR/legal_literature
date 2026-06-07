import json
from pathlib import Path

from lit_agent.mcp_server import calibrate_openalex_oa_counts as mcp_calibrate_openalex_oa_counts
from lit_agent.openalex_calibration import calibrate_openalex_oa_counts


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, _url, params=None, **_kwargs):
        params = dict(params or {})
        self.calls.append(params)
        filters = params.get("filter", "")
        if params.get("cursor"):
            return _Response(
                {
                    "meta": {"count": 2, "next_cursor": None},
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "doi": "https://doi.org/10.1000/a",
                            "title": "CRISPR detection assay",
                            "publication_year": 2025,
                            "open_access": {"is_oa": True, "oa_url": "https://example.org/a"},
                            "best_oa_location": {"pdf_url": "https://example.org/a.pdf", "license": "cc-by", "landing_page_url": "https://example.org/a"},
                            "locations": [],
                        },
                        {
                            "id": "https://openalex.org/W2",
                            "title": "Cas13 diagnostics",
                            "publication_year": 2026,
                            "open_access": {"is_oa": True, "oa_url": "https://example.org/b"},
                            "best_oa_location": {"landing_page_url": "https://example.org/b", "license": None},
                            "locations": [],
                        },
                    ],
                }
            )
        if "open_access.is_oa:true" in filters:
            return _Response({"meta": {"count": 2}, "results": []})
        return _Response({"meta": {"count": 5}, "results": []})


def test_calibrate_openalex_oa_counts_returns_counts_and_report(tmp_path: Path) -> None:
    session = _Session()
    result = calibrate_openalex_oa_counts(
        "CRISPR detection",
        year_from=2025,
        year_to=2026,
        output_dir=str(tmp_path),
        session=session,
    )
    assert result["total_count"] == 5
    assert result["oa_count"] == 2
    assert result["oa_ratio"] == 0.4
    assert result["has_oa_url_count"] == 2
    assert result["has_pdf_url_count"] == 1
    assert result["has_landing_page_count"] == 2
    assert result["has_license_count"] == 1
    assert result["query_used"] == "CRISPR detection"
    assert (tmp_path / "openalex_oa_calibration_2025_2026.json").exists()
    assert (tmp_path / "openalex_oa_calibration_report.md").exists()
    assert not (tmp_path / "downloads").exists()


def test_calibrate_openalex_oa_counts_passes_year_filters(tmp_path: Path) -> None:
    session = _Session()
    calibrate_openalex_oa_counts(
        "CRISPR detection",
        year_from=2025,
        year_to=2026,
        output_dir=str(tmp_path),
        session=session,
        max_scan_pages=1,
    )
    filters = [call.get("filter", "") for call in session.calls]
    assert any("from_publication_date:2025-01-01" in item for item in filters)
    assert any("to_publication_date:2026-12-31" in item for item in filters)
    payload = json.loads((tmp_path / "openalex_oa_calibration_2025_2026.json").read_text(encoding="utf-8"))
    assert payload["filters_used"]["year_from"] == 2025
    assert payload["filters_used"]["year_to"] == 2026


def test_mcp_calibrate_openalex_oa_counts_tool_exists() -> None:
    assert callable(mcp_calibrate_openalex_oa_counts)
