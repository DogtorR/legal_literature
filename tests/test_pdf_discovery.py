from lit_agent.pdf_discovery import discover_direct_pdf_url


def test_unpaywall_url_for_pdf_discovered():
    result = discover_direct_pdf_url(
        {
            "source": "unpaywall",
            "sources": ["unpaywall"],
            "url_for_pdf": "https://example.org/article.pdf",
            "license": "cc-by",
            "is_legal_oa_candidate": True,
        }
    )
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://example.org/article.pdf"


def test_nature_reference_pdf_is_derived_from_unpaywall_doi():
    result = discover_direct_pdf_url(
        {
            "source": "unpaywall",
            "sources": ["unpaywall"],
            "doi": "10.1038/s41467-026-73286-8",
            "landing_url": "https://doi.org/10.1038/s41467-026-73286-8",
            "license": "cc-by",
            "is_legal_oa_candidate": True,
        }
    )
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://www.nature.com/articles/s41467-026-73286-8_reference.pdf"


def test_nature_reference_pdf_is_derived_from_landing_url():
    result = discover_direct_pdf_url(
        {
            "source": "publisher_explicit_oa",
            "sources": ["publisher_explicit_oa"],
            "landing_url": "https://www.nature.com/articles/s41467-026-72685-1",
            "license": "cc-by",
        }
    )
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://www.nature.com/articles/s41467-026-72685-1_reference.pdf"


def test_europe_pmc_pdf_url_discovered():
    result = discover_direct_pdf_url(
        {
            "source": "europe_pmc",
            "sources": ["europe_pmc"],
            "pdf_url_candidate": "https://europepmc.org/articles/PMC123?pdf=render",
            "pmcid": "PMC123",
            "oa_status": "open",
        }
    )
    assert result.status == "direct_pdf_url_found"


def test_europe_pmc_pdf_render_url_counts_as_direct_pdf():
    result = discover_direct_pdf_url(
        {
            "source": "unpaywall",
            "sources": ["unpaywall"],
            "pdf_url_candidate": "https://europepmc.org/articles/PMC123?pdf=render",
            "is_legal_oa_candidate": True,
        }
    )
    assert result.status == "direct_pdf_url_found"


def test_pmcid_official_pdf_url_is_derived_for_pmc_source():
    result = discover_direct_pdf_url({"source": "europe_pmc", "sources": ["europe_pmc"], "pmcid": "PMC123"})
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://europepmc.org/api/getPdf?pmcid=PMC123"


def test_pmcid_pdf_url_is_derived_for_unpaywall_confirmed_oa():
    result = discover_direct_pdf_url({"source": "unpaywall", "sources": ["unpaywall"], "pmcid": "PMC123", "is_legal_oa_candidate": True})
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://europepmc.org/api/getPdf?pmcid=PMC123"


def test_pmcid_pdf_url_is_preferred_over_nature_pdf():
    result = discover_direct_pdf_url(
        {
            "source": "unpaywall",
            "sources": ["unpaywall"],
            "doi": "10.1038/s41565-025-02091-z",
            "url_for_pdf": "https://www.nature.com/articles/s41565-025-02091-z.pdf",
            "pmcid": "PMC12916302",
            "is_legal_oa_candidate": True,
        }
    )
    assert result.status == "direct_pdf_url_found"
    assert result.pdf_url == "https://europepmc.org/api/getPdf?pmcid=PMC12916302"


def test_openalex_only_does_not_discover_pdf():
    result = discover_direct_pdf_url({"source": "openalex", "sources": ["openalex"], "pdf_url_candidate": "https://example.org/a.pdf"})
    assert result.status == "candidate"
    assert result.reason == "metadata_only_source_no_direct_pdf_discovery"


def test_crossref_only_does_not_discover_pdf():
    result = discover_direct_pdf_url({"source": "crossref", "sources": ["crossref"], "pdf_url_candidate": "https://example.org/a.pdf"})
    assert result.status == "candidate"


def test_forbidden_url_is_blocked():
    result = discover_direct_pdf_url({"source": "unpaywall", "sources": ["unpaywall"], "url_for_pdf": "https://sci-hub.example/a.pdf"})
    assert result.status == "blocked"


def test_dry_run_discovery_does_not_call_network():
    class FailingSession:
        def head(self, *args, **kwargs):
            raise AssertionError("network should not be called")

        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    result = discover_direct_pdf_url(
        {"source": "unpaywall", "sources": ["unpaywall"], "url_for_pdf": "https://example.org/a.pdf"},
        dry_run=True,
        allow_network=False,
        session=FailingSession(),
    )
    assert result.status == "direct_pdf_url_found"
